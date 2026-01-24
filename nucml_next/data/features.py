"""
Tier-Based Feature Engineering for Nuclear Cross Sections
==========================================================

Implements systematic feature hierarchy based on Valdez 2021 thesis.

Tier System:
------------
- Tier A (Core): Z, A, Energy, MT (4 features + particle-emission vector)
- Tier B (Geometric): + Radius, Radius Ratio (6 features total)
- Tier C (Energetics): + Mass Excess, Binding Energy, Separation Energies (14 features)
- Tier D (Topological): + Spin, Parity, Valence, P-Factor, Isomeric States (20+ features)
- Tier E (Complete): + All reaction Q-values (28+ features)

Particle-Emission Vector (Valdez Table 4.15):
----------------------------------------------
Replaces one-hot MT encoding with physics-aware representation:
    [n_out, p_out, d_out, t_out, He3_out, α_out]

Examples:
    MT=18 (fission): [~2.5, 0, 0, 0, 0, 0]  # Average fission neutrons
    MT=102 (n,γ): [0, 0, 0, 0, 0, 0]        # Only gamma emission
    MT=103 (n,p): [0, 1, 0, 0, 0, 0]        # One proton out
    MT=107 (n,α): [0, 0, 0, 0, 0, 1]        # One alpha out
    MT=16 (n,2n): [2, 0, 0, 0, 0, 0]        # Two neutrons out

This encoding captures reaction similarity and conservation laws.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Nuclear radius constants (Bethe-Weisskopf formula)
R0 = 1.25  # fm (femtometers)


@dataclass
class TierConfig:
    """Configuration for tier-based feature generation."""
    tiers: List[str]
    use_particle_emission: bool = True  # Use particle vector instead of one-hot MT
    include_n_protons: bool = True  # Add N = A - Z


class FeatureGenerator:
    """
    Generate tier-based features for nuclear cross-section data.

    Integrates with AME2020DataEnricher to add nuclear structure properties.
    """

    def __init__(self, enricher=None):
        """
        Initialize feature generator.

        Args:
            enricher: Optional AME2020DataEnricher instance for Tiers B-E
        """
        self.enricher = enricher

    def generate_features(
        self,
        df: pd.DataFrame,
        tiers: List[str] = ['A'],
        use_particle_emission: bool = True
    ) -> pd.DataFrame:
        """
        Generate tier-based features for a dataset.

        Args:
            df: DataFrame with at minimum Z, A, Energy, MT columns
            tiers: List of tiers to include (e.g., ['A', 'B', 'C'])
            use_particle_emission: If True, use particle-emission vector instead of one-hot MT

        Returns:
            DataFrame with generated features
        """
        result = df.copy()

        # Tier A: Core features (always included)
        if 'A' in tiers or len(tiers) == 0:
            result = self._add_tier_a_features(result, use_particle_emission)

        # Tier B: Geometric features
        if 'B' in tiers:
            result = self._add_tier_b_features(result)

        # Tier C: Energetics features
        if 'C' in tiers:
            result = self._add_tier_c_features(result)

        # Tier D: Topological features
        if 'D' in tiers:
            result = self._add_tier_d_features(result)

        # Tier E: Complete Q-values
        if 'E' in tiers:
            result = self._add_tier_e_features(result)

        return result

    def _add_tier_a_features(
        self,
        df: pd.DataFrame,
        use_particle_emission: bool = True
    ) -> pd.DataFrame:
        """
        Add Tier A (Core) features.

        Features:
        - Z: Atomic number
        - A: Mass number
        - N: Neutron number (A - Z)
        - Energy: Incident neutron energy (eV)
        - MT or Particle Emission: Reaction type encoding

        Args:
            df: Input dataframe
            use_particle_emission: Use particle vector instead of MT code

        Returns:
            DataFrame with Tier A features
        """
        result = df.copy()

        # Add N if not present
        if 'N' not in result.columns:
            result['N'] = result['A'] - result['Z']

        # Add particle-emission vector if requested
        if use_particle_emission and 'MT' in result.columns:
            particle_df = self._compute_particle_emission_vector(result['MT'])
            result = pd.concat([result, particle_df], axis=1)

        return result

    def _add_tier_b_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Tier B (Geometric) features.

        Features:
        - R: Nuclear radius (fm) = R0 × A^(1/3)
        - R_ratio: Radius ratio = R_target / R_projectile

        For neutron-induced reactions, R_projectile ≈ 0, so we use R_target only.

        Args:
            df: Input dataframe with Z, A columns

        Returns:
            DataFrame with Tier B features added
        """
        result = df.copy()

        # Nuclear radius: R = R0 × A^(1/3) in femtometers
        result['R_fm'] = R0 * np.power(result['A'], 1.0/3.0)

        # For thermal neutrons, de Broglie wavelength is large (~1.8 fm at 0.025 eV)
        # For fast neutrons, wavelength is small (~0.01 fm at 1 MeV)
        # Compute interaction parameter: k × R (dimensionless)
        # k = 2π/λ = sqrt(2 × m_n × E) / ħ
        # Simplified: k ≈ 0.22 × sqrt(E_MeV) in fm^-1
        E_MeV = result['Energy'] / 1e6  # Convert eV to MeV
        k = 0.22 * np.sqrt(E_MeV)  # Wave number in fm^-1
        result['kR'] = k * result['R_fm']  # Dimensionless

        return result

    def _add_tier_c_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Tier C (Energetics) features.

        Features from AME2020:
        - Mass_Excess_keV: Mass excess in keV
        - Binding_Energy_keV: Total binding energy in keV
        - Binding_Per_Nucleon_keV: Binding energy per nucleon
        - S_1n, S_2n: One- and two-neutron separation energies (keV)
        - S_1p, S_2p: One- and two-proton separation energies (keV)

        Args:
            df: Input dataframe with Z, A columns

        Returns:
            DataFrame with Tier C features enriched from AME2020
        """
        if self.enricher is None:
            logger.warning("No enricher available. Tier C features require AME2020DataEnricher.")
            return df

        result = df.copy()

        # Enrich with AME2020 data (Tier C columns)
        result = self.enricher.enrich_dataframe(result, tiers=['C'])

        # Convert keV to MeV for better numerical stability in ML
        energy_cols = [
            'Mass_Excess_keV', 'Binding_Energy_keV', 'Binding_Per_Nucleon_keV',
            'S_1n', 'S_2n', 'S_1p', 'S_2p'
        ]

        for col in energy_cols:
            if col in result.columns:
                result[f'{col.replace("_keV", "")}_MeV'] = result[col] / 1000.0
                result = result.drop(columns=[col])  # Remove keV version

        return result

    def _add_tier_d_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Tier D (Topological) features.

        Features from NUBASE2020:
        - Spin: Nuclear spin (J)
        - Parity: Parity (+1 or -1)
        - Isomer_Level: Isomeric state (0=ground, 1=first excited, ...)
        - Valence_N: Valence neutrons (distance to nearest closed shell)
        - Valence_P: Valence protons (distance to nearest closed shell)
        - P_Factor: Pairing factor (even-even=1, even-odd/odd-even=0, odd-odd=-1)
        - Shell_Closure_N: Nearest neutron magic number
        - Shell_Closure_P: Nearest proton magic number

        Args:
            df: Input dataframe

        Returns:
            DataFrame with Tier D features
        """
        if self.enricher is None or self.enricher.nubase_data is None:
            logger.warning("NUBASE2020 data not available. Tier D features unavailable.")
            return df

        result = df.copy()

        # Enrich with NUBASE data
        result = self.enricher.enrich_dataframe(result, tiers=['D'])

        # Compute valence nucleons (distance to nearest magic number)
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]

        def get_valence(n, magic_nums):
            """Distance to nearest magic number."""
            distances = [abs(n - m) for m in magic_nums]
            return min(distances)

        result['Valence_N'] = result['N'].apply(lambda n: get_valence(n, magic_numbers))
        result['Valence_P'] = result['Z'].apply(lambda z: get_valence(z, magic_numbers))

        # Pairing factor
        def pairing_factor(n, z):
            """Compute pairing factor: 1 (even-even), 0 (mixed), -1 (odd-odd)."""
            n_even = (n % 2 == 0)
            z_even = (z % 2 == 0)
            if n_even and z_even:
                return 1
            elif not n_even and not z_even:
                return -1
            else:
                return 0

        result['P_Factor'] = result.apply(
            lambda row: pairing_factor(row['N'], row['Z']),
            axis=1
        )

        # Nearest magic numbers
        def nearest_magic(n, magic_nums):
            """Find nearest magic number."""
            distances = [(abs(n - m), m) for m in magic_nums]
            return min(distances)[1]

        result['Shell_Closure_N'] = result['N'].apply(lambda n: nearest_magic(n, magic_numbers))
        result['Shell_Closure_P'] = result['Z'].apply(lambda z: nearest_magic(z, magic_numbers))

        return result

    def _add_tier_e_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Tier E (Complete) features.

        Features from AME2020 rct1 and rct2:
        - All reaction Q-values from rct1: Q_alpha, Q_2beta_minus, Q_ep, Q_beta_n
        - All reaction Q-values from rct2: Q_4beta_minus, Q_d_alpha, Q_p_alpha, Q_n_alpha

        Args:
            df: Input dataframe

        Returns:
            DataFrame with Tier E features
        """
        if self.enricher is None:
            logger.warning("No enricher available. Tier E features require AME2020DataEnricher.")
            return df

        result = df.copy()

        # Enrich with all Q-values
        result = self.enricher.enrich_dataframe(result, tiers=['E'])

        # Convert keV to MeV
        q_value_cols = [
            'Q_alpha', 'Q_2beta_minus', 'Q_ep', 'Q_beta_n',
            'Q_4beta_minus', 'Q_d_alpha', 'Q_p_alpha', 'Q_n_alpha'
        ]

        for col in q_value_cols:
            if col in result.columns:
                result[f'{col}_MeV'] = result[col] / 1000.0
                result = result.drop(columns=[col])

        return result

    def _compute_particle_emission_vector(self, mt_series: pd.Series) -> pd.DataFrame:
        """
        Compute particle-emission vector from MT codes.

        Replaces one-hot encoding with physics-aware representation.
        Based on Valdez 2021 thesis, Table 4.15.

        Vector format: [n_out, p_out, d_out, t_out, He3_out, α_out]

        Args:
            mt_series: Series of MT codes

        Returns:
            DataFrame with particle emission columns
        """
        # MT code to particle emission mapping
        # Format: {MT: (n, p, d, t, He3, α)}
        emission_map = {
            # Elastic and capture
            2: (0, 0, 0, 0, 0, 0),      # Elastic - neutron scatters
            102: (0, 0, 0, 0, 0, 0),    # (n,γ) - gamma emission only

            # Neutron emission
            16: (2, 0, 0, 0, 0, 0),     # (n,2n)
            17: (3, 0, 0, 0, 0, 0),     # (n,3n)
            37: (4, 0, 0, 0, 0, 0),     # (n,4n)

            # Charged particle emission
            103: (0, 1, 0, 0, 0, 0),    # (n,p)
            104: (0, 0, 1, 0, 0, 0),    # (n,d)
            105: (0, 0, 0, 1, 0, 0),    # (n,t)
            106: (0, 0, 0, 0, 1, 0),    # (n,He3)
            107: (0, 0, 0, 0, 0, 1),    # (n,α)

            # Combined emissions
            22: (1, 0, 0, 0, 0, 1),     # (n,n'α)
            28: (0, 1, 0, 0, 0, 1),     # (n,p+α) = (n,nα)
            32: (0, 0, 1, 0, 0, 0),     # (n,n'd)
            33: (0, 0, 0, 1, 0, 0),     # (n,n't)
            41: (2, 1, 0, 0, 0, 0),     # (n,2n+p)
            42: (3, 0, 0, 0, 0, 0),     # (n,3n+p) [approximation]
            44: (1, 2, 0, 0, 0, 0),     # (n,n+2p)
            45: (0, 0, 0, 0, 0, 2),     # (n,2α)

            # Fission (special case - average multiplicity)
            18: (2.5, 0, 0, 0, 0, 0),   # Fission - average ~2.5 neutrons
            19: (2.5, 0, 0, 0, 0, 0),   # First-chance fission
            20: (2.5, 0, 0, 0, 0, 0),   # Second-chance fission
            21: (2.5, 0, 0, 0, 0, 0),   # Third-chance fission
            38: (2.5, 0, 0, 0, 0, 0),   # Fourth-chance fission

            # Absorption (no particles emitted)
            27: (0, 0, 0, 0, 0, 0),     # Absorption
        }

        # Default for unknown MT codes: assume minimal emission
        default_emission = (0, 0, 0, 0, 0, 0)

        # Vectorized mapping
        emissions = mt_series.map(lambda mt: emission_map.get(mt, default_emission))

        # Convert to DataFrame
        emission_df = pd.DataFrame(
            emissions.tolist(),
            columns=['n_out', 'p_out', 'd_out', 't_out', 'He3_out', 'alpha_out'],
            index=mt_series.index
        )

        return emission_df

    def get_tier_feature_names(self, tiers: List[str]) -> List[str]:
        """
        Get list of feature names for specified tiers.

        Args:
            tiers: List of tier identifiers (e.g., ['A', 'C', 'E'])

        Returns:
            List of feature column names
        """
        features = []

        if 'A' in tiers:
            features.extend(['Z', 'A', 'N', 'Energy'])
            features.extend(['n_out', 'p_out', 'd_out', 't_out', 'He3_out', 'alpha_out'])

        if 'B' in tiers:
            features.extend(['R_fm', 'kR'])

        if 'C' in tiers:
            features.extend([
                'Mass_Excess_MeV', 'Binding_Energy_MeV', 'Binding_Per_Nucleon_MeV',
                'S_1n_MeV', 'S_2n_MeV', 'S_1p_MeV', 'S_2p_MeV'
            ])

        if 'D' in tiers:
            features.extend([
                'Spin', 'Parity', 'Isomer_Level',
                'Valence_N', 'Valence_P', 'P_Factor',
                'Shell_Closure_N', 'Shell_Closure_P'
            ])

        if 'E' in tiers:
            features.extend([
                'Q_alpha_MeV', 'Q_2beta_minus_MeV', 'Q_ep_MeV', 'Q_beta_n_MeV',
                'Q_4beta_minus_MeV', 'Q_d_alpha_MeV', 'Q_p_alpha_MeV', 'Q_n_alpha_MeV'
            ])

        return features

    def get_feature_count_by_tier(self) -> Dict[str, int]:
        """
        Get number of features per tier.

        Returns:
            Dictionary mapping tier name to feature count
        """
        return {
            'A': 10,  # Z, A, N, Energy + 6 particle emission
            'B': 12,  # A + R_fm, kR
            'C': 19,  # B + 7 energetics features
            'D': 27,  # C + 8 topological features
            'E': 35,  # D + 8 Q-value features
        }
