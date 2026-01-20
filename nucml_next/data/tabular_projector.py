"""
Tabular Projector for Nuclear Data
===================================

Projects graph-structured nuclear data to tabular format for classical ML.

Two projection modes:
1. 'naive': Legacy features [Z, A, E, MT_onehot]
   - Simple one-hot encoding of reaction types
   - No physics knowledge
   - Shows why classical ML fails

2. 'physics': Graph-derived features [Z, A, E, Q, Threshold, ΔZ, ΔA]
   - Includes reaction energetics
   - Better but still lacks smoothness
   - Bridge to deep learning
"""

from typing import Literal, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class TabularProjector:
    """
    Projects nuclear graph data to tabular format.

    This enables the educational comparison:
    - Naive XGBoost (fails badly)
    - Physics-aware XGBoost (better, but still jagged)
    - GNN-Transformer (smooth, physics-compliant)
    """

    def __init__(self, df: pd.DataFrame, energy_bins: np.ndarray):
        """
        Initialize tabular projector.

        Args:
            df: DataFrame with nuclear data
            energy_bins: Energy grid
        """
        self.df = df.copy()
        self.energy_bins = energy_bins

        # Get unique MT codes for one-hot encoding
        self.mt_codes = sorted(self.df['MT'].unique())
        self.encoder = OneHotEncoder(sparse_output=False, categories=[self.mt_codes])
        self.encoder.fit(self.df[['MT']])

    def project(
        self,
        mode: Literal['naive', 'physics'] = 'naive',
        reaction_types: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Project to tabular format.

        Args:
            mode: Projection strategy
                'naive': [Z, A, E, MT_onehot] - Legacy approach
                'physics': [Z, A, E, Q, Threshold, ΔZ, ΔA] - Graph features
            reaction_types: Filter to specific MT codes

        Returns:
            DataFrame ready for XGBoost/Decision Trees

        Educational Note:
            The 'naive' mode represents how NUCML v1.0 worked.
            It treats nuclear reactions as independent categories.
            This ignores physics: (n,2n) and (n,3n) are related!

            The 'physics' mode includes reaction energetics,
            but still can't capture smoothness or topology.
        """
        df = self.df.copy()

        # Filter reactions if requested
        if reaction_types is not None:
            df = df[df['MT'].isin(reaction_types)]

        if mode == 'naive':
            return self._project_naive(df)
        elif mode == 'physics':
            return self._project_physics(df)
        else:
            raise ValueError(f"Unknown projection mode: {mode}")

    def _project_naive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Naive projection: [Z, A, E, MT_onehot].

        This is the "legacy" approach that ignores physics.

        Returns:
            DataFrame with features: [Z, A, Energy, MT_2, MT_16, MT_18, MT_102, ...]
                               target: CrossSection
        """
        # One-hot encode MT codes
        mt_onehot = self.encoder.transform(df[['MT']])
        mt_columns = [f'MT_{code}' for code in self.mt_codes]
        mt_df = pd.DataFrame(mt_onehot, columns=mt_columns, index=df.index)

        # Combine features
        features = pd.concat([
            df[['Z', 'A', 'Energy']].reset_index(drop=True),
            mt_df.reset_index(drop=True),
        ], axis=1)

        # Add target
        features['CrossSection'] = df['CrossSection'].values

        return features

    def _project_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physics-aware projection: [Z, A, E, Q, Threshold, ΔZ, ΔA, MT].

        Includes reaction energetics and topology hints.

        Features:
            - Z, A: Isotope identity
            - Energy: Incident neutron energy
            - Q_Value: Reaction energy release
            - Threshold: Minimum energy for reaction
            - Delta_Z: Change in atomic number (from graph edges)
            - Delta_A: Change in mass number (from graph edges)
            - MT: Reaction type (as continuous feature)

        Returns:
            DataFrame with physics-informed features
        """
        # Compute delta values based on reaction type
        df['Delta_Z'] = df['MT'].apply(self._get_delta_z)
        df['Delta_A'] = df['MT'].apply(self._get_delta_a)

        # Normalize features
        features = pd.DataFrame({
            'Z': df['Z'],
            'A': df['A'],
            'N': df['N'],
            'Energy': np.log10(df['Energy'] + 1.0),  # Log scale
            'Q_Value': df['Q_Value'] / 1e6,  # MeV units
            'Threshold': df['Threshold'] / 1e6,  # MeV units
            'Delta_Z': df['Delta_Z'],
            'Delta_A': df['Delta_A'],
            'MT': df['MT'] / 100.0,  # Normalized
            'CrossSection': df['CrossSection'],  # Target
        })

        return features

    @staticmethod
    def _get_delta_z(mt_code: int) -> int:
        """
        Get change in Z for a reaction.

        Args:
            mt_code: ENDF MT code

        Returns:
            Change in atomic number
        """
        # MT code to ΔZ mapping
        delta_z_map = {
            2: 0,    # Elastic
            16: 0,   # (n,2n)
            17: 0,   # (n,3n)
            18: None,  # Fission (multi-product)
            102: 0,  # (n,γ)
            103: 1,  # (n,p)
            104: 2,  # (n,d)
            105: 2,  # (n,t)
            106: 2,  # (n,He3)
            107: 2,  # (n,α)
        }
        return delta_z_map.get(mt_code, 0)

    @staticmethod
    def _get_delta_a(mt_code: int) -> int:
        """
        Get change in A for a reaction.

        Args:
            mt_code: ENDF MT code

        Returns:
            Change in mass number
        """
        # MT code to ΔA mapping
        delta_a_map = {
            2: 0,    # Elastic
            16: -1,  # (n,2n) - loses 1 neutron net
            17: -2,  # (n,3n) - loses 2 neutrons net
            18: None,  # Fission
            102: 1,  # (n,γ) - gains 1 neutron
            103: 0,  # (n,p) - loses proton, gains neutron (net 0)
            104: -1, # (n,d) - loses deuteron, gains neutron (net -1)
            105: -2, # (n,t) - loses triton, gains neutron (net -2)
            106: -2, # (n,He3)
            107: -3, # (n,α) - loses alpha, gains neutron (net -3)
        }
        return delta_a_map.get(mt_code, 0)

    def get_feature_importance_mapping(self, mode: str) -> dict:
        """
        Get human-readable feature names for importance analysis.

        Args:
            mode: Projection mode used

        Returns:
            Dictionary mapping column names to descriptions
        """
        if mode == 'naive':
            mapping = {
                'Z': 'Atomic Number',
                'A': 'Mass Number',
                'Energy': 'Incident Energy',
            }
            for code in self.mt_codes:
                mapping[f'MT_{code}'] = f'Reaction Type {code}'
            return mapping

        elif mode == 'physics':
            return {
                'Z': 'Atomic Number',
                'A': 'Mass Number',
                'N': 'Neutron Number',
                'Energy': 'log10(Energy)',
                'Q_Value': 'Q-Value (MeV)',
                'Threshold': 'Reaction Threshold (MeV)',
                'Delta_Z': 'ΔZ (Atomic Number Change)',
                'Delta_A': 'ΔA (Mass Number Change)',
                'MT': 'Reaction Type (Normalized)',
            }
        else:
            return {}
