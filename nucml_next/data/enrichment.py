"""
AME2020/NUBASE2020 Data Enrichment
====================================

Comprehensive loader for AME2020 and NUBASE2020 nuclear data tables.
Supports tier-based feature enrichment following Valdez 2021 thesis.

File-to-Tier Mapping:
---------------------
- mass_1.mas20: Mass excess, binding energy (Tier B, C)
- rct1.mas20: S(2n), S(2p), Q(alpha), Q(2beta-), Q(ep), Q(beta-n) (Tier C)
- rct2_1.mas20: S(1n), S(1p), Q(4beta-), Q(d,alpha), Q(p,alpha), Q(n,alpha) (Tier C, E)
- nubase_4.mas20: Spin, parity, isomeric states (Tier D) - currently unavailable
- covariance.mas20: Uncertainty covariances (optional)

Tier Hierarchy:
---------------
- Tier A: Z, A, Energy, MT (no enrichment needed)
- Tier B: + Radius/Radius-Ratio features
- Tier C: + Mass Excess, Binding Energy, Separation Energies
- Tier D: + Valence, P-Factor, Isomeric states, Spin, Parity
- Tier E: + All reaction Q-values

Usage:
------
    from nucml_next.data.enrichment import AME2020DataEnricher

    enricher = AME2020DataEnricher(data_dir='data/')
    enricher.load_all()

    # Get enriched data for specific isotope
    u235_data = enricher.get_isotope_data(Z=92, A=235, tiers=['B', 'C'])
"""

import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AME2020DataEnricher:
    """
    Comprehensive enricher for AME2020 and NUBASE2020 nuclear data.

    Loads multiple AME2020 data files and provides enrichment for tier-based
    feature hierarchy system.
    """

    def __init__(self, data_dir: str = 'data/'):
        """
        Initialize AME2020 data enricher.

        Args:
            data_dir: Directory containing AME2020/NUBASE2020 *.mas20.txt files
        """
        self.data_dir = Path(data_dir)

        # Data storage for each file
        self.mass_data: Optional[pd.DataFrame] = None  # mass_1.mas20
        self.rct1_data: Optional[pd.DataFrame] = None  # rct1.mas20
        self.rct2_data: Optional[pd.DataFrame] = None  # rct2_1.mas20
        self.nubase_data: Optional[pd.DataFrame] = None  # nubase_4.mas20

        # Merged enrichment table (all isotopes with all available data)
        self.enrichment_table: Optional[pd.DataFrame] = None

    def load_all(self) -> pd.DataFrame:
        """
        Load all available AME2020/NUBASE2020 data files.

        Returns:
            Merged enrichment table with all available data
        """
        logger.info("Loading AME2020/NUBASE2020 data files...")

        # Load each file if available
        self.mass_data = self._load_mass_1()
        self.rct1_data = self._load_rct1()
        self.rct2_data = self._load_rct2_1()
        self.nubase_data = self._load_nubase_4()

        # Merge all data into single enrichment table
        self._merge_enrichment_table()

        logger.info(f"Loaded {len(self.enrichment_table)} isotopes with enrichment data")
        return self.enrichment_table

    def _load_mass_1(self) -> Optional[pd.DataFrame]:
        """
        Load mass_1.mas20.txt (Mass Excess and Binding Energy).

        Provides:
        - Mass_Excess_keV: Mass excess in keV
        - Binding_Energy_keV: Total binding energy in keV
        - Binding_Per_Nucleon_keV: Binding energy per nucleon

        Returns:
            DataFrame with Z, A, N, Mass_Excess_keV, Binding_Energy_keV, Binding_Per_Nucleon_keV
        """
        filepath = self.data_dir / 'mass_1.mas20.txt'

        if not filepath.exists():
            logger.warning(f"mass_1.mas20.txt not found at {filepath}")
            return None

        logger.info("Loading mass_1.mas20.txt...")
        records = []

        with open(filepath, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                # Skip header lines (lines starting with 1 in first column = page break)
                # Lines starting with 0 or space are data lines
                if line[0] == '1':
                    continue

                # Skip comment/header sections
                if 'ATOMIC' in line or 'A =' in line or '****' in line:
                    continue
                if 'format' in line or 'Warnings' in line or 'cc' in line:
                    continue
                if '....+' in line:  # Column markers
                    continue

                try:
                    # Fixed-width format: a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,...
                    # Columns (1-indexed Fortran style):
                    #   1: control character
                    #   2-4: N-Z
                    #   5-9: N
                    #   10-14: Z
                    #   15-19: A
                    #   21-23: Element
                    #   29-42: Mass excess (keV)
                    #   55-67: Binding energy (keV)

                    if len(line) < 67:
                        continue

                    # Extract N, Z, A (using 0-indexed Python slicing)
                    n_str = line[4:9].strip()
                    z_str = line[9:14].strip()
                    a_str = line[14:19].strip()

                    if not n_str or not z_str or not a_str:
                        continue

                    N = int(n_str)
                    Z = int(z_str)
                    A = int(a_str)

                    # Extract mass excess (cols 29-42)
                    mass_excess_str = line[28:42].strip().replace('#', '').replace('*', '')
                    if not mass_excess_str:
                        continue
                    mass_excess = float(mass_excess_str)

                    # Extract binding energy (cols 55-67)
                    binding_str = line[54:67].strip().replace('#', '').replace('*', '')
                    if binding_str:
                        binding = float(binding_str)
                    else:
                        binding = np.nan

                    records.append({
                        'Z': Z,
                        'A': A,
                        'N': N,
                        'Mass_Excess_keV': mass_excess,
                        'Binding_Energy_keV': binding,
                        'Binding_Per_Nucleon_keV': binding / A if not np.isnan(binding) and A > 0 else np.nan,
                    })

                except (ValueError, IndexError) as e:
                    continue

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} isotopes from mass_1.mas20.txt")
        return df

    def _load_rct1(self) -> Optional[pd.DataFrame]:
        """
        Load rct1.mas20.txt (Separation Energies and Q-values, Part 1).

        Provides:
        - S_2n: Two-neutron separation energy (keV)
        - S_2p: Two-proton separation energy (keV)
        - Q_alpha: Alpha decay Q-value (keV)
        - Q_2beta_minus: Double beta-minus Q-value (keV)
        - Q_ep: Electron capture + positron Q-value (keV)
        - Q_beta_n: Beta-delayed neutron Q-value (keV)

        Returns:
            DataFrame with Z, A, and above columns
        """
        filepath = self.data_dir / 'rct1.mas20.txt'

        if not filepath.exists():
            logger.warning(f"rct1.mas20.txt not found at {filepath}")
            return None

        logger.info("Loading rct1.mas20.txt...")
        records = []

        with open(filepath, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                # Skip page breaks and header lines
                if line[0] == '1':
                    continue

                # Skip comment/header sections
                if 'ATOMIC' in line or 'A =' in line or '****' in line:
                    continue
                if 'format' in line or 'Warnings' in line or 'cc' in line:
                    continue
                if '....+' in line or 'LINEAR' in line:
                    continue

                try:
                    # Fixed-width format: a1,i3,1x,a3,i3,1x,6(f12.4,f10.4)
                    # Columns (1-indexed):
                    #   1: control
                    #   2-4: A
                    #   5: space
                    #   6-8: Element
                    #   9-11: Z
                    #   12: space
                    #   13-24: S(2n) value
                    #   25-34: S(2n) unc
                    #   35-46: S(2p) value
                    #   ... (6 pairs total)

                    if len(line) < 50:
                        continue

                    # Parse A and Z (0-indexed Python)
                    a_str = line[1:4].strip()
                    z_str = line[8:11].strip()

                    if not a_str or not z_str:
                        continue

                    A = int(a_str)
                    Z = int(z_str)

                    # Parse 6 reaction energy values
                    # Each pair: value (12 chars) + uncertainty (10 chars) = 22 chars total
                    values = []
                    pos = 12  # Start after Z field and space (0-indexed)
                    for i in range(6):
                        if pos + 12 > len(line):
                            values.append(np.nan)
                        else:
                            val_str = line[pos:pos+12].strip().replace('#', '').replace('*', '')
                            if val_str:
                                try:
                                    values.append(float(val_str))
                                except ValueError:
                                    values.append(np.nan)
                            else:
                                values.append(np.nan)
                        pos += 22  # Next pair

                    records.append({
                        'Z': Z,
                        'A': A,
                        'S_2n': values[0],
                        'S_2p': values[1],
                        'Q_alpha': values[2],
                        'Q_2beta_minus': values[3],
                        'Q_ep': values[4],
                        'Q_beta_n': values[5],
                    })

                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} isotopes from rct1.mas20.txt")
        return df

    def _load_rct2_1(self) -> Optional[pd.DataFrame]:
        """
        Load rct2_1.mas20.txt (Separation Energies and Q-values, Part 2).

        Provides:
        - S_1n: One-neutron separation energy (keV)
        - S_1p: One-proton separation energy (keV)
        - Q_4beta_minus: Quadruple beta-minus Q-value (keV)
        - Q_d_alpha: (d,alpha) reaction Q-value (keV)
        - Q_p_alpha: (p,alpha) reaction Q-value (keV)
        - Q_n_alpha: (n,alpha) reaction Q-value (keV)

        Returns:
            DataFrame with Z, A, and above columns
        """
        filepath = self.data_dir / 'rct2_1.mas20.txt'

        if not filepath.exists():
            logger.warning(f"rct2_1.mas20.txt not found at {filepath}")
            return None

        logger.info("Loading rct2_1.mas20.txt...")
        records = []

        with open(filepath, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                # Skip page breaks and header lines
                if line[0] == '1':
                    continue

                # Skip comment/header sections
                if 'ATOMIC' in line or 'A =' in line or '****' in line:
                    continue
                if 'format' in line or 'Warnings' in line or 'cc' in line:
                    continue
                if '....+' in line or 'LINEAR' in line:
                    continue

                try:
                    # Same fixed-width format as rct1: a1,i3,1x,a3,i3,1x,6(f12.4,f10.4)

                    if len(line) < 50:
                        continue

                    # Parse A and Z (0-indexed Python)
                    a_str = line[1:4].strip()
                    z_str = line[8:11].strip()

                    if not a_str or not z_str:
                        continue

                    A = int(a_str)
                    Z = int(z_str)

                    # Parse 6 reaction energy values
                    values = []
                    pos = 12  # Start after Z field and space
                    for i in range(6):
                        if pos + 12 > len(line):
                            values.append(np.nan)
                        else:
                            val_str = line[pos:pos+12].strip().replace('#', '').replace('*', '')
                            if val_str:
                                try:
                                    values.append(float(val_str))
                                except ValueError:
                                    values.append(np.nan)
                            else:
                                values.append(np.nan)
                        pos += 22

                    records.append({
                        'Z': Z,
                        'A': A,
                        'S_1n': values[0],
                        'S_1p': values[1],
                        'Q_4beta_minus': values[2],
                        'Q_d_alpha': values[3],
                        'Q_p_alpha': values[4],
                        'Q_n_alpha': values[5],
                    })

                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} isotopes from rct2_1.mas20.txt")
        return df

    def _load_nubase_4(self) -> Optional[pd.DataFrame]:
        """
        Load nubase_4.mas20.txt (Nuclear Structure Properties).

        Provides:
        - Spin: Nuclear spin (J)
        - Parity: Parity (+1 or -1)
        - Isomer_Level: Isomeric state level (0=ground, 1=first excited, etc.)
        - Half_Life_s: Half-life in seconds
        - Decay_Mode: Primary decay mode

        Returns:
            DataFrame with Z, A, and above columns

        Note:
            This file is currently unavailable in the repository.
            Returns None until file is obtained.
        """
        filepath = self.data_dir / 'nubase_4.mas20.txt'

        if not filepath.exists():
            logger.warning(f"nubase_4.mas20.txt not found - Tier D features unavailable")
            return None

        logger.info("Loading nubase_4.mas20.txt...")
        # TODO: Implement NUBASE parser when file becomes available
        # NUBASE format is more complex and requires special handling

        return None

    def _merge_enrichment_table(self):
        """
        Merge all loaded data into single enrichment table.

        Uses left joins on (Z, A) starting from mass_1 data as base.
        """
        if self.mass_data is None:
            logger.error("Cannot create enrichment table: mass_1.mas20.txt not loaded")
            self.enrichment_table = pd.DataFrame()
            return

        # Start with mass data as base
        merged = self.mass_data.copy()

        # Merge rct1 data
        if self.rct1_data is not None:
            merged = merged.merge(self.rct1_data, on=['Z', 'A'], how='left')

        # Merge rct2 data
        if self.rct2_data is not None:
            merged = merged.merge(self.rct2_data, on=['Z', 'A'], how='left')

        # Merge nubase data (when available)
        if self.nubase_data is not None:
            merged = merged.merge(self.nubase_data, on=['Z', 'A'], how='left')

        self.enrichment_table = merged

    def get_isotope_data(
        self,
        Z: int,
        A: int,
        tiers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get enriched data for a specific isotope.

        Args:
            Z: Atomic number
            A: Mass number
            tiers: List of tiers to include (e.g., ['B', 'C', 'E'])
                  If None, returns all available data

        Returns:
            Dictionary of enriched features for the isotope
            Returns empty dict if isotope not found
        """
        if self.enrichment_table is None:
            logger.error("Enrichment table not loaded. Call load_all() first.")
            return {}

        # Find isotope
        mask = (self.enrichment_table['Z'] == Z) & (self.enrichment_table['A'] == A)
        isotope_data = self.enrichment_table[mask]

        if len(isotope_data) == 0:
            logger.warning(f"Isotope Z={Z}, A={A} not found in enrichment table")
            return {}

        # Convert to dict
        data_dict = isotope_data.iloc[0].to_dict()

        # Filter by tiers if specified
        if tiers is not None:
            filtered_data = {}
            tier_columns = self._get_tier_columns(tiers)
            for col in tier_columns:
                if col in data_dict:
                    filtered_data[col] = data_dict[col]
            return filtered_data

        return data_dict

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        tiers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Enrich a dataframe with AME2020 data.

        Args:
            df: DataFrame with Z and A columns
            tiers: List of tiers to include (e.g., ['B', 'C', 'E'])
                  If None, adds all available enrichment data

        Returns:
            Enriched DataFrame with additional columns from AME2020
        """
        if self.enrichment_table is None:
            logger.error("Enrichment table not loaded. Call load_all() first.")
            return df

        # Determine which columns to add
        if tiers is None:
            # Add all columns except Z, A, N (already in df or will be computed)
            enrich_cols = [col for col in self.enrichment_table.columns
                          if col not in ['Z', 'A', 'N']]
        else:
            # Add only tier-specific columns
            tier_cols = self._get_tier_columns(tiers)
            # Filter to columns that exist in enrichment table
            # Exclude Z, A (merge keys) and N (will be computed separately)
            enrich_cols = [col for col in tier_cols
                          if col in self.enrichment_table.columns and col not in ['Z', 'A', 'N']]

        # Select enrichment columns (include Z, A for merge)
        cols_to_select = ['Z', 'A'] + enrich_cols
        enrich_data = self.enrichment_table[cols_to_select].copy()

        # Merge with input dataframe (creates no duplicate columns)
        enriched = df.merge(enrich_data, on=['Z', 'A'], how='left', suffixes=('', '_ame'))

        return enriched

    def _get_tier_columns(self, tiers: List[str]) -> List[str]:
        """
        Get column names for specified tiers.

        Args:
            tiers: List of tier identifiers (e.g., ['B', 'C'])

        Returns:
            List of column names to include
        """
        columns = ['Z', 'A', 'N']  # Always include these

        if 'B' in tiers or 'C' in tiers:
            # Tier B and C both need mass and binding energy
            columns.extend([
                'Mass_Excess_keV',
                'Binding_Energy_keV',
                'Binding_Per_Nucleon_keV'
            ])

        if 'C' in tiers:
            # Tier C adds separation energies
            columns.extend([
                'S_1n', 'S_2n', 'S_1p', 'S_2p'
            ])

        if 'D' in tiers:
            # Tier D adds nuclear structure properties
            columns.extend([
                'Spin', 'Parity', 'Isomer_Level', 'Half_Life_s'
            ])

        if 'E' in tiers:
            # Tier E adds all Q-values
            columns.extend([
                'Q_alpha', 'Q_2beta_minus', 'Q_ep', 'Q_beta_n',
                'Q_4beta_minus', 'Q_d_alpha', 'Q_p_alpha', 'Q_n_alpha'
            ])

        return columns

    def get_available_tiers(self) -> List[str]:
        """
        Get list of tiers that can be implemented with available data.

        Returns:
            List of available tier identifiers (e.g., ['A', 'B', 'C', 'E'])
        """
        available = ['A']  # Tier A always available (Z, A, Energy, MT)

        if self.mass_data is not None:
            available.extend(['B', 'C'])  # Both need mass data

        if self.nubase_data is not None:
            available.append('D')  # Tier D needs NUBASE

        if self.rct2_data is not None:
            available.append('E')  # Tier E needs reaction Q-values

        return sorted(set(available))
