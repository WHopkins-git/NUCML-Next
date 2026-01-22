"""
X4Pro SQLite Ingestor
====================

Clean, single-path ingestion from X4Pro SQLite database to Parquet.

X4Pro Schema Assumptions:
-------------------------
The X4Pro SQLite database (x4sqlite1.db) contains EXFOR experimental nuclear
cross-section data with the following structure:

Core Tables:
- entries: Main entry metadata (ENTRY, AUTHORS, etc.)
- data_points: Reaction data with columns:
  - entry_id: EXFOR entry identifier (e.g., "12345")
  - subentry: Subentry number (e.g., "002")
  - target_z: Atomic number of target nucleus
  - target_a: Mass number of target nucleus (0 for natural)
  - reaction_mt: ENDF MT reaction code (18=fission, 102=capture, etc.)
  - energy: Incident energy in eV
  - xs: Cross section value in barns
  - dxs: Uncertainty in barns (NULL if not reported)
  - lab_frame: Boolean, True if lab frame (vs CM)

Alternative schemas supported:
- If data is split across tables (reactions, energies, cross_sections),
  the ingestor will join them appropriately
- Column name variants (e.g., 'xs'/'cross_section', 'dxs'/'uncertainty')

Pipeline:
---------
1. Connect to X4 SQLite database
2. Extract point data with minimal metadata
3. Normalize to DataFrame (Z, A, MT, Energy, CrossSection, Uncertainty)
4. Enrich with AME2020 isotopic properties (optional)
5. Write partitioned Parquet dataset

Usage:
------
    from nucml_next.ingest import X4Ingestor

    ingestor = X4Ingestor(
        x4_db_path='data/x4sqlite1.db',
        output_path='data/exfor_processed.parquet'
    )
    df = ingestor.ingest()
"""

import sqlite3
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AME2020Loader:
    """
    Loader for AME2020 (Atomic Mass Evaluation 2020) isotopic data.

    Provides mass excess and binding energy for nuclear enrichment.
    Falls back to SEMF approximation if file not available.
    """

    def __init__(self, ame2020_path: Optional[str] = None):
        """
        Initialize AME2020 loader.

        Args:
            ame2020_path: Path to AME2020 mass_1.mas20.txt file.
                         If None or file not found, uses SEMF approximation.
        """
        self.ame2020_path = ame2020_path
        self.data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load AME2020 data or generate SEMF fallback.

        Returns:
            DataFrame with columns: Z, A, N, Mass_Excess_keV, Binding_Energy_keV
        """
        if self.ame2020_path and Path(self.ame2020_path).exists():
            logger.info(f"Loading AME2020 from {self.ame2020_path}")
            self.data = self._parse_ame2020(self.ame2020_path)
        else:
            logger.warning("AME2020 file not found. Using SEMF approximation for common isotopes.")
            self.data = self._generate_semf_fallback()

        logger.info(f"Loaded {len(self.data)} isotopes")
        return self.data

    def _parse_ame2020(self, filepath: str) -> pd.DataFrame:
        """
        Parse AME2020 fixed-width format file.

        Format (space-separated):
        N-Z  N  Z  A  El  Mass_Excess(keV)  unc  Binding_Energy(keV)  unc  flag

        Args:
            filepath: Path to mass_1.mas20.txt

        Returns:
            Parsed DataFrame
        """
        records = []

        with open(filepath, 'r') as f:
            for line in f:
                # Skip comments and headers
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 9:
                    try:
                        # Format: N-Z  N  Z  A  El  Mass_Excess  unc  Binding_Energy  unc  flag
                        N = int(parts[1])
                        Z = int(parts[2])
                        A = int(parts[3])
                        mass_excess = float(parts[5])  # keV
                        binding_energy = float(parts[7])  # keV

                        records.append({
                            'Z': Z,
                            'A': A,
                            'N': N,
                            'Mass_Excess_keV': mass_excess,
                            'Binding_Energy_keV': binding_energy,
                        })
                    except (ValueError, IndexError):
                        continue

        return pd.DataFrame(records)

    def _generate_semf_fallback(self) -> pd.DataFrame:
        """
        Generate SEMF (Semi-Empirical Mass Formula) approximations for common isotopes.

        Returns:
            DataFrame with estimated nuclear properties
        """
        # Common isotopes in EXFOR database
        isotopes = [
            (1, 1),    # H-1
            (1, 2),    # H-2
            (6, 12),   # C-12
            (8, 16),   # O-16
            (17, 35),  # Cl-35
            (17, 37),  # Cl-37
            (92, 233), # U-233
            (92, 235), # U-235
            (92, 238), # U-238
            (94, 239), # Pu-239
            (94, 240), # Pu-240
        ]

        records = []
        for Z, A in isotopes:
            N = A - Z

            # SEMF coefficients
            a_v, a_s, a_c, a_a, a_p = 15.75, 17.8, 0.711, 23.7, 11.18

            # Binding energy calculation
            volume = a_v * A
            surface = -a_s * (A ** (2/3))
            coulomb = -a_c * (Z ** 2) / (A ** (1/3))
            asymmetry = -a_a * ((N - Z) ** 2) / A

            if N % 2 == 0 and Z % 2 == 0:
                pairing = a_p / np.sqrt(A)
            elif N % 2 == 1 and Z % 2 == 1:
                pairing = -a_p / np.sqrt(A)
            else:
                pairing = 0.0

            binding_energy_mev = volume + surface + coulomb + asymmetry + pairing
            mass_excess_mev = -binding_energy_mev

            records.append({
                'Z': Z,
                'A': A,
                'N': N,
                'Mass_Excess_keV': mass_excess_mev * 1000,
                'Binding_Energy_keV': binding_energy_mev * 1000,
            })

        return pd.DataFrame(records)


class X4Ingestor:
    """
    X4Pro SQLite ingestor for EXFOR nuclear cross-section data.

    Responsibilities:
    - Connect to X4 SQLite database
    - Extract reaction point data
    - Normalize to standard schema
    - Optionally enrich with AME2020
    - Write partitioned Parquet output
    """

    def __init__(
        self,
        x4_db_path: str,
        output_path: str = 'data/exfor_processed.parquet',
        ame2020_path: Optional[str] = None,
        partitioning: List[str] = ['Z', 'A', 'MT'],
        max_partitions: int = 10000,
    ):
        """
        Initialize X4 ingestor.

        Args:
            x4_db_path: Path to X4Pro SQLite database (e.g., x4sqlite1.db)
            output_path: Output path for Parquet dataset
            ame2020_path: Optional path to AME2020 file for enrichment
            partitioning: Partition columns for Parquet output
            max_partitions: Maximum number of partitions allowed (default: 10000)
                           The full EXFOR database can have >1000 unique Z/A/MT combinations.
                           PyArrow's default limit is 1024, which is insufficient for full EXFOR.
        """
        self.x4_db_path = Path(x4_db_path)
        self.output_path = Path(output_path)
        self.partitioning = partitioning
        self.max_partitions = max_partitions

        if not self.x4_db_path.exists():
            raise FileNotFoundError(f"X4 database not found: {self.x4_db_path}")

        # Load AME2020 if requested
        if ame2020_path:
            ame_loader = AME2020Loader(ame2020_path)
            self.ame2020 = ame_loader.load()
        else:
            logger.info("No AME2020 enrichment requested")
            self.ame2020 = None

    def ingest(self) -> pd.DataFrame:
        """
        Execute full ingestion pipeline.

        Returns:
            DataFrame with extracted and normalized data
        """
        logger.info("="*70)
        logger.info("X4Pro SQLite Ingestion Pipeline")
        logger.info("="*70)
        logger.info(f"Source: {self.x4_db_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info("="*70)

        # Connect to database
        conn = sqlite3.connect(str(self.x4_db_path))

        try:
            # Extract data points
            df = self._extract_points(conn)
            logger.info(f"Extracted {len(df)} data points")

            # Normalize schema
            df = self._normalize(df)
            logger.info(f"Normalized to standard schema")

            # Enrich with AME2020 if available
            if self.ame2020 is not None:
                df = self._enrich_ame2020(df)
                logger.info(f"Enriched with AME2020 isotopic data")

            # Write to Parquet
            self._write_parquet(df)
            logger.info(f"Saved to {self.output_path}")

            # Print summary
            self._print_summary(df)

            return df

        finally:
            conn.close()

    def _extract_points(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Extract point data from X4 database.

        Attempts multiple query strategies to handle schema variations:
        1. Official X4Pro schema (x4pro_ds + x4pro_x5z)
        2. Legacy data_points table
        3. Separate joined tables
        4. Generic fallback

        Args:
            conn: SQLite connection

        Returns:
            Raw DataFrame from database
        """
        # Inspect schema - check both tables and views
        cursor = conn.cursor()
        cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
        objects = cursor.fetchall()
        tables = [name for name, obj_type in objects if obj_type == 'table']
        views = [name for name, obj_type in objects if obj_type == 'view']
        logger.info(f"Found tables: {', '.join(tables[:10])}..." if len(tables) > 10 else f"Found tables: {', '.join(tables)}")

        # Strategy 1: Official X4Pro schema (x4pro_ds + x4pro_x5z with JSON data)
        if 'x4pro_ds' in tables and 'x4pro_x5z' in tables:
            logger.info("Using X4Pro schema (x4pro_ds + x4pro_x5z)")
            return self._extract_x4pro_json(conn)

        # Strategy 2: X4Pro with c5dat table (if populated)
        elif 'x4pro_ds' in tables and 'x4pro_c5dat' in tables:
            # Check if c5dat has data
            cursor.execute("SELECT COUNT(*) FROM x4pro_c5dat LIMIT 1")
            if cursor.fetchone()[0] > 0:
                logger.info("Using X4Pro schema (x4pro_ds + x4pro_c5dat)")
                return self._extract_x4pro_c5dat(conn)

        # Strategy 3: Legacy data_points table
        if 'data_points' in tables:
            logger.info("Using legacy data_points table")
            query = """
                SELECT
                    entry_id,
                    target_z,
                    target_a,
                    reaction_mt,
                    energy,
                    xs,
                    dxs
                FROM data_points
                WHERE energy IS NOT NULL
                  AND xs IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)
            return df

        # Strategy 4: Separate tables (reactions, energies, cross_sections)
        elif 'reactions' in tables:
            logger.info("Using joined reactions/energies/cross_sections tables")
            query = """
                SELECT
                    r.entry_id,
                    r.target_z,
                    r.target_a,
                    r.mt AS reaction_mt,
                    e.value AS energy,
                    x.value AS xs,
                    u.value AS dxs
                FROM reactions r
                LEFT JOIN energies e ON r.id = e.reaction_id
                LEFT JOIN cross_sections x ON r.id = x.reaction_id
                LEFT JOIN uncertainties u ON r.id = u.reaction_id
                WHERE e.value IS NOT NULL
                  AND x.value IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)
            return df

        # Strategy 5: Generic fallback (inspect first table with relevant columns)
        else:
            logger.warning("Unknown schema. Attempting generic extraction from first suitable table.")
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]

                # Check if table has key columns
                has_energy = any('energy' in col.lower() for col in columns)
                has_xs = any('xs' in col.lower() or 'cross' in col.lower() for col in columns)
                has_z = any('z' in col.lower() or 'target' in col.lower() for col in columns)

                if has_energy and has_xs and has_z:
                    logger.info(f"Using table: {table}")
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    return df

            raise ValueError("Could not determine X4 database schema. Please check database structure.")

    def _extract_x4pro_json(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Extract data from X4Pro schema using JSON format (x4pro_ds + x4pro_x5z).

        The official X4Pro database stores cross-section data in JSON format within
        the jx5z column of x4pro_x5z table. This method:
        1. JOINs x4pro_ds (metadata) with x4pro_x5z (JSON data) on DatasetID
        2. Parses c5data JSON to extract energy (x1), cross-section (y), uncertainty (dy)
        3. Extracts Z, A from target isotope string (e.g., "U-235" → Z=92, A=235)

        Args:
            conn: SQLite connection

        Returns:
            DataFrame with columns: [DatasetID, Z, A, MT, Energy, Data, dData]
        """
        import json
        import re

        # Query to get dataset metadata and JSON data
        # IMPORTANT: DatasetID is a string (e.g., "30649005S"), so quote properly
        query = """
            SELECT
                ds.DatasetID,
                ds.zTarg1 as Z,
                ds.Targ1 as Target,
                ds.MT,
                x5.jx5z
            FROM x4pro_ds ds
            INNER JOIN x4pro_x5z x5 ON ds.DatasetID = x5.DatasetID
            WHERE ds.zTarg1 IS NOT NULL
              AND ds.MT IS NOT NULL
        """

        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        logger.info(f"Found {len(rows)} datasets with JSON data")

        # Parse JSON and extract data points
        all_points = []

        for dataset_id, z, target, mt, jx5z_str in rows:
            if not jx5z_str:
                continue

            try:
                # Parse JSON
                x5data = json.loads(jx5z_str)

                # Extract A from target string (e.g., "U-235" → 235, "Al-27" → 27)
                a_match = re.search(r'-(\d+)', target)
                a = int(a_match.group(1)) if a_match else None

                # Skip if we couldn't parse A
                if a is None:
                    logger.debug(f"Could not parse A from target: {target}")
                    continue

                # Extract c5data (corrected/computational data)
                if 'c5data' not in x5data or not x5data['c5data']:
                    continue

                c5 = x5data['c5data']

                # Extract energy (x1), cross-section (y), and uncertainty (dy)
                if 'x1' not in c5 or 'y' not in c5:
                    continue

                energies = c5['x1'].get('x1', [])
                cross_sections = c5['y'].get('y', [])
                uncertainties = c5.get('dy', {}).get('dy', [None] * len(energies))

                # Ensure all arrays have same length
                n_points = min(len(energies), len(cross_sections))
                if n_points == 0:
                    continue

                # Extend uncertainties if needed
                if len(uncertainties) < n_points:
                    uncertainties = uncertainties + [None] * (n_points - len(uncertainties))

                # Create data points
                for i in range(n_points):
                    all_points.append({
                        'DatasetID': dataset_id,
                        'Z': z,
                        'A': a,
                        'MT': mt,
                        'En': energies[i],
                        'Data': cross_sections[i],
                        'dData': uncertainties[i]
                    })

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.debug(f"Error parsing dataset {dataset_id}: {e}")
                continue

        if not all_points:
            raise ValueError("No valid data points extracted from X4Pro JSON format")

        df = pd.DataFrame(all_points)
        logger.info(f"Extracted {len(df)} data points from {df['DatasetID'].nunique()} datasets")

        return df

    def _extract_x4pro_c5dat(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Extract data from X4Pro schema using c5dat table (x4pro_ds + x4pro_c5dat).

        Alternative X4Pro format where corrected data is stored in x4pro_c5dat table
        instead of JSON. Uses INNER JOIN on DatasetID.

        Args:
            conn: SQLite connection

        Returns:
            DataFrame with columns: [DatasetID, Z, A, MT, En, Data, dData]
        """
        import re

        # INNER JOIN between metadata and data tables
        # DatasetID must be quoted as it's a string
        query = """
            SELECT
                ds.DatasetID,
                ds.zTarg1 as Z,
                ds.Targ1 as Target,
                ds.MT,
                dat.x1 as En,
                dat.y as Data,
                dat.dy as dData
            FROM x4pro_ds ds
            INNER JOIN x4pro_c5dat dat ON ds.DatasetID = dat.DatasetID
            WHERE ds.zTarg1 IS NOT NULL
              AND ds.MT IS NOT NULL
              AND dat.x1 IS NOT NULL
              AND dat.y IS NOT NULL
        """

        df = pd.read_sql_query(query, conn)
        logger.info(f"Extracted {len(df)} data points from x4pro_c5dat")

        # Extract A from target string
        import re
        def extract_a(target_str):
            match = re.search(r'-(\d+)', target_str)
            return int(match.group(1)) if match else None

        df['A'] = df['Target'].apply(extract_a)
        df = df.drop(columns=['Target'])

        # Remove rows where A couldn't be parsed
        df = df.dropna(subset=['A'])
        df['A'] = df['A'].astype(int)

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize extracted data to standard schema.

        Handles column name variations and ensures consistent output format.

        Args:
            df: Raw extracted DataFrame

        Returns:
            Normalized DataFrame with columns:
                [Entry, Z, A, MT, Energy, CrossSection, Uncertainty, N]
        """
        # Map column names to standard names
        column_map = {
            # Entry IDs
            'entry_id': 'Entry',
            'entry': 'Entry',
            'entryid': 'Entry',
            'datasetid': 'Entry',  # X4Pro: DatasetID → Entry

            # Nuclear properties
            'target_z': 'Z',
            'z': 'Z',
            'atomic_number': 'Z',

            'target_a': 'A',
            'a': 'A',
            'mass_number': 'A',

            'reaction_mt': 'MT',
            'mt': 'MT',
            'mt_code': 'MT',

            # Data (including X4Pro naming conventions)
            'energy': 'Energy',
            'energy_value': 'Energy',
            'en': 'Energy',        # X4Pro: En → Energy

            'xs': 'CrossSection',
            'data': 'CrossSection',  # X4Pro: Data → CrossSection
            'cross_section': 'CrossSection',
            'sigma': 'CrossSection',
            'value': 'CrossSection',

            'dxs': 'Uncertainty',
            'ddata': 'Uncertainty',  # X4Pro: dData → Uncertainty
            'uncertainty': 'Uncertainty',
            'error': 'Uncertainty',
            'd_cross_section': 'Uncertainty',
        }

        # Rename columns (case-insensitive)
        df_norm = df.copy()
        df_norm.columns = df_norm.columns.str.lower()

        for old_name, new_name in column_map.items():
            if old_name in df_norm.columns:
                df_norm.rename(columns={old_name: new_name}, inplace=True)

        # Ensure required columns exist
        required = ['Z', 'A', 'MT', 'Energy', 'CrossSection']
        missing = [col for col in required if col not in df_norm.columns]
        if missing:
            raise ValueError(f"Missing required columns after normalization: {missing}")

        # Add Entry if missing
        if 'Entry' not in df_norm.columns:
            df_norm['Entry'] = 'UNKNOWN'

        # Add Uncertainty if missing
        if 'Uncertainty' not in df_norm.columns:
            df_norm['Uncertainty'] = np.nan

        # Select and order final columns
        final_cols = ['Entry', 'Z', 'A', 'MT', 'Energy', 'CrossSection', 'Uncertainty']
        df_norm = df_norm[final_cols]

        # Clean data
        df_norm = df_norm.dropna(subset=['Z', 'A', 'MT', 'Energy', 'CrossSection'])
        df_norm['Z'] = df_norm['Z'].astype(int)
        df_norm['A'] = df_norm['A'].astype(int)
        df_norm['MT'] = df_norm['MT'].astype(int)
        df_norm['Energy'] = df_norm['Energy'].astype(float)
        df_norm['CrossSection'] = df_norm['CrossSection'].astype(float)

        # Calculate neutron number (N = A - Z)
        # This is a fundamental nuclear property and should always be present
        df_norm['N'] = df_norm['A'] - df_norm['Z']

        return df_norm

    def _enrich_ame2020(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataset with AME2020 isotopic properties.

        Args:
            df: Normalized DataFrame (already has N column)

        Returns:
            Enriched DataFrame with Mass_Excess_keV, Binding_Energy_keV
        """
        if self.ame2020 is None:
            return df

        # Select only AME2020 columns we need (avoid N duplication since df already has N)
        ame_cols = ['Z', 'A', 'Mass_Excess_keV', 'Binding_Energy_keV']
        ame_subset = self.ame2020[ame_cols].copy()

        df_enriched = df.merge(
            ame_subset,
            on=['Z', 'A'],
            how='left'
        )

        return df_enriched

    def _write_parquet(self, df: pd.DataFrame):
        """
        Write DataFrame to partitioned Parquet dataset.

        For large datasets like the full EXFOR database, the number of unique
        Z/A/MT combinations can exceed PyArrow's default partition limit (1024).
        This method calculates the expected partition count and uses the
        configured max_partitions limit to handle large datasets.

        Args:
            df: DataFrame to write
        """
        # Calculate expected number of partitions
        if self.partitioning:
            n_partitions = df[self.partitioning].drop_duplicates().shape[0]
            logger.info(f"Creating {n_partitions:,} partitions by {self.partitioning}")

            if n_partitions > 1024:
                logger.info(f"Partition count exceeds PyArrow default limit (1024)")
                logger.info(f"Using max_partitions={self.max_partitions:,}")

        table = pa.Table.from_pandas(df)

        pq.write_to_dataset(
            table,
            root_path=str(self.output_path),
            partition_cols=self.partitioning,
            existing_data_behavior='overwrite_or_ignore',
            max_partitions=self.max_partitions,
        )

    def _print_summary(self, df: pd.DataFrame):
        """Print dataset summary statistics."""
        logger.info("="*70)
        logger.info("Dataset Summary")
        logger.info("="*70)
        logger.info(f"Total data points:     {len(df):,}")
        logger.info(f"Unique isotopes:       {df[['Z', 'A']].drop_duplicates().shape[0]}")
        logger.info(f"Unique reactions (MT): {df['MT'].nunique()}")
        logger.info(f"Energy range:          {df['Energy'].min():.2e} - {df['Energy'].max():.2e} eV")
        logger.info(f"Points with uncertainty: {df['Uncertainty'].notna().sum():,}")

        if 'Mass_Excess_keV' in df.columns:
            logger.info(f"AME2020 enriched:      {df['Mass_Excess_keV'].notna().sum():,} points")

        logger.info("="*70)


# Convenience function
def ingest_x4(
    x4_db_path: str,
    output_path: str = 'data/exfor_processed.parquet',
    ame2020_path: Optional[str] = None,
    max_partitions: int = 10000,
) -> pd.DataFrame:
    """
    Convenience function for X4 ingestion.

    Args:
        x4_db_path: Path to X4Pro SQLite database
        output_path: Output Parquet path
        ame2020_path: Optional AME2020 file for enrichment
        max_partitions: Maximum number of partitions (default: 10000 for full EXFOR)

    Returns:
        Processed DataFrame
    """
    ingestor = X4Ingestor(
        x4_db_path=x4_db_path,
        output_path=output_path,
        ame2020_path=ame2020_path,
        max_partitions=max_partitions,
    )
    return ingestor.ingest()
