"""
X4Pro SQLite Ingestor with Full AME2020/NUBASE2020 Enrichment
=============================================================

Production-ready ingestion pipeline: X4Pro SQLite → Parquet with ALL enrichment.

Architecture: Pre-Enrichment Strategy
--------------------------------------
This ingestor implements a **pre-enrichment architecture**:

1. Load ALL AME2020/NUBASE2020 files once during ingestion (5 files)
2. Merge ALL enrichment columns into EXFOR dataframe
3. Write to Parquet with complete enrichment schema
4. Feature selection becomes simple column selection (no joins)

Benefits:
- AME2020 data is static (published every ~4 years) → load once
- Parquet columnar format → only loads needed columns anyway
- Consistent preprocessing → all users get same enrichment
- Faster feature generation → no file I/O or joins needed
- Production-ready → single data source with all features

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
4. Enrich with ALL AME2020/NUBASE2020 columns (mass, energetics, structure, Q-values)
5. Write partitioned Parquet dataset with complete enrichment schema

Output Schema (with full enrichment):
-------------------------------------
Core EXFOR columns:
  - Entry, Z, A, N, MT, Energy, CrossSection, Uncertainty

Tier B/C (mass_1.mas20.txt):
  - Mass_Excess_keV, Binding_Energy_keV, Binding_Per_Nucleon_keV

Tier C (rct1.mas20.txt + rct2_1.mas20.txt):
  - S_1n, S_2n, S_1p, S_2p (separation energies)

Tier D (nubase_4.mas20.txt):
  - Spin, Parity, Isomer_Level, Half_Life_s

Tier E (rct1.mas20.txt + rct2_1.mas20.txt):
  - Q_alpha, Q_2beta_minus, Q_ep, Q_beta_n, Q_4beta_minus,
    Q_d_alpha, Q_p_alpha, Q_n_alpha

Usage:
------
    from nucml_next.ingest import X4Ingestor

    # Basic ingestion (no enrichment)
    ingestor = X4Ingestor(
        x4_db_path='data/x4sqlite1.db',
        output_path='data/exfor_processed.parquet'
    )
    df = ingestor.ingest()

    # Full enrichment (recommended - adds all tier columns)
    ingestor = X4Ingestor(
        x4_db_path='data/x4sqlite1.db',
        output_path='data/exfor_enriched.parquet',
        ame2020_dir='data/'  # Directory containing *.mas20.txt files
    )
    df = ingestor.ingest()

    # Feature selection now just selects columns (no file I/O, no joins)
    import pandas as pd
    df_parquet = pd.read_parquet('data/exfor_enriched.parquet',
                                  columns=['Z', 'A', 'Energy', 'CrossSection',
                                          'Mass_Excess_keV', 'S_1n', 'Spin'])
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
    Legacy loader for AME2020 (Atomic Mass Evaluation 2020) isotopic data.

    **LEGACY:** This loader is used only for basic ingestion enrichment.
    It loads a single file (mass_1.mas20.txt) with mass excess and binding energy.

    **For tier-based features:** Use AME2020DataEnricher in nucml_next.data.enrichment
    which loads all AME2020/NUBASE2020 files:
    - mass_1.mas20.txt: Mass excess, binding energy
    - rct1.mas20.txt: Separation energies (S_2n, S_2p, Q-values)
    - rct2_1.mas20.txt: Separation energies (S_1n, S_1p, Q-values)
    - nubase_4.mas20.txt: Spin, parity, half-life (Tier D)

    Provides:
    - Mass excess and binding energy for basic ingestion enrichment
    - Falls back to SEMF approximation if file not available
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
    - Enrich with AME2020/NUBASE2020 data (all 5 files)
    - Write partitioned Parquet output with ALL enrichment columns

    Architecture Philosophy:
    - Pre-enrichment: Load AME2020/NUBASE2020 once during ingestion
    - All enrichment columns added to Parquet schema
    - Feature selection becomes simple column selection (no joins)
    - Production-ready: Single preprocessing step, consistent data
    """

    def __init__(
        self,
        x4_db_path: str,
        output_path: str = 'data/exfor_processed.parquet',
        ame2020_dir: Optional[str] = None,
        partitioning: List[str] = ['Z', 'A', 'MT'],
        max_partitions: int = 10000,
    ):
        """
        Initialize X4 ingestor with full AME2020/NUBASE2020 enrichment.

        Args:
            x4_db_path: Path to X4Pro SQLite database (e.g., x4sqlite1.db)
            output_path: Output path for Parquet dataset
            ame2020_dir: Optional directory containing AME2020/NUBASE2020 *.mas20.txt files
                        If provided, all 5 files will be loaded:
                        - mass_1.mas20.txt (Tier B, C)
                        - rct1.mas20.txt (Tier C, E)
                        - rct2_1.mas20.txt (Tier C, E)
                        - nubase_4.mas20.txt (Tier D)
                        - covariance.mas20.txt (optional)
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

        # Load AME2020/NUBASE2020 enrichment if requested
        if ame2020_dir:
            from nucml_next.data.enrichment import AME2020DataEnricher

            logger.info(f"Loading AME2020/NUBASE2020 from {ame2020_dir}")
            self.ame_enricher = AME2020DataEnricher(data_dir=ame2020_dir)
            self.ame_enricher.load_all()

            available_tiers = self.ame_enricher.get_available_tiers()
            logger.info(f"Available feature tiers: {available_tiers}")
        else:
            logger.info("No AME2020 enrichment requested")
            self.ame_enricher = None

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

            # Enrich with AME2020/NUBASE2020 if available
            if self.ame_enricher is not None:
                df = self._enrich_ame2020(df)
                logger.info(f"Enriched with AME2020/NUBASE2020 data")

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

        # Data quality filtering
        initial_count = len(df_norm)

        # Filter 1: Remove non-positive energies (Energy must be > 0)
        df_norm = df_norm[df_norm['Energy'] > 0]
        energy_filtered = initial_count - len(df_norm)
        if energy_filtered > 0:
            logger.info(f"Filtered {energy_filtered:,} points with Energy ≤ 0")

        # Filter 2: Remove non-positive cross sections (CrossSection must be > 0)
        df_norm = df_norm[df_norm['CrossSection'] > 0]
        xs_filtered = initial_count - energy_filtered - len(df_norm)
        if xs_filtered > 0:
            logger.info(f"Filtered {xs_filtered:,} points with CrossSection ≤ 0")

        # Filter 3: Remove infinite values
        df_norm = df_norm[np.isfinite(df_norm['Energy'])]
        df_norm = df_norm[np.isfinite(df_norm['CrossSection'])]
        inf_filtered = initial_count - energy_filtered - xs_filtered - len(df_norm)
        if inf_filtered > 0:
            logger.info(f"Filtered {inf_filtered:,} points with infinite values")

        # Filter 4: Remove unrealistic values
        # Energy: 1e-5 eV (thermal) to 1e9 eV (1 GeV)
        # CrossSection: 1e-10 barns to 1e6 barns
        df_norm = df_norm[
            (df_norm['Energy'] >= 1e-5) &
            (df_norm['Energy'] <= 1e9) &
            (df_norm['CrossSection'] >= 1e-10) &
            (df_norm['CrossSection'] <= 1e6)
        ]
        range_filtered = initial_count - energy_filtered - xs_filtered - inf_filtered - len(df_norm)
        if range_filtered > 0:
            logger.info(f"Filtered {range_filtered:,} points outside physical ranges")

        total_filtered = initial_count - len(df_norm)
        if total_filtered > 0:
            logger.info(f"Total filtered: {total_filtered:,} / {initial_count:,} points ({100*total_filtered/initial_count:.1f}%)")

        if len(df_norm) == 0:
            raise ValueError("No valid data points after quality filtering!")

        # Calculate neutron number (N = A - Z)
        # This is a fundamental nuclear property and should always be present
        df_norm['N'] = df_norm['A'] - df_norm['Z']

        return df_norm

    def _enrich_ame2020(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataset with ALL AME2020/NUBASE2020 properties.

        This method adds all available enrichment columns to the EXFOR dataframe:
        - Tier B/C: Mass excess, binding energy, binding per nucleon
        - Tier C: Separation energies (S_1n, S_2n, S_1p, S_2p)
        - Tier D: Spin, parity, isomer level, half-life
        - Tier E: All reaction Q-values (Q_alpha, Q_2beta_minus, etc.)

        Architecture Note:
        - This is pre-enrichment: Add ALL columns now
        - Feature selection happens later via column selection (no joins)
        - Parquet columnar format only loads needed columns anyway

        Args:
            df: Normalized DataFrame (already has N column)

        Returns:
            Enriched DataFrame with ALL available AME2020/NUBASE2020 columns
        """
        if self.ame_enricher is None:
            logger.info("No AME2020 enrichment requested - skipping")
            return df

        logger.info("Enriching with ALL AME2020/NUBASE2020 columns...")

        # Get full enrichment table (all columns)
        enrichment_table = self.ame_enricher.enrichment_table

        if enrichment_table is None or len(enrichment_table) == 0:
            logger.warning("Enrichment table is empty - skipping enrichment")
            return df

        # Merge ALL enrichment columns (except N which df already has)
        # Select all columns except 'N' (df already has it)
        enrich_cols = [col for col in enrichment_table.columns if col not in ['N']]
        enrichment_data = enrichment_table[enrich_cols].copy()

        # Merge with left join (keep all EXFOR data, add AME2020 where available)
        df_enriched = df.merge(
            enrichment_data,
            on=['Z', 'A'],
            how='left'
        )

        # Report enrichment coverage
        n_enriched = df_enriched['Mass_Excess_keV'].notna().sum() if 'Mass_Excess_keV' in df_enriched.columns else 0
        coverage = 100 * n_enriched / len(df_enriched) if len(df_enriched) > 0 else 0
        logger.info(f"Enriched {n_enriched:,} / {len(df_enriched):,} points ({coverage:.1f}% coverage)")

        # Log which enrichment columns were added
        added_cols = [col for col in df_enriched.columns if col not in df.columns]
        logger.info(f"Added {len(added_cols)} enrichment columns: {', '.join(added_cols[:10])}{'...' if len(added_cols) > 10 else ''}")

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
        """Print dataset summary statistics with tier-based enrichment coverage."""
        logger.info("="*70)
        logger.info("Dataset Summary")
        logger.info("="*70)
        logger.info(f"Total data points:       {len(df):,}")
        logger.info(f"Unique isotopes:         {df[['Z', 'A']].drop_duplicates().shape[0]}")
        logger.info(f"Unique reactions (MT):   {df['MT'].nunique()}")
        logger.info(f"Energy range:            {df['Energy'].min():.2e} - {df['Energy'].max():.2e} eV")
        logger.info(f"Points with uncertainty: {df['Uncertainty'].notna().sum():,}")

        # Report enrichment coverage by tier
        if self.ame_enricher is not None:
            logger.info("-" * 70)
            logger.info("AME2020/NUBASE2020 Enrichment Coverage")
            logger.info("-" * 70)

            # Tier B/C: Mass data
            if 'Mass_Excess_keV' in df.columns:
                coverage = df['Mass_Excess_keV'].notna().sum()
                pct = 100 * coverage / len(df) if len(df) > 0 else 0
                logger.info(f"Tier B/C (Mass):         {coverage:,} / {len(df):,} points ({pct:.1f}%)")

            # Tier C: Separation energies
            if 'S_1n' in df.columns:
                coverage = df['S_1n'].notna().sum()
                pct = 100 * coverage / len(df) if len(df) > 0 else 0
                logger.info(f"Tier C (S_1n):           {coverage:,} / {len(df):,} points ({pct:.1f}%)")

            # Tier D: Nuclear structure
            if 'Spin' in df.columns:
                coverage = df['Spin'].notna().sum()
                pct = 100 * coverage / len(df) if len(df) > 0 else 0
                logger.info(f"Tier D (Spin):           {coverage:,} / {len(df):,} points ({pct:.1f}%)")

            if 'Half_Life_s' in df.columns:
                coverage = df['Half_Life_s'].notna().sum()
                pct = 100 * coverage / len(df) if len(df) > 0 else 0
                logger.info(f"Tier D (Half-life):      {coverage:,} / {len(df):,} points ({pct:.1f}%)")

            # Tier E: Q-values
            if 'Q_alpha' in df.columns:
                coverage = df['Q_alpha'].notna().sum()
                pct = 100 * coverage / len(df) if len(df) > 0 else 0
                logger.info(f"Tier E (Q_alpha):        {coverage:,} / {len(df):,} points ({pct:.1f}%)")

            available_tiers = self.ame_enricher.get_available_tiers()
            logger.info(f"Available tiers:         {available_tiers}")

        logger.info("="*70)


# Convenience function
def ingest_x4(
    x4_db_path: str,
    output_path: str = 'data/exfor_processed.parquet',
    ame2020_dir: Optional[str] = None,
    max_partitions: int = 10000,
) -> pd.DataFrame:
    """
    Convenience function for X4 ingestion with full AME2020/NUBASE2020 enrichment.

    Args:
        x4_db_path: Path to X4Pro SQLite database
        output_path: Output Parquet path
        ame2020_dir: Optional directory containing AME2020/NUBASE2020 *.mas20.txt files
                    If provided, all 5 files will be loaded and merged into Parquet:
                    - mass_1.mas20.txt, rct1.mas20.txt, rct2_1.mas20.txt,
                      nubase_4.mas20.txt, covariance.mas20.txt
        max_partitions: Maximum number of partitions (default: 10000 for full EXFOR)

    Returns:
        Processed DataFrame with all EXFOR data + AME2020/NUBASE2020 enrichment

    Example:
        >>> # Basic ingestion (no enrichment)
        >>> df = ingest_x4('data/x4sqlite1.db', 'data/exfor.parquet')

        >>> # Full enrichment (all tiers)
        >>> df = ingest_x4(
        ...     x4_db_path='data/x4sqlite1.db',
        ...     output_path='data/exfor_enriched.parquet',
        ...     ame2020_dir='data/'
        ... )
        >>> # Parquet now contains all enrichment columns for tier-based feature selection
    """
    ingestor = X4Ingestor(
        x4_db_path=x4_db_path,
        output_path=output_path,
        ame2020_dir=ame2020_dir,
        max_partitions=max_partitions,
    )
    return ingestor.ingest()
