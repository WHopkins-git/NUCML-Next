"""
X4Pro SQLite Ingestor - Lean EXFOR Data Extraction
===================================================

Production-ready ingestion pipeline: X4Pro SQLite → Parquet (EXFOR data only).

Architecture: Lean Ingestion + On-Demand Enrichment
----------------------------------------------------
This ingestor extracts ONLY EXFOR experimental data:

1. Extract cross-section measurements from X4 SQLite database
2. Normalize to standard schema (Z, A, N, MT, Energy, CrossSection, Uncertainty)
3. Write lean Parquet file (no AME data duplication)

AME2020/NUBASE2020 enrichment happens during feature generation:
- AME files are small (~MB) and loaded once when needed
- Joined on-the-fly during feature generation based on (Z, A)
- No data duplication in Parquet
- Faster ingestion and smaller file size

Benefits:
- Lean Parquet: ~10x smaller without redundant AME data
- Fast ingestion: No AME joins during ingest
- Flexible: Users can choose enrichment tiers at runtime
- Same enrichment: AME enricher ensures consistency

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
  - Entry, Z, A, N, Projectile, MT, Energy, CrossSection, Uncertainty

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
                                          'Mass_Excess_keV', 'S_1n_keV', 'Spin'])
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
        partitioning: List[str] = [],  # No partitioning by default (changed from ['Z', 'A', 'MT'])
        max_partitions: int = 10000,
        run_svgp: bool = False,
        svgp_config: Optional[Any] = None,
        z_filter: Optional[List[int]] = None,
    ):
        """
        Initialize X4 ingestor - extracts lean EXFOR data only.

        Args:
            x4_db_path: Path to X4Pro SQLite database (e.g., x4sqlite1.db)
            output_path: Output path for Parquet dataset (default: data/exfor_processed.parquet)
            ame2020_dir: DEPRECATED - AME enrichment now happens during feature generation.
                        This parameter is kept for backward compatibility but ignored.
                        AME files are loaded on-demand by NucmlDataset during feature generation.
            partitioning: Partition columns for Parquet output (default: [] = no partitioning).
                         Options: [] (no partitioning), ['Z'] (by element), ['Z', 'A'] (by isotope).
                         WARNING: ['Z', 'A', 'MT'] creates 10K+ tiny files and 10-minute load times!
                         For datasets <10GB, no partitioning is recommended (fastest load, simplest).
            max_partitions: Maximum number of partitions allowed (default: 10000)
                           Only relevant if partitioning is enabled. Allows >1024 partitions.
            run_svgp: Run SVGP outlier detection after normalization. Default: False.
                     When True, adds z_score, gp_mean, gp_std columns to Parquet.
            svgp_config: SVGPConfig object with SVGP hyperparameters (device, epochs, etc.).
                        If None and run_svgp=True, uses SVGPConfig defaults.
            z_filter: Optional list of atomic numbers (Z) to include. Default: None (all elements).
                     Example: [17, 92] for Chlorine and Uranium only.
                     Useful for creating test subsets for faster development/testing.

        Note:
            This ingestor now produces LEAN Parquet files containing only EXFOR measurements.
            AME2020/NUBASE2020 enrichment is handled during feature generation to:
            - Reduce Parquet file size (~10x smaller)
            - Speed up ingestion
            - Avoid data duplication
            - Enable flexible tier selection at runtime
        """
        self.x4_db_path = Path(x4_db_path)
        self.output_path = Path(output_path)
        self.partitioning = partitioning
        self.max_partitions = max_partitions
        self.run_svgp = run_svgp
        self.svgp_config = svgp_config
        self.z_filter = z_filter

        if not self.x4_db_path.exists():
            raise FileNotFoundError(f"X4 database not found: {self.x4_db_path}")

        # AME enrichment has been moved to feature generation time
        # Keep this for backward compatibility (parameter is ignored)
        if ame2020_dir:
            logger.warning(
                "⚠️  ame2020_dir parameter is deprecated and will be ignored.\n"
                "   AME enrichment now happens during feature generation for better performance.\n"
                "   The lean Parquet file will not include AME columns.\n"
                "   NucmlDataset will load AME files automatically when needed."
            )

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
        if self.z_filter:
            logger.info(f"Z filter: {self.z_filter} (subset mode)")
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

            # Apply Z filter if specified (subset mode)
            if self.z_filter:
                before = len(df)
                df = df[df['Z'].isin(self.z_filter)]
                logger.info(f"Z filter applied: {before:,} -> {len(df):,} points "
                           f"(Z in {self.z_filter})")

            # SVGP outlier detection (optional)
            if self.run_svgp:
                logger.info("Running SVGP outlier detection...")
                from nucml_next.data.outlier_detection import SVGPOutlierDetector, SVGPConfig
                config = self.svgp_config if self.svgp_config is not None else SVGPConfig()
                detector = SVGPOutlierDetector(config)
                df = detector.score_dataframe(df)
                n_outliers = (df['z_score'] > 3).sum()
                logger.info(f"SVGP scoring complete: {n_outliers:,} outliers at z>3 "
                           f"({100 * n_outliers / len(df):.2f}%)")

            # Write lean Parquet (EXFOR data only, plus optional SVGP columns)
            # AME enrichment now happens during feature generation
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

        IMPORTANT - Unit Handling:
        --------------------------
        The c5data (computational format) is ALREADY unit-normalized by X4Pro:
        - Energy (x1): Always in eV (electronvolts)
        - CrossSection (y): Always in barns (b)
        - Uncertainty (dy): Always in barns (b)

        This is the "C5" computational format used by ENDF/B and all major nuclear
        data libraries. X4Pro performs unit conversion from the original EXFOR format
        (which may have keV, MeV, mb, μb, etc.) to standard eV/barns during ingestion.

        Reference: https://www-nds.iaea.org/nrdc/basics/exfor-basics-3.html
        "The standard units used by the translation program were selected to be
        the same as the units used by ENDF (e.g., eV, barns, etc.)"

        Args:
            conn: SQLite connection

        Returns:
            DataFrame with columns: [DatasetID, Z, A, MT, Energy, Data, dData]
        """
        import json
        import re

        # Query to get dataset metadata and JSON data
        # IMPORTANT: DatasetID is a string (e.g., "30649005S"), so quote properly
        # Extract Proj (projectile) column for explicit neutron/charged particle filtering
        query = """
            SELECT
                ds.DatasetID,
                ds.zTarg1 as Z,
                ds.Targ1 as Target,
                ds.Proj as Projectile,
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

        for dataset_id, z, target, projectile, mt, jx5z_str in rows:
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

                # Extract uncertainties - nested inside their parent objects
                # Cross-section uncertainty: c5data.y.dy (NOT c5data.dy)
                # Energy uncertainty: c5data.x1.dx1 (less common, ~20% of records)
                xs_uncertainties = c5.get('y', {}).get('dy', [None] * len(energies))
                energy_uncertainties = c5.get('x1', {}).get('dx1', [None] * len(energies))

                # Ensure all arrays have same length
                n_points = min(len(energies), len(cross_sections))
                if n_points == 0:
                    continue

                # Extend uncertainties if needed
                if len(xs_uncertainties) < n_points:
                    xs_uncertainties = xs_uncertainties + [None] * (n_points - len(xs_uncertainties))
                if len(energy_uncertainties) < n_points:
                    energy_uncertainties = energy_uncertainties + [None] * (n_points - len(energy_uncertainties))

                # Create data points
                for i in range(n_points):
                    all_points.append({
                        'DatasetID': dataset_id,
                        'Z': z,
                        'A': a,
                        'Projectile': projectile,  # Add projectile for explicit filtering
                        'MT': mt,
                        'En': energies[i],
                        'dEn': energy_uncertainties[i],  # Energy uncertainty (eV)
                        'Data': cross_sections[i],
                        'dData': xs_uncertainties[i]  # Cross-section uncertainty (barns)
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

        IMPORTANT - Unit Handling:
        --------------------------
        The c5dat table contains "C5" computational format data that is ALREADY
        unit-normalized by X4Pro:
        - x1 (Energy): Always in eV (electronvolts)
        - y (CrossSection): Always in barns (b)
        - dy (Uncertainty): Always in barns (b)

        X4Pro performs unit conversion from the original EXFOR format (which may have
        keV, MeV, mb, μb, etc.) to standard eV/barns during database creation.

        Args:
            conn: SQLite connection

        Returns:
            DataFrame with columns: [DatasetID, Z, A, MT, En, Data, dData]
        """
        import re

        # INNER JOIN between metadata and data tables
        # DatasetID must be quoted as it's a string
        # Extract Proj (projectile) for explicit neutron/charged particle filtering
        query = """
            SELECT
                ds.DatasetID,
                ds.zTarg1 as Z,
                ds.Targ1 as Target,
                ds.Proj as Projectile,
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

            # Projectile (X4Pro: Proj → Projectile)
            'proj': 'Projectile',
            'projectile': 'Projectile',
            'particle': 'Projectile',

            # Data (including X4Pro naming conventions)
            'energy': 'Energy',
            'energy_value': 'Energy',
            'en': 'Energy',        # X4Pro: En → Energy

            'den': 'Energy_Uncertainty',  # X4Pro: dEn → Energy_Uncertainty
            'd_energy': 'Energy_Uncertainty',
            'energy_error': 'Energy_Uncertainty',

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

        # Add Uncertainty if missing (cross-section uncertainty)
        if 'Uncertainty' not in df_norm.columns:
            df_norm['Uncertainty'] = np.nan

        # Add Energy_Uncertainty if missing (energy uncertainty, less common ~20%)
        if 'Energy_Uncertainty' not in df_norm.columns:
            df_norm['Energy_Uncertainty'] = np.nan

        # Add Projectile if missing (legacy data may not have it)
        if 'Projectile' not in df_norm.columns:
            logger.warning("Projectile column not found in source data - will rely on MT codes for filtering")
            df_norm['Projectile'] = None  # Will be inferred from MT codes if needed

        # Select and order final columns
        # Note: Energy_Uncertainty added for completeness but rarely used in ML training
        final_cols = ['Entry', 'Z', 'A', 'Projectile', 'MT', 'Energy', 'Energy_Uncertainty', 'CrossSection', 'Uncertainty']
        df_norm = df_norm[final_cols]

        # Clean data
        df_norm = df_norm.dropna(subset=['Z', 'A', 'MT', 'Energy', 'CrossSection'])
        df_norm['Z'] = df_norm['Z'].astype(int)
        df_norm['A'] = df_norm['A'].astype(int)
        df_norm['MT'] = df_norm['MT'].astype(int)
        df_norm['Energy'] = df_norm['Energy'].astype(float)
        df_norm['CrossSection'] = df_norm['CrossSection'].astype(float)

        # =====================================================================
        # UNIT VERIFICATION SANITY CHECK
        # =====================================================================
        # The X4Pro C5 format should have Energy in eV and CrossSection in barns.
        # Log warnings if data appears to be in wrong units.
        #
        # Expected ranges:
        # - Energy: 1e-5 eV (thermal) to 1e9 eV (1 GeV) - nuclear reaction range
        # - CrossSection: 1e-10 b (very small) to 1e6 b (compound nuclei) - nuclear scale
        #
        # If median energy >> 1e9 or << 1e-5, data may be in keV/MeV not eV
        # If median XS << 1e-10 or >> 1e6, data may be in mb/μb not barns

        median_energy = df_norm['Energy'].median()
        median_xs = df_norm['CrossSection'].median()

        # Check if energy might be in wrong units
        if median_energy > 1e12:  # Median > 1 TeV suggests MeV not eV
            logger.warning(f"UNIT WARNING: Median energy {median_energy:.2e} suggests data may be in keV or MeV, not eV!")
            logger.warning("  Expected: eV (electronvolts). X4Pro C5 format should be pre-converted.")
        elif median_energy < 1e-8:  # Median < 10 neV suggests eV is correct but very low energy
            logger.info(f"Note: Very low median energy {median_energy:.2e} eV (ultra-cold neutrons?)")

        # Check if cross section might be in wrong units
        if median_xs < 1e-15:  # Median < 1 femtobarn suggests mb or μb not barns
            logger.warning(f"UNIT WARNING: Median XS {median_xs:.2e} suggests data may be in mb or μb, not barns!")
            logger.warning("  Expected: barns. X4Pro C5 format should be pre-converted.")
        elif median_xs > 1e9:  # Median > 1 Gb suggests kilobarns reported as barns
            logger.warning(f"UNIT WARNING: Median XS {median_xs:.2e} seems very large for barns!")

        # Data quality filtering
        initial_count = len(df_norm)

        # Filter 0: Remove natural element targets (A=0)
        # EXFOR uses A=0 to indicate natural element targets (e.g., Mo-NAT, Fe-NAT)
        # These cannot be enriched with AME2020 data (which requires specific isotopes)
        # and produce invalid N = A - Z = -Z values
        natural_mask = df_norm['A'] == 0
        natural_filtered = natural_mask.sum()
        if natural_filtered > 0:
            df_norm = df_norm[~natural_mask]
            logger.info(f"Filtered {natural_filtered:,} natural element targets (A=0)")

        # Filter 1: Remove non-positive energies (Energy must be > 0)
        before_energy = len(df_norm)
        df_norm = df_norm[df_norm['Energy'] > 0]
        energy_filtered = before_energy - len(df_norm)
        if energy_filtered > 0:
            logger.info(f"Filtered {energy_filtered:,} points with Energy ≤ 0")

        # Filter 2: Remove non-positive cross sections (CrossSection must be > 0)
        before_xs = len(df_norm)
        df_norm = df_norm[df_norm['CrossSection'] > 0]
        xs_filtered = before_xs - len(df_norm)
        if xs_filtered > 0:
            logger.info(f"Filtered {xs_filtered:,} points with CrossSection ≤ 0")

        # Filter 3: Remove infinite values
        before_inf = len(df_norm)
        df_norm = df_norm[np.isfinite(df_norm['Energy'])]
        df_norm = df_norm[np.isfinite(df_norm['CrossSection'])]
        inf_filtered = before_inf - len(df_norm)
        if inf_filtered > 0:
            logger.info(f"Filtered {inf_filtered:,} points with infinite values")

        # Filter 4: Remove unrealistic values
        # Energy: 1e-5 eV (thermal) to 1e9 eV (1 GeV)
        # CrossSection: 1e-10 barns to 1e6 barns
        before_range = len(df_norm)
        df_norm = df_norm[
            (df_norm['Energy'] >= 1e-5) &
            (df_norm['Energy'] <= 1e9) &
            (df_norm['CrossSection'] >= 1e-10) &
            (df_norm['CrossSection'] <= 1e6)
        ]
        range_filtered = before_range - len(df_norm)
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

        Note:
            If output path already exists, it will be deleted and recreated to
            prevent accumulation of duplicate partition files from multiple ingestion runs.

        Args:
            df: DataFrame to write
        """
        # Clean up existing output directory to prevent duplicate partitions
        # PyArrow's 'overwrite_or_ignore' doesn't delete old files, causing duplicates
        if self.output_path.exists():
            logger.info(f"Removing existing Parquet directory: {self.output_path}")
            import shutil
            shutil.rmtree(self.output_path)

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
        """Print dataset summary statistics (lean EXFOR data only)."""
        logger.info("="*70)
        logger.info("Dataset Summary")
        logger.info("="*70)
        logger.info(f"Total data points:       {len(df):,}")
        logger.info(f"Unique isotopes:         {df[['Z', 'A']].drop_duplicates().shape[0]}")
        logger.info(f"Unique reactions (MT):   {df['MT'].nunique()}")
        logger.info(f"Energy range:            {df['Energy'].min():.2e} - {df['Energy'].max():.2e} eV")

        # Uncertainty statistics
        xs_unc_count = df['Uncertainty'].notna().sum()
        xs_unc_pct = 100 * xs_unc_count / len(df) if len(df) > 0 else 0
        logger.info(f"With XS uncertainty:     {xs_unc_count:,} ({xs_unc_pct:.1f}%)")

        if 'Energy_Uncertainty' in df.columns:
            e_unc_count = df['Energy_Uncertainty'].notna().sum()
            e_unc_pct = 100 * e_unc_count / len(df) if len(df) > 0 else 0
            logger.info(f"With energy uncertainty: {e_unc_count:,} ({e_unc_pct:.1f}%)")

        logger.info("="*70)


# Convenience function
def ingest_x4(
    x4_db_path: str,
    output_path: str = 'data/exfor_processed.parquet',
    ame2020_dir: Optional[str] = None,
    max_partitions: int = 10000,
    run_svgp: bool = False,
    svgp_config: Optional[Any] = None,
    z_filter: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Convenience function for lean X4 ingestion (EXFOR data only).

    Args:
        x4_db_path: Path to X4Pro SQLite database
        output_path: Output Parquet path (default: data/exfor_processed.parquet)
        ame2020_dir: DEPRECATED - This parameter is ignored.
                    AME enrichment now happens during feature generation.
        max_partitions: Maximum number of partitions (default: 10000 for full EXFOR)
        run_svgp: Run SVGP outlier detection (adds z_score column). Default: False.
        svgp_config: SVGPConfig object with SVGP hyperparameters. Default: None.
        z_filter: Optional list of atomic numbers (Z) to include. Default: None (all).
                 Example: [17, 92] for Chlorine and Uranium only.

    Returns:
        Processed DataFrame with EXFOR data (lean, no AME duplication)

    Example:
        >>> # Basic ingestion (no enrichment)
        >>> df = ingest_x4('data/x4sqlite1.db', 'data/exfor.parquet')

        >>> # With SVGP outlier detection
        >>> from nucml_next.data.outlier_detection import SVGPConfig
        >>> df = ingest_x4(
        ...     x4_db_path='data/x4sqlite1.db',
        ...     output_path='data/exfor.parquet',
        ...     run_svgp=True,
        ...     svgp_config=SVGPConfig(device='cuda'),
        ... )
        >>> # Parquet now contains z_score column for outlier filtering

        >>> # Test subset (Uranium + Chlorine only)
        >>> df = ingest_x4(
        ...     x4_db_path='data/x4sqlite1.db',
        ...     output_path='data/exfor_test.parquet',
        ...     z_filter=[17, 92],  # Cl=17, U=92
        ...     run_svgp=True,
        ... )
    """
    ingestor = X4Ingestor(
        x4_db_path=x4_db_path,
        output_path=output_path,
        ame2020_dir=ame2020_dir,
        max_partitions=max_partitions,
        run_svgp=run_svgp,
        svgp_config=svgp_config,
        z_filter=z_filter,
    )
    return ingestor.ingest()
