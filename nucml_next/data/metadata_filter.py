"""
EXFOR Metadata Filter -- Non-Pure Data Exclusion
=================================================

Filters EXFOR datasets based on REACTION subfield metadata to exclude
data that is not point-wise absolute cross sections.

This implements the equivalent of the IAEA Data Explorer's
"Exclude non pure data" checkbox, but without applying C5
renormalisations (which would introduce evaluator bias).

Background
----------
EXFOR REACTION strings have subfields (SF1-SF9). For a "pure"
point-wise absolute cross section measurement:
- SF6 = 'SIG' (cross section parameter)
- SF8 is NULL (no modifier -- excludes REL, MXW, SPA, etc.)
- SF5 is NULL (no branch qualifier -- excludes PRE, SEC, etc.)

Non-pure entries include:
- Relative data (SF8=REL): arbitrary normalisation, not absolute barns
- Ratio data (SF6=SIG/RAT or compound fullCode): dimensionless ratios
- Spectrum-averaged (SF8=MXW/SPA/FIS/AV): integral averages, not pointwise
- Spectrum-weighted (SF8=BRA/BRS): bremsstrahlung-weighted averages
- Modified quantities (SF8=RTE/FCT/RAT): transformed representations
- Non-experimental (SF9=CALC/DERIV/EVAL/RECOM): not real measurements

These produce systematic clusters of "outliers" that no GP can learn
to identify because they're not random noise -- they're data with
fundamentally different meaning/units plotted on the same axes.

Usage
-----
    from nucml_next.data.metadata_filter import MetadataFilter

    # During ingestion, after extraction but before normalisation
    mf = MetadataFilter(conn)
    df = mf.enrich_and_filter(df, exclude_non_pure=True)

Author: NUCML-Next Team
"""

import sqlite3
import logging
import re
from typing import Optional, Set, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# SF8 modifier codes that indicate non-pure cross section data
# Reference: EXFOR Dictionary 34 (Reaction quantity modifiers)
# =====================================================================

# Tier 1: Definitely NOT absolute point-wise cross sections
# These have fundamentally different units or meaning
SF8_EXCLUDE_ABSOLUTE = {
    'REL',    # Relative measurement (arbitrary units, not barns)
    'RAT',    # Ratio to another quantity (dimensionless)
    'RTE',    # Rate (sigma * v, different units)
    'FCT',    # Arbitrary factor applied
}

# Tier 2: Spectrum-averaged or integral quantities
# The "energy" field has different meaning (characteristic E, not pointwise)
SF8_EXCLUDE_AVERAGED = {
    'MXW',    # Maxwellian-averaged (E = kT)
    'SPA',    # Spectrum-averaged (reactor spectrum)
    'FIS',    # Fission-spectrum averaged
    'AV',     # Averaged over energy interval
    'BRA',    # Bremsstrahlung-spectrum weighted average
    'BRS',    # Bremsstrahlung-spectrum weighted (4pi)
}

# Tier 3: Other modifiers that may warrant exclusion
# These are less clearly "wrong" but indicate non-standard data
SF8_EXCLUDE_OTHER = {
    'ETA',    # eta = nu * sigma_f / sigma_a (derived quantity)
    'ALF',    # alpha = sigma_gamma / sigma_f (ratio)
    'RES',    # Resonance integral (integrated over energy)
    'G',      # g-factor modified (Westcott convention)
    'S0',     # S-wave neutron strength function (resonance parameter, eV^-1/2)
    '2G',     # 2g*Gamma_n^0 spin-statistical factor (resonance parameter)
    'RM',     # R-Matrix parameters (model fit, not direct measurement)
}
# NOTE: RAW (uncorrected data), TTA (thick-target approximation), and
# SDT (measurement technique tag) are intentionally KEPT in the dataset.
# RAW = real measurements before corrections (69K points);
# TTA = valid measurement technique for charged particles (17K);
# SDT = measurement technique descriptor, not a data modifier (2.4K).

# Default: Tiers 1 + 2 (safe, unambiguous exclusions)
SF8_EXCLUDE_DEFAULT = SF8_EXCLUDE_ABSOLUTE | SF8_EXCLUDE_AVERAGED

# Full: All three tiers
SF8_EXCLUDE_FULL = SF8_EXCLUDE_DEFAULT | SF8_EXCLUDE_OTHER


# =====================================================================
# SF9 data type codes -- non-experimental data
# Reference: EXFOR Dictionary 36 (Quantity data types)
# =====================================================================

# These are NOT experimental measurements and should be excluded
# from an ML training set that learns from experiments
SF9_EXCLUDE = {
    'CALC',   # Calculated (theoretical prediction, not measurement)
    'DERIV',  # Derived from other measurements (not independent)
    'EVAL',   # Evaluated value (would create circular dependency)
    'RECOM',  # Recommended value (same as EVAL effectively)
}

# Note: SF9=EXP (experimental) or SF9=NULL are both acceptable.

# =====================================================================
# SF5 branch codes that modify what's being measured
# =====================================================================
SF5_EXCLUDE = {
    # ── Original branch codes ──
    'PRE',    # Pre-neutron-emission (prompt fission, not total)
    'SEC',    # Secondary (post-neutron-emission)
    'TER',    # Ternary (ternary fission)
    'QTR',    # Quaternary
    'DL',     # Delayed (delayed neutrons, not total)
    'PAR',    # Partial (partial cross section to specific level)
    # ── Fission-yield branch codes ──
    'CUM',    # Cumulative yield
    '(CUM)',  # Cumulative yield (EXFOR bracket notation)
    '(CUM)/M+',  # Cumulative including metastable
    'CHN',    # Chain yield
    'IND',    # Independent yield
    'UNW',    # Unweighted
    # ── Level-specific / metastable codes ──
    '(M)',    # Metastable state only
    'M+',     # Metastable + ground
    'EXL',    # Exclusive (specific channel)
    'POT',    # Potential scattering
    # ── Numeric level codes (partial XS to excited states) ──
    '1', '2', '3', '4',
}

# =====================================================================
# SF6 quantity codes -- what we WANT (whitelist approach)
# =====================================================================
SF6_ACCEPT = {
    'SIG',    # Cross section (the primary quantity we want)
    'WID',    # Width (resonance width -- keep for resonance studies)
}

# SF6 codes that are definitely not cross sections (for documentation)
SF6_EXCLUDE = {
    'SIG/RAT',  # Cross section ratio (dimensionless)
    'DA',        # Differential cross section d-sigma/d-Omega
    'DE',        # Energy differential d-sigma/dE
    'DA/DE',     # Double differential d2-sigma/d-Omega-dE
    'RI',        # Resonance integral
    'INT',       # Integrated cross section
    'ARE',       # Area (resonance area)
    'KER',       # KERMA factor
    'FY',        # Fission yield (not a cross section)
    'NU',        # Neutron multiplicity
    'ETA',       # eta parameter
    'ALF',       # alpha = sigma_gamma / sigma_f
}


class MetadataFilter:
    """
    Filters EXFOR datasets based on REACTION metadata subfields.

    Discovers the X4Pro schema at runtime and extracts sf5, sf6, sf8
    from the REACODE table (or JSON metadata) to enable principled
    filtering of non-pure data.

    This is TYPE FILTERING, not evaluator bias:
    - Relative data literally has different units
    - Ratio data is dimensionless, not in barns
    - Spectrum-averaged data has different x-axis meaning
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialise with a database connection.

        Args:
            conn: SQLite connection to X4Pro database
        """
        self.conn = conn
        self._schema = self._discover_schema()

    def _discover_schema(self) -> Dict:
        """
        Discover available tables and columns for metadata filtering.

        Returns:
            Dict describing available filtering capabilities
        """
        cursor = self.conn.cursor()

        schema = {
            'has_reacode': False,
            'reacode_cols': [],
            'has_fullcode': False,
            'fullcode_col': None,
            'has_sf_columns': False,
            'sf_columns': {},   # maps 'sf5' -> actual column name
            'has_reac_combi': False,
            'reac_combi_col': None,
            'join_key': None,   # column name to join REACODE to x4pro_ds
            'has_spsdd': False,
            'spsdd_source': None,
        }

        # Check for REACODE table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='REACODE'")
        if cursor.fetchone():
            schema['has_reacode'] = True
            cursor.execute("PRAGMA table_info(REACODE)")
            cols = [(c[1], c[2]) for c in cursor.fetchall()]
            schema['reacode_cols'] = [c[0] for c in cols]

            col_names_lower = {c[0].lower(): c[0] for c in cols}

            # Find fullCode column (lowercase 'f' in X4Pro)
            for candidate in ['fullcode', 'reacode', 'full_code', 'code']:
                if candidate in col_names_lower:
                    schema['has_fullcode'] = True
                    schema['fullcode_col'] = col_names_lower[candidate]
                    break

            # Find reacCombi column (identifies compound/ratio reactions)
            for candidate in ['reaccombi', 'reac_combi', 'reactioncombi']:
                if candidate in col_names_lower:
                    schema['has_reac_combi'] = True
                    schema['reac_combi_col'] = col_names_lower[candidate]
                    break

            # Find SF columns (direct, if available)
            for sf in ['sf5', 'sf6', 'sf7', 'sf8', 'sf9']:
                for variant in [sf, sf.upper(), f'SF{sf[-1]}']:
                    if variant.lower() in col_names_lower:
                        schema['sf_columns'][sf] = col_names_lower[variant.lower()]
                        schema['has_sf_columns'] = True
                        break

            # Find join key (REACODE -> x4pro_ds)
            for candidate in ['ReacodeID', 'DatasetID', 'reacodeID', 'datasetID', 'ID']:
                if candidate.lower() in col_names_lower:
                    schema['join_key'] = col_names_lower[candidate.lower()]
                    break

        # Check for SUBENT table with SPSDD column (superseded status)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [r[0] for r in cursor.fetchall()]

        for table in all_tables:
            cursor.execute(f"PRAGMA table_info([{table}])")
            table_cols = {c[1].lower(): c[1] for c in cursor.fetchall()}
            if 'spsdd' in table_cols:
                schema['has_spsdd'] = True
                schema['spsdd_source'] = table
                schema['spsdd_col'] = table_cols['spsdd']
                break

        # Log discovery results
        if schema['has_reacode']:
            logger.info(f"REACODE table found with {len(schema['reacode_cols'])} columns")
            if schema['has_sf_columns']:
                logger.info(f"  SF columns available: {schema['sf_columns']}")
            elif schema['has_fullcode']:
                logger.info(f"  fullCode column found: {schema['fullcode_col']} "
                            f"(will parse SF fields from reaction string)")
            else:
                logger.warning("  No SF columns or fullCode found in REACODE "
                               "-- cannot filter by reaction type")
            if schema['has_reac_combi']:
                logger.info(f"  reacCombi column found: {schema['reac_combi_col']} "
                            f"(will detect compound/ratio reactions)")
        else:
            logger.warning("REACODE table not found -- metadata filtering will be limited")

        if schema['has_spsdd']:
            logger.info(f"Superseded status available via {schema['spsdd_source']}.SPSDD")

        return schema

    def get_dataset_metadata(self) -> Optional[pd.DataFrame]:
        """
        Extract reaction metadata for all datasets.

        Returns:
            DataFrame with columns: [DatasetID, sf5, sf6, sf8, sf9, reacCombi]
            or None if metadata not available
        """
        if not self._schema['has_reacode']:
            return None

        join_key = self._schema['join_key'] or 'ReacodeID'

        # Strategy 1: Direct SF columns available
        if self._schema['has_sf_columns']:
            sf_cols = self._schema['sf_columns']

            select_parts = [f"[{join_key}] as DatasetID"]
            for sf_name, col_name in sf_cols.items():
                select_parts.append(f"[{col_name}] as {sf_name}")

            if self._schema['has_fullcode']:
                select_parts.append(f"[{self._schema['fullcode_col']}] as FullCode")
            if self._schema['has_reac_combi']:
                select_parts.append(f"[{self._schema['reac_combi_col']}] as reacCombi")

            query = f"SELECT {', '.join(select_parts)} FROM REACODE"
            df = pd.read_sql_query(query, self.conn)

            for sf in ['sf5', 'sf6', 'sf8', 'sf9']:
                if sf not in df.columns:
                    df[sf] = None

            return df

        # Strategy 2: Parse fullCode to extract SF fields
        elif self._schema.get('has_fullcode'):
            fullcode_col = self._schema['fullcode_col']

            select_parts = [
                f"[{join_key}] as DatasetID",
                f"[{fullcode_col}] as FullCode",
            ]
            if self._schema['has_reac_combi']:
                select_parts.append(f"[{self._schema['reac_combi_col']}] as reacCombi")

            query = f"SELECT {', '.join(select_parts)} FROM REACODE"
            df = pd.read_sql_query(query, self.conn)

            # Detect compound/ratio reactions via reacCombi BEFORE parsing fullCode
            # reacCombi = 'R1#' -> single reaction, safe to parse
            # reacCombi contains '/' -> ratio, skip fullCode parsing
            # reacCombi contains '+' -> sum, skip fullCode parsing
            if 'reacCombi' in df.columns:
                is_compound = df['reacCombi'].fillna('').str.contains(r'[/+]', regex=True)
                is_ratio = df['reacCombi'].fillna('').str.contains('/', regex=False)
                is_sum = df['reacCombi'].fillna('').str.contains('+', regex=False)
                logger.info(f"  Compound reactions: {is_compound.sum():,} "
                            f"(ratio: {is_ratio.sum():,}, sum: {is_sum.sum():,})")
            else:
                is_compound = pd.Series(False, index=df.index)
                is_ratio = pd.Series(False, index=df.index)

            # Parse SF fields only for simple (non-compound) reactions
            simple_mask = ~is_compound
            df['sf5'] = None
            df['sf6'] = None
            df['sf8'] = None
            df['sf9'] = None

            if simple_mask.any():
                simple_codes = df.loc[simple_mask, 'FullCode']
                df.loc[simple_mask, 'sf5'] = simple_codes.apply(self._parse_sf5)
                df.loc[simple_mask, 'sf6'] = simple_codes.apply(self._parse_sf6)
                df.loc[simple_mask, 'sf8'] = simple_codes.apply(self._parse_sf8)
                df.loc[simple_mask, 'sf9'] = simple_codes.apply(self._parse_sf9)

            # Mark compound ratio reactions explicitly
            df.loc[is_ratio, 'sf6'] = 'SIG/RAT'

            return df

        return None

    def _get_superseded_ids(self) -> Optional[Set[str]]:
        """
        Get DatasetIDs that are superseded (SPSDD='1' or 'D').

        SUBENT.SPSDD values:
        - '0' or NULL: not superseded (keep)
        - '1': superseded by another entry
        - 'D': deprecated/deleted

        Returns:
            Set of superseded SubentIDs, or None if SPSDD not available
        """
        if not self._schema['has_spsdd']:
            return None

        table = self._schema['spsdd_source']
        col = self._schema['spsdd_col']

        query = f"""
            SELECT SubentID FROM [{table}]
            WHERE [{col}] IN ('1', 'D')
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            superseded = {row[0] for row in cursor.fetchall()}
            logger.info(f"Found {len(superseded):,} superseded subentries in {table}")
            return superseded
        except Exception as e:
            logger.warning(f"Failed to query superseded status: {e}")
            return None

    @staticmethod
    def _parse_sf5(fullcode: str) -> Optional[str]:
        """Extract SF5 (branch/isomer qualifier) from EXFOR REACTION string.

        Format: TARGET(PROJ,PROCESS)RESIDUAL,SF5,SF6,SF7,SF8,SF9
        SF5 is the first comma-delimited field after the residual nucleus.
        """
        if not fullcode or not isinstance(fullcode, str):
            return None
        match = re.search(r'\)[^,]*,([^,]*),', fullcode)
        if match:
            val = match.group(1).strip()
            return val if val else None
        return None

    @staticmethod
    def _parse_sf6(fullcode: str) -> Optional[str]:
        """Extract SF6 (quantity parameter) from EXFOR REACTION string.

        SF6 is the second comma-separated field after the residual.
        """
        if not fullcode or not isinstance(fullcode, str):
            return None
        match = re.search(r'\)[^,]*,[^,]*,([^,)]+)', fullcode)
        if match:
            val = match.group(1).strip()
            return val if val else None
        return None

    @staticmethod
    def _parse_sf8(fullcode: str) -> Optional[str]:
        """Extract SF8 (modifier) from EXFOR REACTION string.

        SF8 is the fourth comma-separated field after the residual.
        Pattern: )RESIDUAL,SF5,SF6,SF7,SF8
        """
        if not fullcode or not isinstance(fullcode, str):
            return None
        # Split on the LAST closing paren (handles nested targets)
        parts = fullcode.rsplit(')', 1)
        if len(parts) < 2:
            return None
        after_residual = parts[1].rstrip(')')
        fields = after_residual.split(',')
        # fields[0] = residual (may be empty), [1]=sf5, [2]=sf6, [3]=sf7, [4]=sf8
        if len(fields) >= 5:
            val = fields[4].strip()
            return val if val else None
        return None

    @staticmethod
    def _parse_sf9(fullcode: str) -> Optional[str]:
        """Extract SF9 (data type: EXP/CALC/DERIV/EVAL) from EXFOR REACTION string.

        SF9 is the fifth comma-separated field after the residual.
        """
        if not fullcode or not isinstance(fullcode, str):
            return None
        parts = fullcode.rsplit(')', 1)
        if len(parts) < 2:
            return None
        after_residual = parts[1].rstrip(')')
        fields = after_residual.split(',')
        # fields[0] = residual, [1]=sf5, [2]=sf6, [3]=sf7, [4]=sf8, [5]=sf9
        if len(fields) >= 6:
            val = fields[5].strip()
            return val if val else None
        return None

    def classify_dataset(
        self,
        sf5: Optional[str],
        sf6: Optional[str],
        sf8: Optional[str],
        sf9: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Classify a dataset based on its REACTION subfields.

        Returns:
            Dict with classification flags:
            - is_pure: True if absolute point-wise cross section
            - is_relative: True if relative data (SF8=REL)
            - is_ratio: True if ratio data (SF6 contains RAT)
            - is_averaged: True if spectrum-averaged (SF8 in MXW/SPA/FIS/AV)
            - is_partial: True if partial/branch measurement (SF5 set)
            - is_non_xs: True if SF6 is not SIG (different quantity entirely)
            - is_non_experimental: True if SF9 indicates calculated/derived/evaluated
        """
        sf5 = sf5.strip().upper() if isinstance(sf5, str) and sf5.strip() else None
        sf6 = sf6.strip().upper() if isinstance(sf6, str) and sf6.strip() else None
        sf8 = sf8.strip().upper() if isinstance(sf8, str) and sf8.strip() else None
        sf9 = sf9.strip().upper() if isinstance(sf9, str) and sf9.strip() else None

        # Handle compound SF8 codes (e.g. "BRS/REL") by splitting on "/"
        sf8_tokens = set(sf8.split('/')) if sf8 else set()
        is_relative = bool(sf8_tokens & SF8_EXCLUDE_ABSOLUTE)
        is_averaged = bool(sf8_tokens & SF8_EXCLUDE_AVERAGED)
        is_other_modified = bool(sf8_tokens & SF8_EXCLUDE_OTHER)
        is_ratio = sf6 is not None and 'RAT' in sf6
        is_partial = sf5 in SF5_EXCLUDE if sf5 else False
        is_non_xs = sf6 not in SF6_ACCEPT if sf6 else False
        is_non_experimental = sf9 in SF9_EXCLUDE if sf9 else False

        is_pure = (
            not is_relative
            and not is_averaged
            and not is_ratio
            and not is_partial
            and not is_non_xs
            and not is_non_experimental
            and not is_other_modified
        )

        return {
            'is_pure': is_pure,
            'is_relative': is_relative,
            'is_ratio': is_ratio,
            'is_averaged': is_averaged,
            'is_partial': is_partial,
            'is_non_xs': is_non_xs,
            'is_non_experimental': is_non_experimental,
        }

    def enrich_and_filter(
        self,
        df: pd.DataFrame,
        exclude_non_pure: bool = True,
        exclude_superseded: bool = True,
        sf8_exclude_set: Optional[Set[str]] = None,
        keep_metadata_columns: bool = True,
        diagnostics: bool = False,
    ) -> pd.DataFrame:
        """
        Enrich DataFrame with reaction metadata and optionally filter.

        This should be called AFTER extraction but BEFORE normalisation
        in the ingestion pipeline (while DatasetID still exists).

        Args:
            df: DataFrame with a 'DatasetID' or 'Entry' column
            exclude_non_pure: Remove non-pure data (default True)
            exclude_superseded: Remove superseded entries (default True)
            sf8_exclude_set: Custom SF8 codes to exclude (default: SF8_EXCLUDE_DEFAULT)
            keep_metadata_columns: Keep sf5, sf6, sf8, is_pure columns in output
            diagnostics: Include FullCode column in output for interactive inspection

        Returns:
            Enriched (and optionally filtered) DataFrame with new columns:
            - sf5, sf6, sf8, sf9: REACTION subfield values
            - is_pure: Boolean classification
            - data_type: Descriptive string ('absolute', 'relative', 'ratio', etc.)
        """
        if sf8_exclude_set is None:
            sf8_exclude_set = SF8_EXCLUDE_DEFAULT

        initial_count = len(df)
        logger.info(f"Metadata filter: starting with {initial_count:,} data points")

        # ---- Diagnostic: log reatyp distribution if available ----
        self._log_reatyp_distribution()

        # ---- Superseded status filtering ----
        if exclude_superseded:
            df = self._filter_superseded(df)

        # ---- Reaction metadata enrichment ----
        metadata = self.get_dataset_metadata()

        if metadata is None:
            logger.warning("No reaction metadata available -- skipping metadata filter")
            if keep_metadata_columns:
                df['sf5'] = None
                df['sf6'] = None
                df['sf8'] = None
                df['sf9'] = None
                df['is_pure'] = True
                df['data_type'] = 'unknown'
            return df

        # Determine the join column
        if 'DatasetID' in df.columns:
            join_col = 'DatasetID'
        elif 'Entry' in df.columns:
            join_col = 'Entry'
        else:
            logger.warning("No DatasetID or Entry column found -- cannot join metadata")
            return df

        # Merge metadata (left join to keep all data points)
        merge_cols = ['DatasetID', 'sf5', 'sf6', 'sf8']
        if 'sf9' in metadata.columns:
            merge_cols.append('sf9')
        if diagnostics and 'FullCode' in metadata.columns:
            merge_cols.append('FullCode')

        df = df.merge(
            metadata[merge_cols].rename(columns={'DatasetID': join_col}),
            on=join_col,
            how='left',
        )

        if 'sf9' not in df.columns:
            df['sf9'] = None

        # ---- Vectorized classification (critical for 13M+ rows) ----

        # Normalise SF columns: strip whitespace, uppercase, convert empty -> None
        for col in ['sf5', 'sf6', 'sf8', 'sf9']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df.loc[df[col].isin(['', 'NONE', 'NAN', 'NULL']), col] = None

        # Classification flags
        # SF8: compound codes like "BRS/REL" must be split on "/" and each
        # token checked independently (isin() only matches exact strings).
        sf8_split = df['sf8'].fillna('').str.split('/')
        sf8_exploded = sf8_split.explode()
        df['is_relative'] = sf8_exploded.isin(SF8_EXCLUDE_ABSOLUTE).groupby(level=0).any()
        df['is_averaged'] = sf8_exploded.isin(SF8_EXCLUDE_AVERAGED).groupby(level=0).any()
        df['is_other_modified'] = sf8_exploded.isin(SF8_EXCLUDE_OTHER).groupby(level=0).any()

        df['is_ratio'] = df['sf6'].fillna('').str.contains('RAT', na=False)
        df['is_partial'] = df['sf5'].isin(SF5_EXCLUDE)
        df['is_non_xs'] = ~df['sf6'].isin(SF6_ACCEPT) & df['sf6'].notna()
        df['is_non_experimental'] = df['sf9'].isin(SF9_EXCLUDE)

        # Pure = experimental, absolute, point-wise cross section, no modifiers
        df['is_pure'] = (
            ~df['is_relative']
            & ~df['is_averaged']
            & ~df['is_ratio']
            & ~df['is_partial']
            & ~df['is_non_xs']
            & ~df['is_non_experimental']
            & ~df['is_other_modified']
        )

        # Descriptive data_type column (vectorized with np.select)
        conditions = [
            df['is_non_experimental'],
            df['is_relative'],
            df['is_ratio'],
            df['is_averaged'],
            df['is_non_xs'],
            df['is_partial'],
            df['is_pure'],
        ]
        choices = [
            'calculated', 'relative', 'ratio', 'averaged',
            'non_xs', 'partial', 'absolute',
        ]
        df['data_type'] = np.select(conditions, choices, default='other_modified')

        # Log distribution before filtering
        type_counts = df['data_type'].value_counts()
        logger.info("Reaction data type distribution:")
        for dtype, count in type_counts.items():
            pct = 100 * count / len(df)
            logger.info(f"  {dtype:20s}: {count:>10,} ({pct:5.1f}%)")

        # ---- Filter non-pure data ----
        if exclude_non_pure:
            pure_mask = df['is_pure'] == True  # noqa: E712
            non_pure_count = (~pure_mask).sum()
            if non_pure_count > 0:
                # Detailed breakdown of what was removed
                removed = df[~pure_mask]
                removed_types = removed['data_type'].value_counts()
                logger.info(
                    f"Metadata filter: removing {non_pure_count:,} non-pure data points "
                    f"({100 * non_pure_count / initial_count:.1f}%)"
                )
                for dtype, count in removed_types.items():
                    logger.info(f"  Removed {dtype}: {count:,}")

                df = df[pure_mask].copy()

        # Clean up temporary classification columns
        drop_cols = ['is_relative', 'is_ratio', 'is_averaged', 'is_partial',
                     'is_non_xs', 'is_non_experimental', 'is_other_modified']
        if not keep_metadata_columns:
            drop_cols.extend(['sf5', 'sf6', 'sf8', 'sf9', 'is_pure', 'data_type'])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        logger.info(f"Metadata filter: {len(df):,} data points retained "
                     f"({100 * len(df) / initial_count:.1f}% of original)")

        return df

    def _filter_superseded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove superseded entries using SUBENT.SPSDD."""
        superseded_ids = self._get_superseded_ids()
        if superseded_ids is None:
            logger.info("No superseded status information available -- skipping")
            return df

        if not superseded_ids:
            logger.info("No superseded entries found in database")
            return df

        # Determine join column
        if 'DatasetID' in df.columns:
            id_col = 'DatasetID'
        elif 'Entry' in df.columns:
            id_col = 'Entry'
        else:
            return df

        before = len(df)
        # DatasetID format is like "30649005S", SubentID is like "30649005"
        # Match by checking if the DatasetID starts with any superseded SubentID
        # For efficiency, extract the first 8 chars of DatasetID for matching
        ds_prefix = df[id_col].astype(str).str[:8]
        superseded_mask = ds_prefix.isin(superseded_ids)
        n_superseded = superseded_mask.sum()

        if n_superseded > 0:
            df = df[~superseded_mask].copy()
            logger.info(f"Superseded filter: removed {n_superseded:,} data points "
                         f"from {len(superseded_ids):,} superseded subentries "
                         f"({before:,} -> {len(df):,})")
        else:
            logger.info("Superseded filter: no matching superseded entries in dataset")

        return df

    def _log_reatyp_distribution(self) -> None:
        """Log x4pro_ds.reatyp distribution as a diagnostic (NOT used as a filter)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT reatyp, COUNT(*) as cnt
                FROM x4pro_ds
                WHERE reatyp IS NOT NULL
                GROUP BY reatyp
                ORDER BY cnt DESC
                LIMIT 15
            """)
            rows = cursor.fetchall()
            if rows:
                total = sum(r[1] for r in rows)
                logger.info(f"x4pro_ds.reatyp distribution (diagnostic, {total:,} datasets):")
                for reatyp, count in rows:
                    pct = 100 * count / total
                    logger.info(f"  {reatyp or 'NULL':10s}: {count:>8,} ({pct:5.1f}%)")
        except Exception:
            pass  # reatyp column may not exist; this is purely diagnostic
