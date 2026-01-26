"""
Nuclear Data Dataset with Dual-View Interface
==============================================

Provides unified access to nuclear cross-section data in both:
1. Graph format (for GNNs): PyTorch Geometric Data objects
2. Tabular format (for classical ML): Pandas DataFrames

This solves the data engineering challenge: one Parquet file, multiple views.
"""

from typing import Optional, Literal, List, Dict, Any
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional PyTorch imports (only required for graph mode)
try:
    from torch.utils.data import Dataset as TorchDataset
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object  # Fallback base class for tabular-only mode
    torch = None
    Data = None

# Lazy import GraphBuilder (requires torch) - only imported when needed
# from nucml_next.data.graph_builder import GraphBuilder  # Moved to lazy import in __init__
from nucml_next.data.selection import DataSelection, default_selection
from nucml_next.data.transformations import TransformationPipeline


class NucmlDataset(TorchDataset):
    """
    Main dataset class for NUCML-Next with dual-view interface.

    The dataset represents nuclear cross-section data with:
    - Isotopes (characterized by Z, A, N)
    - Reactions (MT codes, Q-values, thresholds)
    - Energy-dependent cross sections σ(E)

    Attributes:
        data_path: Path to Parquet file containing nuclear data
        mode: 'graph' or 'tabular' view mode
        graph_builder: Constructs PyG graph objects
        tabular_projector: Projects to DataFrame format

    Example:
        >>> # For GNN training
        >>> dataset = NucmlDataset('data.parquet', mode='graph')
        >>> graph = dataset[0]  # Returns PyG Data object
        >>>
        >>> # For XGBoost training
        >>> dataset = NucmlDataset('data.parquet', mode='tabular')
        >>> df = dataset.to_tabular(tiers=['A', 'C'])
    """

    def __init__(
        self,
        data_path: str,
        mode: Literal['graph', 'tabular'] = 'graph',
        energy_bins: Optional[np.ndarray] = None,
        cache_graphs: bool = True,
        filters: Optional[Dict[str, List]] = None,
        lazy_load: bool = False,
        selection: Optional[DataSelection] = None,
    ):
        """
        Initialize NUCML dataset from EXFOR data.

        Args:
            data_path: Path to EXFOR Parquet file/directory (REQUIRED).
                       Supports both single files and partitioned datasets.
            mode: 'graph' for GNN training, 'tabular' for classical ML
            energy_bins: Energy grid for cross-section evaluation (eV)
            cache_graphs: Whether to cache constructed graphs in memory
            filters: (DEPRECATED) Legacy dict filters, e.g. {'Z': [92], 'MT': [18, 102]}.
                     Use 'selection' parameter instead for physics-aware filtering.
            lazy_load: Enable lazy loading for large datasets (loads on demand)
            selection: DataSelection object for physics-aware data filtering.
                       If None, uses default reactor physics selection.

        Raises:
            ValueError: If data_path is not provided
            FileNotFoundError: If data_path does not exist

        Example:
            >>> # Default: reactor physics, neutrons, essential reactions
            >>> dataset = NucmlDataset(
            ...     data_path='data/exfor_processed.parquet',
            ...     mode='tabular'
            ... )

            >>> # Custom: threshold reactions only
            >>> from nucml_next.data.selection import DataSelection
            >>> sel = DataSelection(mt_mode='threshold_only')
            >>> dataset = NucmlDataset(
            ...     data_path='data/exfor_processed.parquet',
            ...     mode='tabular',
            ...     selection=sel
            ... )
        """
        super().__init__()

        # Require data_path
        if not data_path:
            raise ValueError(
                "❌ ERROR: data_path is required!\n"
                "   NUCML-Next requires real EXFOR experimental data.\n"
                "   \n"
                "   To obtain EXFOR data:\n"
                "   1. Download from: https://www-nds.iaea.org/exfor/\n"
                "   2. Run: python scripts/ingest_exfor.py --exfor-root /path/to/EXFOR-X5json/\n"
                "   3. Load: NucmlDataset(data_path='data/exfor_processed.parquet')"
            )

        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"❌ ERROR: EXFOR data not found at {self.data_path}\n"
                f"   Please run EXFOR ingestor first:\n"
                f"   python scripts/ingest_exfor.py --exfor-root /path/to/EXFOR-X5json/ --output {self.data_path}"
            )

        # Check PyTorch availability for graph mode
        if mode == 'graph' and not TORCH_AVAILABLE:
            raise ImportError(
                "❌ ERROR: PyTorch is required for graph mode!\n"
                "   Install with: pip install torch torch-geometric\n"
                "   Or use mode='tabular' for classical ML without PyTorch"
            )

        self.mode = mode
        self.cache_graphs = cache_graphs
        self.lazy_load = lazy_load
        self._graph_cache: Dict[int, Data] = {}

        # Handle selection vs legacy filters
        if selection is not None and filters is not None:
            raise ValueError("Cannot specify both 'selection' and 'filters'. Use 'selection' for new code.")

        if selection is None and filters is None:
            # Default: use reactor physics selection
            self.selection = default_selection()
            self.filters = None  # Legacy filters not used
        elif selection is not None:
            # New physics-aware selection
            self.selection = selection
            self.filters = None
        else:
            # Legacy filters (still supported for evaluation/simple selections)
            self.selection = None
            self.filters = filters
            # Only show informational note if this looks like a training use case
            # (no specific isotope filtering suggests broad usage)
            if 'Z' not in filters and 'A' not in filters:
                print("💡 Note: Consider using DataSelection for physics-aware filtering with predicate pushdown.")
                print("   Legacy 'filters' still work but don't provide performance optimizations.")

        # Default energy grid: 1 eV to 20 MeV (logarithmic)
        if energy_bins is None:
            self.energy_bins = np.logspace(0, 7, 100)  # 1 eV to 10 MeV
        else:
            self.energy_bins = energy_bins

        # Load EXFOR data from Parquet (optimized for large files)
        print(f"\nLoading data from {self.data_path}...")
        self.df = self._load_parquet_data(self.data_path, self.selection, self.filters, lazy_load)
        print(f"✓ Loaded {len(self.df):,} EXFOR data points from {self.data_path}")

        # Apply on-demand AME enrichment if needed (based on requested tiers)
        if self.selection and self.selection.tiers:
            self._enrich_with_ame_if_needed(self.selection.tiers)

        # Initialize graph builder (only for graph mode) and tabular projector
        if self.mode == 'graph':
            # Lazy import GraphBuilder (requires torch)
            from nucml_next.data.graph_builder import GraphBuilder
            self.graph_builder = GraphBuilder(self.df, self.energy_bins)
        else:
            self.graph_builder = None

        # Lazy graph building - only build when accessed (saves initialization time)
        self.graph_data = None
        if self.mode == 'graph' and not lazy_load:
            # Build graph eagerly only if explicitly requested
            print("Building global graph structure...")
            self.graph_data = self.graph_builder.build_global_graph()
            print(f"✓ Graph built: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")

    def _load_parquet_data(
        self,
        data_path: Path,
        selection: Optional[DataSelection],
        legacy_filters: Optional[Dict[str, List]],
        lazy_load: bool
    ) -> pd.DataFrame:
        """
        Load data from Parquet with physics-aware predicate pushdown.

        OPTIMIZATIONS for large files (e.g., 4.7GB EXFOR):
        - Predicate pushdown: Filter at PyArrow fragment level (reduces I/O by ~90%)
        - Column pruning: Only read essential columns (reduces memory by 50%+)
        - Memory mapping: Faster I/O without full RAM allocation
        - Fragment-based reading: Progress tracking for large datasets

        Supports:
        - Single Parquet files (.parquet)
        - Partitioned Parquet datasets (directories with Z=/A=/MT= structure)
        - Physics-aware filtering (projectile, energy range, MT modes)
        - Holdout isotopes for true extrapolation testing

        Args:
            data_path: Path to Parquet file or directory
            selection: DataSelection object for physics-aware filtering
            legacy_filters: (Deprecated) Legacy dict filters
            lazy_load: If True, loads only metadata initially

        Returns:
            DataFrame with cross-section data (filtered and validated)
        """
        # Essential columns needed for NUCML-Next (column pruning optimization)
        # This can reduce read time by 50%+ for wide tables
        # NOTE: For partitioned data, Z, A, MT come from Hive partition metadata
        # NOTE: For single-file data, Z, A, MT must be in the column list
        # NOTE: AME columns (Mass_Excess_keV, etc.) are added later via on-demand enrichment
        essential_columns = [
            'Entry', 'Z', 'A', 'N', 'MT', 'Energy', 'CrossSection', 'Uncertainty'
        ]

        # Check if partitioned dataset (directory) or single file
        if data_path.is_dir():
            # Partitioned dataset - use PyArrow dataset API
            import pyarrow.dataset as ds
            import time

            if lazy_load:
                # Load only schema initially
                print(f"⚠️  Lazy loading enabled. Data will be loaded on-demand.")
                # For lazy loading, read a small subset to get schema

                # Explicitly specify Hive partitioning so PyArrow knows about Z, A, MT columns
                from pyarrow.dataset import HivePartitioning
                partitioning = HivePartitioning(
                    pa.schema([
                        ('Z', pa.int64()),
                        ('A', pa.int64()),
                        ('MT', pa.int64())
                    ])
                )
                dataset = ds.dataset(
                    str(data_path),
                    format='parquet',
                    partitioning=partitioning
                )

                # Build filter expression (use selection if available, else legacy filters)
                filter_expr = self._build_selection_filter(selection) if selection else self._build_dataset_filter(legacy_filters)

                table = dataset.to_table(
                    columns=None,  # Read all columns
                    filter=filter_expr
                )
                df = table.to_pandas().head(1000)
            else:
                # Load full dataset with optimizations
                print("  Reading partitioned dataset (this may take 5-10 minutes for large datasets)...")
                start_total = time.time()

                # Use PyArrow dataset API for partitioned data
                # Explicitly specify Hive partitioning so PyArrow knows about Z, A, MT columns
                from pyarrow.dataset import HivePartitioning
                partitioning = HivePartitioning(
                    pa.schema([
                        ('Z', pa.int64()),
                        ('A', pa.int64()),
                        ('MT', pa.int64())
                    ])
                )
                dataset = ds.dataset(
                    str(data_path),
                    format='parquet',
                    partitioning=partitioning
                )

                # Get fragments for progress tracking with predicate pushdown
                # This is critical - filtering at fragment level reduces I/O by ~90%
                filter_expr = self._build_selection_filter(selection) if selection else self._build_dataset_filter(legacy_filters)

                print(f"  Applying filter at fragment level (predicate pushdown)...")
                fragments = list(dataset.get_fragments(filter=filter_expr))
                total_fragments = len(fragments)

                if total_fragments == 0:
                    print("  ⚠️  Warning: No fragments matched filter criteria. Returning empty DataFrame.")
                    return pd.DataFrame()

                print(f"  ⏳ Reading {total_fragments} partition fragments (showing progress every 10%)...")
                start = time.time()

                # Read fragments in batches with progress updates
                tables = []
                report_interval = max(1, total_fragments // 10)  # Report every 10%

                for i, fragment in enumerate(fragments):
                    # Read fragment data (without partition columns initially)
                    fragment_table = fragment.to_table(
                        columns=None,  # Read all data columns
                        use_threads=True
                    )

                    # Extract partition values from fragment path if not already in data
                    # Fragment partition_expression contains values like: (Z == 92) and (A == 235) and (MT == 18)
                    # Note: Newer Parquet files store Z, A, MT as dictionary columns in the data itself
                    partition_dict = {}
                    if hasattr(fragment, 'partition_expression') and fragment.partition_expression is not None:
                        # Parse partition expression to extract Z, A, MT values
                        expr_str = str(fragment.partition_expression)
                        import re
                        for key in ['Z', 'A', 'MT']:
                            # Only add from partition if not already in data
                            if key not in fragment_table.column_names:
                                match = re.search(rf'{key} == (\d+)', expr_str)
                                if match:
                                    partition_dict[key] = int(match.group(1))

                    # Add partition columns as constant arrays (only if missing from data)
                    num_rows = len(fragment_table)
                    for key, value in partition_dict.items():
                        if key not in fragment_table.column_names:
                            fragment_table = fragment_table.append_column(
                                key,
                                pa.array([value] * num_rows, type=pa.int64())
                            )

                    # Apply column pruning after reading (if needed)
                    if essential_columns:
                        available_columns = [col for col in essential_columns if col in fragment_table.column_names]
                        fragment_table = fragment_table.select(available_columns)

                    tables.append(fragment_table)

                    # Show progress every 10%
                    if (i + 1) % report_interval == 0 or (i + 1) == total_fragments:
                        percent = int(((i + 1) / total_fragments) * 100)
                        elapsed = time.time() - start
                        rate = (i + 1) / elapsed
                        eta = (total_fragments - (i + 1)) / rate if rate > 0 else 0
                        print(f"    Progress: {percent:3d}% ({i+1}/{total_fragments} fragments, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

                # Concatenate all tables
                print("  ⏳ Concatenating fragments...")
                import pyarrow.compute as pc
                table = pa.concat_tables(tables)
                read_time = time.time() - start
                print(f"  ✓ Read complete: {read_time:.1f}s, {table.nbytes / 1e9:.2f} GB")

                # Convert to pandas (this is often the slowest part)
                print("  ⏳ Converting to Pandas...")
                start = time.time()
                df = table.to_pandas()
                convert_time = time.time() - start
                print(f"  ✓ Conversion complete: {convert_time:.1f}s")

                total_time = time.time() - start_total
                print(f"  ✓ Total load time: {total_time:.1f}s")

        else:
            # Single Parquet file
            import time

            print("  Reading Parquet file...")
            start_total = time.time()

            # Read table
            print("  ⏳ Reading Parquet file...")
            start = time.time()
            table = pq.read_table(
                str(data_path),
                columns=essential_columns,  # Column pruning
                filters=self._build_filters(filters),  # Filter pushdown
                memory_map=True,  # Memory-mapped I/O (faster, less RAM)
                use_threads=True  # Parallel read
            )
            read_time = time.time() - start
            print(f"  ✓ Read complete: {read_time:.1f}s, {table.nbytes / 1e9:.2f} GB")

            # Convert to pandas
            print("  ⏳ Converting to Pandas...")
            start = time.time()
            df = table.to_pandas()
            convert_time = time.time() - start
            print(f"  ✓ Conversion complete: {convert_time:.1f}s")

            total_time = time.time() - start_total
            print(f"  ✓ Total load time: {total_time:.1f}s")

        # Post-load filtering (applied to DataFrame after reading)
        if selection is not None:
            initial_rows = len(df)
            print(f"\nApplying post-load filters...")

            # 1. Projectile filtering (neutrons only)
            # Prefer explicit Projectile column if available, otherwise use MT codes
            # NOTE: Skip if already filtered at fragment level (mt_mode='all_physical' + projectile='neutron')
            # In that case, predicate pushdown already filtered to neutron MT codes
            needs_projectile_filter = (
                selection.projectile == 'neutron' and
                not (selection.mt_mode == 'all_physical')  # Already filtered at fragment level
            )
            if needs_projectile_filter:
                before = len(df)

                # Method 1: Use explicit Projectile column if available (most accurate)
                if 'Projectile' in df.columns and df['Projectile'].notna().any():
                    df = df[df['Projectile'] == 'n']  # 'n' = neutron in EXFOR
                    removed = before - len(df)
                    if removed > 0:
                        print(f"  ✓ Projectile filter (neutrons): Removed {removed:,} non-neutron reactions (using explicit Projectile column)")

                # Method 2: Fall back to MT code filtering (for legacy data without Projectile)
                else:
                    projectile_mt = selection.get_projectile_mt_filter()
                    if projectile_mt is not None:
                        df = df[df['MT'].isin(projectile_mt)]
                        removed = before - len(df)
                        if removed > 0:
                            print(f"  ✓ Projectile filter (neutrons): Removed {removed:,} non-neutron reactions (inferred from MT codes)")

            # 2. Holdout isotopes (exclude specific Z/A pairs for evaluation)
            if selection.holdout_isotopes:
                before = len(df)
                for z, a in selection.holdout_isotopes:
                    df = df[~((df['Z'] == z) & (df['A'] == a))]
                removed = before - len(df)
                if removed > 0:
                    isotopes_str = ', '.join([f"({z},{a})" for z, a in selection.holdout_isotopes])
                    print(f"  ✓ Holdout filter: Removed {removed:,} measurements from {isotopes_str}")

            # 3. Data validity (drop NaN or non-positive cross-sections)
            if selection.drop_invalid:
                before = len(df)

                # Drop NaN cross-sections
                df = df.dropna(subset=['CrossSection'])

                # Drop non-positive cross-sections (required for log-transform)
                df = df[df['CrossSection'] > 0]

                removed = before - len(df)
                if removed > 0:
                    print(f"  ✓ Validity filter: Removed {removed:,} invalid measurements (NaN or ≤0)")

            final_rows = len(df)
            if initial_rows != final_rows:
                print(f"  Summary: {initial_rows:,} → {final_rows:,} ({100 * final_rows / initial_rows:.1f}% retained)\n")

        return df

    def _enrich_with_ame_if_needed(self, tiers: List[str]) -> None:
        """
        Apply on-demand AME2020/NUBASE2020 enrichment if requested tiers require it.

        Architecture: Lean Parquet + On-Demand Enrichment
        --------------------------------------------------
        Instead of pre-enriching during ingestion (which duplicates AME data for every
        EXFOR measurement), we:
        1. Load lean Parquet (EXFOR data only)
        2. Check if requested tiers need AME data (B, C, D, or E)
        3. Load AME files once from data/ directory
        4. Merge AME data into self.df on (Z, A)

        Benefits:
        - Lean Parquet files (~10x smaller)
        - Faster ingestion
        - AME loaded once per dataset instance (not duplicated in file)
        - Same enrichment capabilities

        Args:
            tiers: List of requested tiers (e.g., ['A', 'C', 'D'])
        """
        # Tier A only needs basic features (Z, A, N, Energy, MT) - no AME needed
        needs_ame = any(tier in ['B', 'C', 'D', 'E'] for tier in tiers)

        if not needs_ame:
            return

        # Check if already enriched (avoid double enrichment)
        ame_indicator_columns = ['Mass_Excess_keV', 'Binding_Energy_keV', 'Spin', 'Q_alpha_keV']
        already_enriched = any(col in self.df.columns for col in ame_indicator_columns)

        if already_enriched:
            print("  ℹ️  AME enrichment columns already present in Parquet")
            return

        # Load AME enricher from data/ directory
        print(f"\n  Loading AME2020/NUBASE2020 data for tiers {tiers}...")

        # Try to find AME files in common locations
        from pathlib import Path
        ame_search_paths = [
            Path('data'),           # Current working directory/data
            Path('../data'),        # Parent directory/data (notebooks)
            Path(__file__).parent.parent.parent / 'data',  # Repository root/data
        ]

        ame_dir = None
        for search_path in ame_search_paths:
            if search_path.exists() and (search_path / 'mass_1.mas20.txt').exists():
                ame_dir = str(search_path)
                break

        if ame_dir is None:
            print("\n  ⚠️  WARNING: AME2020 files not found in common locations:")
            for path in ame_search_paths:
                print(f"      - {path.absolute()}")
            print("\n  Continuing without AME enrichment. To enable:")
            print("      1. Download: wget https://www-nds.iaea.org/amdc/ame2020/*.mas20.txt")
            print("      2. Place *.mas20.txt files in data/ directory")
            print("\n  Required files: mass_1.mas20.txt, rct1.mas20.txt, rct2_1.mas20.txt")
            print("                  nubase_4.mas20.txt, covariance.mas20.txt\n")
            return

        # Load AME enricher
        try:
            from nucml_next.data.enrichment import AME2020DataEnricher

            print(f"  Found AME files in: {ame_dir}")
            enricher = AME2020DataEnricher(data_dir=ame_dir)

            # Load all AME files into memory (they're tiny ~MBs vs GB for EXFOR)
            # This loads all available files and merges them into enrichment_table
            print(f"  Loading all AME2020/NUBASE2020 files into memory...")
            enrichment_table = enricher.load_all()

            if enrichment_table is None or len(enrichment_table) == 0:
                print("  ⚠️  AME enrichment table is empty - skipping enrichment")
                return

            # Merge AME data into self.df (left join on Z, A)
            print(f"  Merging AME data with EXFOR measurements...")

            # CRITICAL: Filter to ground states only (Isomer_Level == 0) to avoid duplicates
            # The enrichment table includes both ground states and isomeric states for the same (Z, A).
            # Merging without filtering would duplicate EXFOR measurements.
            if 'Isomer_Level' in enrichment_table.columns:
                enrichment_table = enrichment_table[enrichment_table['Isomer_Level'] == 0].copy()
                print(f"    Filtered to {len(enrichment_table)} ground-state isotopes (Isomer_Level == 0)")

            enrich_cols = [col for col in enrichment_table.columns if col not in ['N']]
            enrichment_data = enrichment_table[enrich_cols].copy()

            initial_cols = len(self.df.columns)
            initial_rows = len(self.df)
            self.df = self.df.merge(enrichment_data, on=['Z', 'A'], how='left')
            added_cols = len(self.df.columns) - initial_cols
            final_rows = len(self.df)

            # Verify no row duplication occurred
            if final_rows != initial_rows:
                raise ValueError(
                    f"❌ ERROR: AME enrichment caused row duplication!\n"
                    f"   Before merge: {initial_rows:,} rows\n"
                    f"   After merge: {final_rows:,} rows\n"
                    f"   This indicates duplicate (Z, A) entries in enrichment table.\n"
                    f"   Please report this issue."
                )

            # Report coverage
            n_enriched = self.df['Mass_Excess_keV'].notna().sum() if 'Mass_Excess_keV' in self.df.columns else 0
            coverage = 100 * n_enriched / len(self.df) if len(self.df) > 0 else 0

            print(f"  ✓ Added {added_cols} AME enrichment columns")
            print(f"  ✓ Coverage: {n_enriched:,} / {len(self.df):,} ({coverage:.1f}%) measurements enriched\n")

        except Exception as e:
            print(f"\n  ⚠️  WARNING: Failed to load AME enrichment: {e}")
            print("      Continuing without AME enrichment\n")
            return

    @staticmethod
    def _build_filters(filters: Dict[str, List]) -> Optional[List]:
        """
        Build PyArrow filters from dictionary (legacy format for read_table).

        Args:
            filters: Dictionary of column -> values, e.g. {'Z': [92, 94]}

        Returns:
            PyArrow filter expression

        Example:
            {'Z': [92], 'MT': [18]} → [('Z', 'in', [92]), ('MT', 'in', [18])]
        """
        if not filters:
            return None

        filter_list = []
        for col, values in filters.items():
            if isinstance(values, list) and len(values) > 0:
                filter_list.append((col, 'in', values))

        return filter_list if filter_list else None

    @staticmethod
    def _build_selection_filter(selection: DataSelection):
        """
        Build PyArrow filter expression from DataSelection (predicate pushdown).

        This enables efficient filtering at the fragment level, reducing I/O by ~90%
        for selective queries (e.g., neutrons only, specific energy range).

        Args:
            selection: DataSelection object with filter criteria

        Returns:
            PyArrow compute expression or None

        Example:
            DataSelection(projectile='neutron', energy_min=1e3, energy_max=1e6, mt_mode='reactor_core')
            → (Energy >= 1000) & (Energy <= 1e6) & (MT.isin([2,4,16,18,102,103,107]))
        """
        import pyarrow.compute as pc

        filter_expr = None

        # Energy range filter (critical for predicate pushdown)
        if selection.energy_min is not None:
            energy_min_filter = pc.field('Energy') >= selection.energy_min
            filter_expr = energy_min_filter if filter_expr is None else filter_expr & energy_min_filter

        if selection.energy_max is not None:
            energy_max_filter = pc.field('Energy') <= selection.energy_max
            filter_expr = energy_max_filter if filter_expr is None else filter_expr & energy_max_filter

        # MT code filter (reaction type selection)
        # CRITICAL: When projectile='neutron' and mt_mode='all_physical',
        # get_mt_codes() returns neutron MT codes for predicate pushdown
        mt_codes = selection.get_mt_codes()
        if mt_codes is not None and len(mt_codes) > 0:
            mt_filter = pc.field('MT').isin(mt_codes)
            filter_expr = mt_filter if filter_expr is None else filter_expr & mt_filter

        # Projectile filter: Handled by MT code filtering when possible
        # For mt_mode='all_physical' + projectile='neutron': filtered at fragment level
        # For other modes: filtered post-load (see post-load filtering section)

        # Holdout isotopes filter (exclude specific Z/A pairs)
        # Done post-load for clarity and flexibility

        return filter_expr

    @staticmethod
    def _build_dataset_filter(filters: Dict[str, List]):
        """
        Build PyArrow dataset filter expression from legacy dict filters.

        DEPRECATED: Use DataSelection with _build_selection_filter() instead.

        Args:
            filters: Dictionary of column -> values, e.g. {'Z': [92, 94]}

        Returns:
            PyArrow compute expression or None

        Example:
            {'Z': [92], 'MT': [18]} → (field('Z').isin([92])) & (field('MT').isin([18]))
        """
        if not filters:
            return None

        import pyarrow.compute as pc

        filter_expr = None
        for col, values in filters.items():
            if isinstance(values, list) and len(values) > 0:
                # Create isin expression for this column
                col_filter = pc.field(col).isin(values)

                # Combine with existing filters using AND
                if filter_expr is None:
                    filter_expr = col_filter
                else:
                    filter_expr = filter_expr & col_filter

        return filter_expr

    def get_global_graph(self) -> Data:
        """
        Get the global graph structure (lazy building).

        Builds the graph on first access if not already built.
        This allows faster initialization for large datasets.

        Returns:
            PyG Data object with complete nuclear topology
        """
        if self.mode != 'graph':
            raise ValueError("get_global_graph() is only available in graph mode")

        if self.graph_data is None:
            print("Building global graph structure (first access)...")
            self.graph_data = self.graph_builder.build_global_graph()
            print(f"✓ Graph built: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
        return self.graph_data

    def __len__(self) -> int:
        """Return number of samples (isotope-energy combinations)."""
        if self.mode == 'graph':
            # For graph mode, we return energy-specific subgraphs
            return len(self.energy_bins)
        else:
            return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            For graph mode: PyG Data object with nuclide graph at specific energy
            For tabular mode: Not typically used (use to_tabular() instead)
        """
        if self.mode == 'graph':
            # Check cache
            if self.cache_graphs and idx in self._graph_cache:
                return self._graph_cache[idx]

            # Build energy-specific subgraph
            energy = self.energy_bins[idx]
            graph = self.graph_builder.build_energy_graph(energy)

            if self.cache_graphs:
                self._graph_cache[idx] = graph

            return graph
        else:
            # For tabular mode, just return the row (not typically used)
            row = self.df.iloc[idx]
            return row.to_dict()

    def to_tabular(
        self,
        tiers: Optional[List[str]] = None,
        reaction_types: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Project graph data to tabular format for classical ML.

        Uses tier-based feature engineering following Valdez 2021 hierarchy.
        MT codes are transformed into particle emission vectors (out_n, out_p, etc.)
        rather than one-hot encoding.

        Args:
            tiers: List of feature tiers to include (e.g., ['A', 'C', 'E'])
                   If None, uses selection.tiers or defaults to ['A'].
            reaction_types: Filter to specific MT codes (None = all reactions)

        Returns:
            DataFrame with tier-based features ready for ML training

        Example:
            >>> # Use default tiers from DataSelection
            >>> df = dataset.to_tabular()
            >>>
            >>> # Specify custom tiers
            >>> df = dataset.to_tabular(tiers=['A', 'B', 'C'])
            >>> # Features: Z, A, N, Energy + particle vector + geometry + energetics
            >>>
            >>> # Filter to specific reactions
            >>> df_fission = dataset.to_tabular(tiers=['A', 'C'], reaction_types=[18])
        """
        from nucml_next.data.features import FeatureGenerator

        # Determine which tiers to use
        if tiers is None:
            # Use tiers from DataSelection if available
            if self.selection is not None:
                tiers = self.selection.tiers
            else:
                tiers = ['A']  # Default to core features

        # Note: No enricher needed - self.df already has AME columns from NucmlDataset loading
        # FeatureGenerator will compute derived features from existing columns
        generator = FeatureGenerator(enricher=None)

        # Generate tier-based features
        df = self.df.copy()
        if reaction_types is not None:
            df = df[df['MT'].isin(reaction_types)]

        return generator.generate_features(df, tiers=tiers)

    def get_isotope_graph(self, Z: int, A: int) -> Data:
        """
        Get the subgraph for a specific isotope.

        Args:
            Z: Atomic number
            A: Mass number

        Returns:
            PyG Data object for this isotope and its neighbors
        """
        if self.mode != 'graph':
            raise ValueError("get_isotope_graph() is only available in graph mode")

        return self.graph_builder.build_isotope_subgraph(Z, A)

    def save_to_parquet(self, output_path: str) -> None:
        """
        Save dataset to Parquet format.

        Args:
            output_path: Path to save Parquet file
        """
        table = pa.Table.from_pandas(self.df)
        pq.write_table(table, output_path)
        print(f"✓ Saved dataset to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics for reporting.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_isotopes': self.df[['Z', 'A']].drop_duplicates().shape[0],
            'num_reactions': self.df['MT'].nunique(),
            'num_energies': len(self.energy_bins),
            'num_data_points': len(self.df),
            'energy_range': (self.energy_bins.min(), self.energy_bins.max()),
            'cross_section_range': (self.df['CrossSection'].min(), self.df['CrossSection'].max()),
        }

        if self.mode == 'graph':
            stats.update({
                'num_nodes': self.graph_data.num_nodes,
                'num_edges': self.graph_data.num_edges,
            })

        return stats

    def get_transformation_pipeline(
        self,
        feature_columns: Optional[List[str]] = None,
        auto_detect_tiers: bool = True
    ) -> TransformationPipeline:
        """
        Create and fit a transformation pipeline for ML training.

        Implements reversible transformations:
        1. Log-scaling for cross-sections: σ' = log₁₀(σ + 10⁻¹⁰)
        2. Log-scaling for energies: E' = log₁₀(E)
        3. StandardScaler for features: X' = (X - μ) / σ

        Args:
            feature_columns: List of columns to standardize.
                           If None and auto_detect_tiers=True, automatically
                           detects tier-based features from self.df columns.
            auto_detect_tiers: Automatically detect standardizable features
                             based on tier column names (Z, A, N, R_fm, etc.)

        Returns:
            Fitted TransformationPipeline ready for transform()

        Example:
            >>> # Get tier-based features (particle emission vectors by default)
            >>> df = dataset.to_tabular(tiers=['A', 'B', 'C'])
            >>>
            >>> # Create and fit transformation pipeline
            >>> pipeline = dataset.get_transformation_pipeline()
            >>>
            >>> # Transform features and target
            >>> X = df.drop(columns=['CrossSection', 'Uncertainty', 'Entry', 'MT'])
            >>> y = df['CrossSection']
            >>> energy = df['Energy']
            >>>
            >>> X_t, y_t = pipeline.fit_transform(X, y, energy)
            >>>
            >>> # Train model on transformed data
            >>> model.fit(X_t, y_t)
            >>>
            >>> # Make predictions and inverse transform
            >>> y_pred_log = model.predict(X_test_t)
            >>> y_pred = pipeline.inverse_transform_target(pd.Series(y_pred_log))
        """
        if feature_columns is None and auto_detect_tiers:
            # Auto-detect standardizable features based on tier system
            # These are continuous numerical features that benefit from standardization
            tier_features = [
                # Tier A (Core) - standardize Z, A, N (not particle vector or Energy)
                'Z', 'A', 'N',
                # Tier B (Geometric)
                'R_fm', 'kR',
                # Tier C (Energetics)
                'Mass_Excess_MeV', 'Binding_Energy_MeV', 'Binding_Per_Nucleon_MeV',
                'S_1n_MeV', 'S_2n_MeV', 'S_1p_MeV', 'S_2p_MeV',
                # Tier D (Topological) - only continuous features
                'Spin', 'Valence_N', 'Valence_P', 'P_Factor',
                'Shell_Closure_N', 'Shell_Closure_P',
                # Tier E (Q-values)
                'Q_alpha_MeV', 'Q_2beta_minus_MeV', 'Q_ep_MeV', 'Q_beta_n_MeV',
                'Q_4beta_minus_MeV', 'Q_d_alpha_MeV', 'Q_p_alpha_MeV', 'Q_n_alpha_MeV'
            ]

            # Filter to columns actually present in self.df
            feature_columns = [col for col in tier_features if col in self.df.columns]

            logger.info(f"Auto-detected {len(feature_columns)} standardizable features")
            logger.info(f"  Features: {feature_columns[:5]}...")

        # Create pipeline with config from DataSelection
        pipeline = TransformationPipeline(config=self.selection.transformation_config)

        # Fit on current dataset
        # Note: User should split train/test BEFORE fitting to avoid data leakage
        logger.info("Fitting transformation pipeline on dataset...")
        logger.info(f"  Using transformation config: scaler={self.selection.transformation_config.scaler_type}, "
                   f"log_target={self.selection.transformation_config.log_target}, "
                   f"log_energy={self.selection.transformation_config.log_energy}")
        logger.warning(
            "⚠️  Pipeline fitted on entire dataset. "
            "For proper ML workflow, fit only on training split!"
        )

        pipeline.fit(
            self.df,
            y=self.df['CrossSection'] if 'CrossSection' in self.df.columns else None,
            energy=self.df['Energy'] if 'Energy' in self.df.columns else None,
            feature_columns=feature_columns
        )

        return pipeline

    def to_tabular_transformed(
        self,
        tiers: Optional[List[str]] = None,
        fit_pipeline: bool = True,
        pipeline: Optional[TransformationPipeline] = None,
        return_pipeline: bool = False
    ) -> pd.DataFrame:
        """
        Get transformed tabular data ready for ML training.

        Combines to_tabular() with automatic transformation pipeline:
        1. Generate tier-based features (particle emission vectors by default)
        2. Apply log-scaling to Energy and CrossSection
        3. Apply StandardScaler to features

        Args:
            tiers: Feature tiers to include (e.g., ['A', 'B', 'C'])
            fit_pipeline: If True, fit new pipeline on this data
                        If False, must provide pre-fitted pipeline
            pipeline: Pre-fitted TransformationPipeline (optional)
                     If None and fit_pipeline=True, creates new pipeline
            return_pipeline: If True, returns (df_transformed, pipeline)
                           If False, returns only df_transformed

        Returns:
            Transformed DataFrame ready for model.fit()
            If return_pipeline=True: Tuple of (df_transformed, pipeline)

        Warning:
            For proper ML workflow, split your data BEFORE calling this method,
            then fit pipeline only on training data:

            >>> # Split data first
            >>> train_dataset = NucmlDataset(..., selection=train_selection)
            >>> test_dataset = NucmlDataset(..., selection=test_selection)
            >>>
            >>> # Fit pipeline on training data
            >>> train_df, pipeline = train_dataset.to_tabular_transformed(
            ...     tiers=['A', 'C'], return_pipeline=True
            ... )
            >>>
            >>> # Transform test data with same pipeline (no fitting!)
            >>> test_df = test_dataset.to_tabular_transformed(
            ...     tiers=['A', 'C'], fit_pipeline=False,
            ...     pipeline=pipeline
            ... )

        Example:
            >>> # Quick start (single dataset, proper split needed)
            >>> df_transformed = dataset.to_tabular_transformed(
            ...     tiers=['A', 'B', 'C']
            ... )
            >>> X = df_transformed.drop(columns=['CrossSection_log', 'Entry', 'MT'])
            >>> y = df_transformed['CrossSection_log']
            >>> model.fit(X, y)
        """
        # Get tabular features (uses particle emission vectors by default)
        df = self.to_tabular(tiers=tiers)

        # Create or use existing pipeline
        if pipeline is None:
            if not fit_pipeline:
                raise ValueError(
                    "Must provide pipeline if fit_pipeline=False. "
                    "Either set fit_pipeline=True or pass a pre-fitted pipeline."
                )

            # Create and fit new pipeline
            pipeline = self.get_transformation_pipeline()

        # Separate features and target
        feature_cols = [col for col in df.columns
                       if col not in ['CrossSection', 'Uncertainty', 'Entry', 'MT']]
        X = df[feature_cols]
        y = df['CrossSection'] if 'CrossSection' in df.columns else None
        energy = df['Energy'] if 'Energy' in df.columns else None

        # Apply transformations
        if fit_pipeline:
            X_transformed, y_transformed = pipeline.fit_transform(X, y, energy)
        else:
            X_transformed = pipeline.transform(X, energy)
            y_transformed = pipeline.transform_target(y) if y is not None else None

        # Combine into single DataFrame
        df_transformed = X_transformed.copy()
        if y_transformed is not None:
            df_transformed['CrossSection_log'] = y_transformed

        # Keep metadata columns
        if 'Entry' in df.columns:
            df_transformed['Entry'] = df['Entry']
        if 'MT' in df.columns:
            df_transformed['MT'] = df['MT']
        if 'Uncertainty' in df.columns:
            df_transformed['Uncertainty'] = df['Uncertainty']

        if return_pipeline:
            return df_transformed, pipeline
        else:
            return df_transformed
