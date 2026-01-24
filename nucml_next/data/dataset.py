"""
Nuclear Data Dataset with Dual-View Interface
==============================================

Provides unified access to nuclear cross-section data in both:
1. Graph format (for GNNs): PyTorch Geometric Data objects
2. Tabular format (for classical ML): Pandas DataFrames

This solves the data engineering challenge: one Parquet file, multiple views.
"""

from typing import Optional, Literal, List, Dict, Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

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
from nucml_next.data.tabular_projector import TabularProjector
from nucml_next.data.selection import DataSelection, default_selection


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
        >>> df = dataset.to_tabular(mode='naive')
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
            print("Using default DataSelection (reactor physics, neutrons, essential reactions)")
        elif selection is not None:
            # New physics-aware selection
            self.selection = selection
            self.filters = None
            print(f"Using DataSelection:\n{self.selection}")
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

        # Initialize graph builder (only for graph mode) and tabular projector
        if self.mode == 'graph':
            # Lazy import GraphBuilder (requires torch)
            from nucml_next.data.graph_builder import GraphBuilder
            self.graph_builder = GraphBuilder(self.df, self.energy_bins)
        else:
            self.graph_builder = None

        self.tabular_projector = TabularProjector(self.df, self.energy_bins)

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
        essential_columns = [
            'Entry', 'Z', 'A', 'N', 'MT', 'Energy', 'CrossSection',
            'Uncertainty', 'Mass_Excess_keV', 'Binding_Energy_keV'
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

                    # Extract partition values from fragment path
                    # Fragment partition_expression contains values like: (Z == 92) and (A == 235) and (MT == 18)
                    partition_dict = {}
                    if hasattr(fragment, 'partition_expression') and fragment.partition_expression is not None:
                        # Parse partition expression to extract Z, A, MT values
                        expr_str = str(fragment.partition_expression)
                        import re
                        for key in ['Z', 'A', 'MT']:
                            match = re.search(rf'{key} == (\d+)', expr_str)
                            if match:
                                partition_dict[key] = int(match.group(1))

                    # Add partition columns as constant arrays
                    num_rows = len(fragment_table)
                    for key, value in partition_dict.items():
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

            # 1. Projectile filtering (neutrons only - based on MT codes)
            # NOTE: Skip if already filtered at fragment level (mt_mode='all_physical' + projectile='neutron')
            # In that case, predicate pushdown already filtered to neutron MT codes
            needs_projectile_filter = (
                selection.projectile == 'neutron' and
                not (selection.mt_mode == 'all_physical')  # Already filtered at fragment level
            )
            if needs_projectile_filter:
                projectile_mt = selection.get_projectile_mt_filter()
                if projectile_mt is not None:
                    before = len(df)
                    df = df[df['MT'].isin(projectile_mt)]
                    removed = before - len(df)
                    if removed > 0:
                        print(f"  ✓ Projectile filter (neutrons): Removed {removed:,} non-neutron reactions")

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
        mode: Literal['naive', 'physics', 'tier'] = 'naive',
        reaction_types: Optional[List[int]] = None,
        tiers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Project graph data to tabular format for classical ML.

        Supports three projection strategies:
        - mode='naive': Legacy features [Z, A, E, MT] (one-hot encoded)
        - mode='physics': Graph-derived features [Z, A, E, Q, Threshold, ΔZ, ΔA]
        - mode='tier': Tier-based features using Valdez 2021 hierarchy

        Args:
            mode: Projection strategy
            reaction_types: Filter to specific MT codes (None = all reactions)
            tiers: List of feature tiers to include (e.g., ['A', 'C', 'E'])
                   Only used when mode='tier'. If None, uses selection.tiers.

        Returns:
            DataFrame ready for XGBoost/Decision Trees

        Example:
            >>> # Legacy approach (shows limitations)
            >>> df_naive = dataset.to_tabular(mode='naive')
            >>> xgb.fit(df_naive[['Z', 'A', 'Energy', 'MT_18']], df_naive['CrossSection'])
            >>>
            >>> # Physics-aware approach (better, but still not smooth)
            >>> df_physics = dataset.to_tabular(mode='physics')
            >>> xgb.fit(df_physics[['Z', 'A', 'Energy', 'Q_Value', 'Threshold']], ...)
            >>>
            >>> # Tier-based approach (Valdez 2021 hierarchy)
            >>> df_tier_c = dataset.to_tabular(mode='tier', tiers=['A', 'B', 'C'])
            >>> # Features include: Z, A, N, Energy, particle emission, radius, energetics
        """
        if mode == 'tier':
            # Use FeatureGenerator for tier-based features
            from nucml_next.data.enrichment import AME2020DataEnricher
            from nucml_next.data.features import FeatureGenerator

            # Initialize enricher if needed for Tiers B+
            if tiers is None:
                # Use tiers from DataSelection if available
                if self.selection is not None:
                    tiers = self.selection.tiers
                else:
                    tiers = ['A']  # Default to core features

            # Check if we need enrichment (Tiers B, C, D, E)
            needs_enrichment = any(tier in tiers for tier in ['B', 'C', 'D', 'E'])

            if needs_enrichment:
                # Load AME2020 data
                enricher = AME2020DataEnricher(data_dir='data/')
                enricher.load_all()
                generator = FeatureGenerator(enricher=enricher)
            else:
                # No enrichment needed for Tier A only
                generator = FeatureGenerator(enricher=None)

            # Generate tier-based features
            df = self.df.copy()
            if reaction_types is not None:
                df = df[df['MT'].isin(reaction_types)]

            return generator.generate_features(df, tiers=tiers)

        else:
            # Use legacy TabularProjector for 'naive' and 'physics' modes
            return self.tabular_projector.project(mode=mode, reaction_types=reaction_types)

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
