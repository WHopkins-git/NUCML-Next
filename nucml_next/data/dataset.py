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
    ):
        """
        Initialize NUCML dataset from EXFOR data.

        Args:
            data_path: Path to EXFOR Parquet file/directory (REQUIRED).
                       Supports both single files and partitioned datasets.
            mode: 'graph' for GNN training, 'tabular' for classical ML
            energy_bins: Energy grid for cross-section evaluation (eV)
            cache_graphs: Whether to cache constructed graphs in memory
            filters: Filters for lazy loading, e.g. {'Z': [92], 'MT': [18, 102]}
            lazy_load: Enable lazy loading for large datasets (loads on demand)

        Raises:
            ValueError: If data_path is not provided
            FileNotFoundError: If data_path does not exist
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
        self.filters = filters or {}
        self.lazy_load = lazy_load
        self._graph_cache: Dict[int, Data] = {}

        # Default energy grid: 1 eV to 20 MeV (logarithmic)
        if energy_bins is None:
            self.energy_bins = np.logspace(0, 7, 100)  # 1 eV to 10 MeV
        else:
            self.energy_bins = energy_bins

        # Load EXFOR data from Parquet (optimized for large files)
        print(f"Loading data from {self.data_path}...")
        self.df = self._load_parquet_data(self.data_path, self.filters, lazy_load)
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
        filters: Dict[str, List],
        lazy_load: bool
    ) -> pd.DataFrame:
        """
        Load data from Parquet (single file or partitioned dataset).

        OPTIMIZATIONS for large files (e.g., 4.7GB EXFOR):
        - Column pruning: Only read essential columns
        - Memory mapping: Faster I/O without full RAM allocation
        - PyArrow filters: Push down filters to read less data
        - Lazy conversion: Delay pandas conversion until necessary

        Supports:
        - Single Parquet files (.parquet)
        - Partitioned Parquet datasets (directories)
        - Efficient filtering by partition columns (Z, A, MT)
        - Lazy loading for large datasets

        Args:
            data_path: Path to Parquet file or directory
            filters: Column filters, e.g. {'Z': [92], 'A': [235]}
            lazy_load: If True, loads only metadata initially

        Returns:
            DataFrame with cross-section data
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
                dataset = ds.dataset(str(data_path), format='parquet')

                # Filter out partition columns for partitioned datasets
                partition_columns = {'Z', 'A', 'MT'}
                data_columns = [col for col in essential_columns if col not in partition_columns]

                table = dataset.to_table(
                    columns=data_columns if data_columns else None,
                    filter=self._build_dataset_filter(filters)
                )
                df = table.to_pandas().head(1000)
            else:
                # Load full dataset with optimizations
                print("  Reading partitioned dataset (this may take 5-10 minutes for large datasets)...")
                start_total = time.time()

                # Use PyArrow dataset API for partitioned data
                dataset = ds.dataset(str(data_path), format='parquet')

                # Get fragments for progress tracking
                filter_expr = self._build_dataset_filter(filters)
                fragments = list(dataset.get_fragments(filter=filter_expr))
                total_fragments = len(fragments)

                print(f"  ⏳ Reading {total_fragments} partition fragments (showing progress every 10%)...")
                start = time.time()

                # Read fragments in batches with progress updates
                tables = []
                report_interval = max(1, total_fragments // 10)  # Report every 10%

                for i, fragment in enumerate(fragments):
                    # Read fragment - PyArrow will automatically reconstruct partition columns
                    # from directory names (Z=92/A=235/MT=18/ -> adds Z, A, MT columns)
                    fragment_table = fragment.to_table(
                        columns=None,  # Read all columns (includes partition columns from path)
                        use_threads=True
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
    def _build_dataset_filter(filters: Dict[str, List]):
        """
        Build PyArrow dataset filter expression (new dataset API).

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
        import pyarrow as pa

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
        mode: Literal['naive', 'physics'] = 'naive',
        reaction_types: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Project graph data to tabular format for classical ML.

        This is the key method for the educational pathway:
        - mode='naive': Legacy features [Z, A, E, MT] (one-hot encoded)
        - mode='physics': Graph-derived features [Z, A, E, Q, Threshold, ΔZ, ΔA]

        Args:
            mode: Projection strategy
            reaction_types: Filter to specific MT codes (None = all reactions)

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
        """
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
