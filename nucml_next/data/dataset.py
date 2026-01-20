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
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data

from nucml_next.data.graph_builder import GraphBuilder
from nucml_next.data.tabular_projector import TabularProjector


class NucmlDataset(Dataset):
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
        data_path: Optional[str] = None,
        mode: Literal['graph', 'tabular'] = 'graph',
        energy_bins: Optional[np.ndarray] = None,
        cache_graphs: bool = True,
        filters: Optional[Dict[str, List]] = None,
        lazy_load: bool = False,
    ):
        """
        Initialize NUCML dataset.

        Args:
            data_path: Path to Parquet file/directory. If None, generates synthetic data.
                       Supports both single files and partitioned datasets.
            mode: 'graph' for GNN training, 'tabular' for classical ML
            energy_bins: Energy grid for cross-section evaluation (eV)
            cache_graphs: Whether to cache constructed graphs in memory
            filters: Filters for lazy loading, e.g. {'Z': [92], 'MT': [18, 102]}
            lazy_load: Enable lazy loading for large datasets (loads on demand)
        """
        super().__init__()
        self.data_path = Path(data_path) if data_path else None
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

        # Load data from Parquet or generate synthetic
        if self.data_path and self.data_path.exists():
            self.df = self._load_parquet_data(self.data_path, self.filters, lazy_load)
            print(f"✓ Loaded {len(self.df)} data points from {self.data_path}")
        else:
            print("⚠️  No data file provided. Generating synthetic nuclear data...")
            self.df = self._generate_synthetic_data()

        # Initialize graph builder and tabular projector
        self.graph_builder = GraphBuilder(self.df, self.energy_bins)
        self.tabular_projector = TabularProjector(self.df, self.energy_bins)

        # Build graph structure once
        if self.mode == 'graph':
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
        # Check if partitioned dataset (directory) or single file
        if data_path.is_dir():
            # Partitioned dataset
            dataset = pq.ParquetDataset(str(data_path), filters=self._build_filters(filters))

            if lazy_load:
                # Load only schema initially
                print(f"⚠️  Lazy loading enabled. Data will be loaded on-demand.")
                # For lazy loading, read a small subset to get schema
                df = dataset.read(columns=None).to_pandas().head(1000)
            else:
                # Load full dataset with filters
                df = dataset.read().to_pandas()

        else:
            # Single Parquet file
            table = pq.read_table(str(data_path), filters=self._build_filters(filters))
            df = table.to_pandas()

        return df

    @staticmethod
    def _build_filters(filters: Dict[str, List]) -> Optional[List]:
        """
        Build PyArrow filters from dictionary.

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

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic nuclear cross-section data for demonstration.

        Creates realistic-looking data with:
        - Multiple isotopes (U-235, U-238, Pu-239, etc.)
        - Multiple reaction types (fission, capture, elastic)
        - Resonance structures
        - Physical constraints (thresholds, unitarity)

        Returns:
            DataFrame with columns: [Z, A, N, MT, Energy, CrossSection, Q_Value, Threshold]
        """
        np.random.seed(42)

        # Define common isotopes
        isotopes = [
            (92, 235, 143, "U-235"),   # Fissile
            (92, 238, 146, "U-238"),   # Fertile
            (94, 239, 145, "Pu-239"),  # Fissile
            (94, 240, 146, "Pu-240"),  # Absorber
            (1, 1, 0, "H-1"),          # Moderator
            (8, 16, 8, "O-16"),        # Coolant
        ]

        # MT codes (ENDF-6 format)
        reactions = {
            2: ("Elastic", 0.0, 0.0),        # (n,n)
            18: ("Fission", 200e6, 0.0),     # (n,f) - releases 200 MeV
            102: ("Capture", 5e6, 0.0),      # (n,γ)
            16: ("n2n", -8e6, 8e6),          # (n,2n) - endothermic, threshold at 8 MeV
        }

        records = []

        for Z, A, N, isotope_name in isotopes:
            for mt_code, (reaction_name, q_value, threshold) in reactions.items():
                # Skip unphysical reactions
                if mt_code == 18 and Z < 90:  # Only actinides fission
                    continue
                if mt_code == 16 and A < 10:  # Skip (n,2n) for light isotopes
                    continue

                # Generate cross sections with realistic features
                for energy in self.energy_bins:
                    # Base cross section (decreasing with energy, roughly 1/v law)
                    base_xs = 10.0 / np.sqrt(energy + 1.0)

                    # Add resonance peaks (especially for capture)
                    if mt_code == 102 and 1 < energy < 1000:
                        # Add resonances at ~10, 50, 100 eV
                        resonance = 100.0 * np.exp(-((np.log10(energy) - 1.0) ** 2) / 0.5)
                        resonance += 50.0 * np.exp(-((np.log10(energy) - 1.7) ** 2) / 0.3)
                        base_xs += resonance

                    # Fission cross section (large for fissile isotopes at thermal)
                    if mt_code == 18:
                        if "235" in isotope_name or "239" in isotope_name:
                            base_xs = 500.0 / np.sqrt(energy + 1.0) + 2.0
                        else:
                            base_xs = 0.1  # Very small for non-fissile

                    # (n,2n) threshold behavior
                    if mt_code == 16:
                        if energy < threshold:
                            base_xs = 0.0
                        else:
                            # Rises sharply above threshold
                            base_xs = 2.0 * (1.0 - np.exp(-(energy - threshold) / 1e6))

                    # Apply threshold
                    if energy < threshold:
                        cross_section = 0.0
                    else:
                        cross_section = max(0.0, base_xs)

                    records.append({
                        'Z': Z,
                        'A': A,
                        'N': N,
                        'Isotope': isotope_name,
                        'MT': mt_code,
                        'Reaction': reaction_name,
                        'Energy': energy,
                        'CrossSection': cross_section,
                        'Q_Value': q_value,
                        'Threshold': threshold,
                    })

        df = pd.DataFrame(records)
        print(f"✓ Generated {len(df)} synthetic data points for {len(isotopes)} isotopes")
        return df

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
