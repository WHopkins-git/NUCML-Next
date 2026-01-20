"""
Graph Builder for Nuclear Data
===============================

Constructs graph representations of nuclear data where:
- Nodes: Isotopes (characterized by Z, A, N)
- Edges: Nuclear reactions/decays connecting isotopes
- Node features: Nuclear properties (binding energy, spin, etc.)
- Edge features: Reaction properties (Q-value, threshold, cross section)

This captures the topology of the Chart of Nuclides.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class GraphBuilder:
    """
    Builds PyTorch Geometric graph representations of nuclear data.

    The graph structure encodes physical relationships:
    - Isotopes connected by reactions (n,γ), (n,2n), (n,f), etc.
    - Features include nuclear properties and reaction characteristics
    - Multi-edge graph: multiple reaction channels between isotopes
    """

    def __init__(self, df: pd.DataFrame, energy_bins: np.ndarray):
        """
        Initialize graph builder.

        Args:
            df: DataFrame with nuclear data
            energy_bins: Energy grid for cross-section evaluation
        """
        self.df = df
        self.energy_bins = energy_bins

        # Build isotope registry
        self.isotopes = df[['Z', 'A', 'N']].drop_duplicates().reset_index(drop=True)
        self.isotope_to_idx = {
            (row['Z'], row['A']): idx
            for idx, row in self.isotopes.iterrows()
        }
        self.num_isotopes = len(self.isotopes)

    def build_global_graph(self) -> Data:
        """
        Build the global Chart of Nuclides graph.

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, num_node_features]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, num_edge_features]
        """
        # Build node features
        node_features = self._build_node_features()

        # Build edge list and features
        edge_index, edge_attr = self._build_edges()

        # Create PyG Data object
        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            num_nodes=self.num_isotopes,
        )

        return graph

    def _build_node_features(self) -> np.ndarray:
        """
        Build node feature matrix.

        Features for each isotope:
            [Z, A, N, N/Z ratio, Mass excess (approx), Is_Fissile]

        Returns:
            Array of shape [num_isotopes, num_features]
        """
        features = []

        for _, row in self.isotopes.iterrows():
            Z, A, N = row['Z'], row['A'], row['N']

            # Nuclear properties
            nz_ratio = N / Z if Z > 0 else 0.0

            # Approximate mass excess using semi-empirical mass formula (SEMF)
            # This is a simplified version
            mass_excess = self._approximate_mass_excess(Z, A, N)

            # Is this a fissile isotope?
            is_fissile = 1.0 if (Z == 92 and A == 235) or (Z == 94 and A == 239) else 0.0

            # Is this stable? (Very rough heuristic)
            is_stable = 1.0 if abs(N - Z) < 5 and Z < 83 else 0.0

            feat = [
                Z / 100.0,          # Normalized Z
                A / 250.0,          # Normalized A
                N / 150.0,          # Normalized N
                nz_ratio,           # N/Z ratio
                mass_excess / 1e8,  # Normalized mass excess
                is_fissile,         # Binary flag
                is_stable,          # Binary flag
            ]
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def _approximate_mass_excess(self, Z: int, A: int, N: int) -> float:
        """
        Approximate mass excess using semi-empirical mass formula.

        Args:
            Z: Atomic number
            A: Mass number
            N: Neutron number

        Returns:
            Approximate mass excess in eV
        """
        # SEMF parameters (simplified)
        a_v = 15.75e6   # Volume term
        a_s = 17.8e6    # Surface term
        a_c = 0.711e6   # Coulomb term
        a_a = 23.7e6    # Asymmetry term
        a_p = 11.18e6   # Pairing term

        # Calculate binding energy
        volume = a_v * A
        surface = -a_s * (A ** (2/3))
        coulomb = -a_c * (Z ** 2) / (A ** (1/3))
        asymmetry = -a_a * ((N - Z) ** 2) / A

        # Pairing term
        if N % 2 == 0 and Z % 2 == 0:
            pairing = a_p / np.sqrt(A)
        elif N % 2 == 1 and Z % 2 == 1:
            pairing = -a_p / np.sqrt(A)
        else:
            pairing = 0.0

        binding_energy = volume + surface + coulomb + asymmetry + pairing

        # Mass excess ≈ -B/c² (simplified)
        mass_excess = -binding_energy

        return mass_excess

    def _build_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edge connectivity and edge features.

        Edges represent nuclear reactions:
        - (n,γ): Capture - adds neutron
        - (n,f): Fission - splits nucleus
        - (n,2n): Knock-out - removes neutron
        - Elastic: Self-loop

        Returns:
            edge_index: [2, num_edges] array of source-target pairs
            edge_attr: [num_edges, num_features] edge feature matrix
        """
        edge_list = []
        edge_features = []

        # MT code to delta-Z, delta-N mapping
        reaction_deltas = {
            2: (0, 0),      # Elastic (self-loop)
            18: None,       # Fission (special case - multiple products)
            102: (0, 1),    # (n,γ) - adds neutron
            16: (0, -1),    # (n,2n) - removes neutron
        }

        for _, row in self.df.iterrows():
            src_Z, src_A = int(row['Z']), int(row['A'])
            mt_code = int(row['MT'])

            if src_Z not in self.isotope_to_idx:
                continue

            src_idx = self.isotope_to_idx.get((src_Z, src_A))
            if src_idx is None:
                continue

            # Determine target isotope based on reaction
            if mt_code in reaction_deltas:
                delta = reaction_deltas[mt_code]

                if delta is None:  # Fission - skip for now (multi-edge)
                    continue

                delta_Z, delta_N = delta
                tgt_Z = src_Z + delta_Z
                tgt_A = src_A + delta_N  # Assumes delta_A = delta_N for these reactions

                tgt_idx = self.isotope_to_idx.get((tgt_Z, tgt_A))

                if tgt_idx is not None:
                    # Create edge
                    edge_list.append([src_idx, tgt_idx])

                    # Edge features: [MT_code, Q_value, Threshold, CrossSection_avg]
                    q_value = row['Q_Value']
                    threshold = row['Threshold']
                    cross_section = row['CrossSection']

                    edge_feat = [
                        mt_code / 100.0,        # Normalized MT code
                        q_value / 1e8,          # Normalized Q-value
                        threshold / 1e7,        # Normalized threshold
                        np.log10(cross_section + 1e-10),  # Log cross section
                    ]
                    edge_features.append(edge_feat)

        if len(edge_list) == 0:
            # Return empty graph
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 4), dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_attr = np.array(edge_features, dtype=np.float32)

        return edge_index, edge_attr

    def build_energy_graph(self, energy: float) -> Data:
        """
        Build graph at a specific energy.

        The cross sections are energy-dependent, so we create
        energy-specific graphs for training.

        Args:
            energy: Incident neutron energy (eV)

        Returns:
            PyG Data object with energy-specific cross sections
        """
        # Filter data at this energy
        df_energy = self.df[np.isclose(self.df['Energy'], energy, rtol=1e-3)]

        if len(df_energy) == 0:
            # Find closest energy
            idx = np.argmin(np.abs(self.energy_bins - energy))
            closest_energy = self.energy_bins[idx]
            df_energy = self.df[np.isclose(self.df['Energy'], closest_energy, rtol=1e-3)]

        # Build node features (same as global)
        node_features = self._build_node_features()

        # Build edges with energy-specific cross sections
        edge_list = []
        edge_features = []

        for _, row in df_energy.iterrows():
            src_Z, src_A = int(row['Z']), int(row['A'])
            src_idx = self.isotope_to_idx.get((src_Z, src_A))

            if src_idx is None:
                continue

            # For simplicity, use self-loops with cross section as feature
            edge_list.append([src_idx, src_idx])

            # Edge features at this energy
            edge_feat = [
                row['MT'] / 100.0,
                row['Q_Value'] / 1e8,
                row['Threshold'] / 1e7,
                np.log10(row['CrossSection'] + 1e-10),
            ]
            edge_features.append(edge_feat)

        if len(edge_list) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 4), dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_attr = np.array(edge_features, dtype=np.float32)

        # Add energy as graph-level attribute
        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            energy=torch.tensor([energy], dtype=torch.float32),
            num_nodes=self.num_isotopes,
        )

        return graph

    def build_isotope_subgraph(self, Z: int, A: int, k_hops: int = 2) -> Data:
        """
        Build k-hop subgraph around a specific isotope.

        Args:
            Z: Atomic number
            A: Mass number
            k_hops: Number of hops to include

        Returns:
            PyG Data subgraph
        """
        # This is a simplified version - full implementation would use
        # torch_geometric.utils.k_hop_subgraph

        center_idx = self.isotope_to_idx.get((Z, A))
        if center_idx is None:
            raise ValueError(f"Isotope {Z}-{A} not found in dataset")

        # For now, return the full graph
        # TODO: Implement proper k-hop subgraph extraction
        return self.build_global_graph()
