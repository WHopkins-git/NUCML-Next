"""
Nuclide Graph Neural Network
=============================

GNN that learns embeddings for isotopes based on the Chart of Nuclides topology.

Why GNN for Nuclear Data?
    - Isotopes are not independent! They're connected by reactions
    - U-235 and U-238 differ by 3 neutrons → should have similar embeddings
    - (n,γ) connects isotopes along N=const lines
    - (n,2n) connects along diagonal lines

Architecture:
    - Input: Node features [Z, A, N, N/Z, mass_excess, ...]
    - GNN layers: Message passing on reaction graph
    - Output: Rich isotope embeddings that encode topology

This solves the extrapolation problem!
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch


class NuclideGNN(nn.Module):
    """
    Graph Neural Network for learning nuclide embeddings.

    The network learns to propagate information through the nuclear
    reaction network, creating embeddings that respect physical relationships.

    Example:
        U-235 (fissile) should be close to U-233 (also fissile)
        but far from U-238 (fertile)
    """

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 3,
        gnn_type: str = 'GCN',
        dropout: float = 0.1,
    ):
        """
        Initialize Nuclide GNN.

        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            embedding_dim: Final embedding dimension
            num_layers: Number of GNN layers
            gnn_type: 'GCN' or 'GAT' (Graph Attention)
            dropout: Dropout rate
        """
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # GNN layers
        self.conv_layers = nn.ModuleList()

        if gnn_type == 'GCN':
            for i in range(num_layers):
                in_dim = hidden_dim
                out_dim = hidden_dim if i < num_layers - 1 else embedding_dim
                self.conv_layers.append(GCNConv(in_dim, out_dim))

        elif gnn_type == 'GAT':
            # Graph Attention Networks learn importance of edges
            for i in range(num_layers):
                in_dim = hidden_dim
                out_dim = hidden_dim if i < num_layers - 1 else embedding_dim
                heads = 4 if i < num_layers - 1 else 1  # Multi-head attention
                self.conv_layers.append(
                    GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
                )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else embedding_dim)
            for i in range(num_layers)
        ])

        # Edge feature encoder (optional)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Edge features: [MT, Q, Threshold, XS]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        data: Data,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_features]
            return_attention: If True and using GAT, return attention weights

        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Optional: Encode edge features
        # For now, we use simple message passing without edge features
        # In advanced version, edge_attr could modulate message passing

        # GNN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_input = x

            # Message passing
            if self.gnn_type == 'GCN':
                x = conv(x, edge_index)
            elif self.gnn_type == 'GAT':
                x = conv(x, edge_index, return_attention_weights=False)

            # Batch norm
            x = bn(x)

            # Activation (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # Residual connection (if dimensions match)
                if x_input.shape == x.shape:
                    x = x + x_input

        return x

    def get_isotope_embedding(
        self,
        data: Data,
        isotope_idx: int,
    ) -> torch.Tensor:
        """
        Get embedding for a specific isotope.

        Args:
            data: PyG Data object
            isotope_idx: Node index of isotope

        Returns:
            Embedding vector [embedding_dim]
        """
        embeddings = self.forward(data)
        return embeddings[isotope_idx]

    def compute_similarity(
        self,
        data: Data,
        isotope_idx_1: int,
        isotope_idx_2: int,
    ) -> float:
        """
        Compute cosine similarity between two isotopes.

        Educational Use:
            Check if fissile isotopes (U-235, Pu-239) are close
            and fertile isotopes (U-238, Th-232) are close.

        Args:
            data: PyG Data object
            isotope_idx_1: First isotope index
            isotope_idx_2: Second isotope index

        Returns:
            Cosine similarity in [-1, 1]
        """
        embeddings = self.forward(data)
        emb1 = embeddings[isotope_idx_1]
        emb2 = embeddings[isotope_idx_2]

        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()

    def get_nearest_neighbors(
        self,
        data: Data,
        isotope_idx: int,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest neighbors in embedding space.

        Educational Use:
            Visualize which isotopes the GNN thinks are similar.
            Should reflect physics: similar Z, A, or reaction behavior.

        Args:
            data: PyG Data object
            isotope_idx: Query isotope index
            k: Number of neighbors

        Returns:
            (neighbor_indices, distances)
        """
        embeddings = self.forward(data)
        query_emb = embeddings[isotope_idx].unsqueeze(0)

        # Compute distances to all isotopes
        distances = torch.cdist(query_emb, embeddings)

        # Get k nearest (excluding self)
        k_nearest = torch.topk(distances, k + 1, largest=False)
        neighbor_indices = k_nearest.indices[0, 1:]  # Exclude self
        neighbor_distances = k_nearest.values[0, 1:]

        return neighbor_indices, neighbor_distances


class NuclideGNNWithPooling(nn.Module):
    """
    GNN with graph-level readout for predicting global properties.

    Use case: Predict average cross section for a set of isotopes
    or classify reactor type based on isotope composition.
    """

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 3,
        output_dim: int = 1,
        pooling: str = 'mean',
    ):
        """
        Initialize GNN with pooling.

        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension
            embedding_dim: Embedding dimension
            num_layers: Number of GNN layers
            output_dim: Output dimension (e.g., 1 for regression)
            pooling: 'mean', 'max', or 'add'
        """
        super().__init__()

        self.gnn = NuclideGNN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )

        self.pooling = pooling

        # Graph-level prediction head
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass with graph pooling.

        Args:
            data: PyG Data object (can be batched)

        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        # Get node embeddings
        node_embeddings = self.gnn(data)

        # Pool to graph-level representation
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(
                node_embeddings,
                data.batch if hasattr(data, 'batch') else torch.zeros(node_embeddings.size(0), dtype=torch.long)
            )
        else:
            raise NotImplementedError(f"Pooling {self.pooling} not implemented")

        # Predict
        output = self.predictor(graph_embedding)

        return output
