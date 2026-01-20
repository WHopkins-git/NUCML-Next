"""
Data Fabric Module
==================

Dual-view data handling for nuclear cross-section data:
- Graph View: PyG Data objects for GNN training
- Tabular View: Pandas DataFrames for classical ML

Key Components:
    NucmlDataset: Main dataset class with dual-view interface
    GraphBuilder: Constructs nuclide topology graphs
    TabularProjector: Projects graph data to tabular format
"""

from nucml_next.data.dataset import NucmlDataset
from nucml_next.data.graph_builder import GraphBuilder
from nucml_next.data.tabular_projector import TabularProjector

__all__ = [
    "NucmlDataset",
    "GraphBuilder",
    "TabularProjector",
]
