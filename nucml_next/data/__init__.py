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
    EXFORIngestor: Bulk ingestor for IAEA EXFOR-X5json database
    AME2020Loader: Atomic Mass Evaluation 2020 data loader
"""

from nucml_next.data.dataset import NucmlDataset
from nucml_next.data.graph_builder import GraphBuilder
from nucml_next.data.tabular_projector import TabularProjector
from nucml_next.data.exfor_ingestor import EXFORIngestor, AME2020Loader, ingest_exfor

__all__ = [
    "NucmlDataset",
    "GraphBuilder",
    "TabularProjector",
    "EXFORIngestor",
    "AME2020Loader",
    "ingest_exfor",
]
