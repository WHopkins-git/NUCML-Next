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
# GraphBuilder requires torch - lazy import to avoid forcing torch dependency
# from nucml_next.data.graph_builder import GraphBuilder
from nucml_next.data.tabular_projector import TabularProjector

# Re-export ingestion for backward compatibility
from nucml_next.ingest import X4Ingestor, ingest_x4, AME2020Loader

def __getattr__(name):
    """Lazy import for GraphBuilder (requires torch)."""
    if name == "GraphBuilder":
        from nucml_next.data.graph_builder import GraphBuilder
        return GraphBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NucmlDataset",
    "GraphBuilder",
    "TabularProjector",
    "X4Ingestor",
    "ingest_x4",
    "AME2020Loader",
]
