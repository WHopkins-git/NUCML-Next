"""
Data Fabric Module
==================

Dual-view data handling for nuclear cross-section data:
- Graph View: PyG Data objects for GNN training
- Tabular View: Pandas DataFrames with tier-based feature engineering

Key Components:
    NucmlDataset: Main dataset class with dual-view interface
    GraphBuilder: Constructs nuclide topology graphs
    FeatureGenerator: Generates tier-based features from nuclear data
"""

from nucml_next.data.dataset import NucmlDataset
# GraphBuilder requires torch - lazy import to avoid forcing torch dependency
# from nucml_next.data.graph_builder import GraphBuilder
from nucml_next.data.selection import (
    DataSelection,
    default_selection,
    full_spectrum_selection,
    evaluation_selection,
    REACTOR_CORE_MT,
    THRESHOLD_MT,
    FISSION_DETAILS_MT,
)
from nucml_next.data.mt_codes import (
    MT_NAMES,
    MT_CATEGORIES,
    get_mt_name,
    get_mt_category,
    get_reactor_critical_mt_codes,
    get_common_mt_codes,
)

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
    "DataSelection",
    "default_selection",
    "full_spectrum_selection",
    "evaluation_selection",
    "REACTOR_CORE_MT",
    "THRESHOLD_MT",
    "FISSION_DETAILS_MT",
    "MT_NAMES",
    "MT_CATEGORIES",
    "get_mt_name",
    "get_mt_category",
    "get_reactor_critical_mt_codes",
    "get_common_mt_codes",
    "X4Ingestor",
    "ingest_x4",
    "AME2020Loader",
]
