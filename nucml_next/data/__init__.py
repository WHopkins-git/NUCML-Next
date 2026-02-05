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
    """Lazy import for torch-dependent modules."""
    if name == "GraphBuilder":
        from nucml_next.data.graph_builder import GraphBuilder
        return GraphBuilder
    if name == "HoldoutConfig":
        from nucml_next.experiment import HoldoutConfig
        return HoldoutConfig
    # Outlier detection (requires torch/gpytorch)
    if name == "SVGPOutlierDetector":
        from nucml_next.data.outlier_detection import SVGPOutlierDetector
        return SVGPOutlierDetector
    if name == "SVGPConfig":
        from nucml_next.data.outlier_detection import SVGPConfig
        return SVGPConfig
    if name == "extract_svgp_hyperparameters":
        from nucml_next.data.outlier_detection import extract_svgp_hyperparameters
        return extract_svgp_hyperparameters
    # Per-experiment outlier detection (new approach)
    if name == "ExperimentOutlierDetector":
        from nucml_next.data.experiment_outlier import ExperimentOutlierDetector
        return ExperimentOutlierDetector
    if name == "ExperimentOutlierConfig":
        from nucml_next.data.experiment_outlier import ExperimentOutlierConfig
        return ExperimentOutlierConfig
    if name == "ExactGPExperiment":
        from nucml_next.data.experiment_gp import ExactGPExperiment
        return ExactGPExperiment
    if name == "ExactGPExperimentConfig":
        from nucml_next.data.experiment_gp import ExactGPExperimentConfig
        return ExactGPExperimentConfig
    if name == "ConsensusBuilder":
        from nucml_next.data.consensus import ConsensusBuilder
        return ConsensusBuilder
    if name == "ConsensusConfig":
        from nucml_next.data.consensus import ConsensusConfig
        return ConsensusConfig
    if name == "prepare_log_uncertainties":
        from nucml_next.data.experiment_gp import prepare_log_uncertainties
        return prepare_log_uncertainties
    if name == "compute_wasserstein_calibration":
        from nucml_next.data.calibration import compute_wasserstein_calibration
        return compute_wasserstein_calibration
    if name == "optimize_lengthscale_wasserstein":
        from nucml_next.data.calibration import optimize_lengthscale_wasserstein
        return optimize_lengthscale_wasserstein
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
    "HoldoutConfig",
    # Legacy SVGP outlier detection
    "SVGPOutlierDetector",
    "SVGPConfig",
    "extract_svgp_hyperparameters",
    # Per-experiment outlier detection (new approach)
    "ExperimentOutlierDetector",
    "ExperimentOutlierConfig",
    "ExactGPExperiment",
    "ExactGPExperimentConfig",
    "ConsensusBuilder",
    "ConsensusConfig",
    "prepare_log_uncertainties",
    "compute_wasserstein_calibration",
    "optimize_lengthscale_wasserstein",
]
