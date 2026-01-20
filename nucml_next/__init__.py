"""
NUCML-Next: Next-Generation Nuclear Data Evaluation Framework
================================================================

An educational platform for physics-informed machine learning in nuclear engineering.

Learning Pathway:
    1. Legacy Baselines (XGBoost, Decision Trees) → Show limitations
    2. Graph Neural Networks → Capture nuclear topology
    3. Transformers → Smooth energy-dependent predictions
    4. Physics-Informed Loss → Enforce physical constraints
    5. OpenMC Validation → Solve the validation paradox

Modules:
    data: Apache Arrow/Parquet data fabric with dual-view (Graph + Tabular)
    baselines: Classical ML models (XGBoost, Decision Trees)
    model: Deep learning architectures (GNN, Transformer)
    physics: Physics-informed loss functions (Unitarity, Thresholds)
    validation: OpenMC integration for reactor validation
    utils: Visualization and helper utilities

Authors: NUCML-Next Team
License: MIT
"""

__version__ = "1.0.0"

from nucml_next import data, baselines, model, physics, validation, utils

__all__ = [
    "data",
    "baselines",
    "model",
    "physics",
    "validation",
    "utils",
]
