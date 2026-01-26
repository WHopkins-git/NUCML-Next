"""
Deep Learning Models Module
============================

Physics-informed neural architectures for nuclear data evaluation.

Key Components:
    NuclideGNN: Graph Neural Network for learning nuclear topology
    EnergyTransformer: Transformer for smooth energy-dependent predictions
    GNNTransformerEvaluator: Integrated model combining GNN + Transformer

Architecture:
    1. GNN embeds nuclides based on Chart of Nuclides topology
    2. Transformer predicts smooth cross-section curves σ(E)
    3. Physics-informed loss enforces physical constraints

This solves:
    - Extrapolation problem (GNN captures physics relationships)
    - Smoothness problem (Transformer handles continuous energy)
    - Validation paradox (physics loss prioritizes reactor safety)
"""

from nucml_next.model.nuclide_gnn import NuclideGNN
from nucml_next.model.energy_transformer import EnergyTransformer
from nucml_next.model.gnn_transformer_evaluator import GNNTransformerEvaluator

__all__ = [
    "NuclideGNN",
    "EnergyTransformer",
    "GNNTransformerEvaluator",
]
