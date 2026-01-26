"""
Physics-Informed Loss Functions Module
=======================================

Custom loss functions that encode nuclear physics constraints.

Key Components:
    PhysicsInformedLoss: Combined loss with MSE + physics constraints
    UnitarityConstraint: Ensures unitarity of scattering matrix
    ThresholdConstraint: Enforces reaction thresholds
    SensitivityWeightedLoss: Weights loss by reactor sensitivity coefficients

Physics Constraints:
    1. Unitarity: Sum of partial cross sections ≤ total cross section
    2. Thresholds: σ(E) = 0 for E < E_threshold
    3. Smoothness: Derivatives should be bounded (no unphysical jumps)
    4. Conservation: Nuclear reaction conservation laws

This solves the Validation Paradox:
    Low MSE ≠ Safe Reactor
    Instead: Minimize sensitivity-weighted error on safety-critical reactions
"""

from nucml_next.physics.physics_informed_loss import PhysicsInformedLoss
from nucml_next.physics.unitarity_constraint import UnitarityConstraint
from nucml_next.physics.threshold_constraint import ThresholdConstraint
from nucml_next.physics.sensitivity_weighted_loss import SensitivityWeightedLoss

__all__ = [
    "PhysicsInformedLoss",
    "UnitarityConstraint",
    "ThresholdConstraint",
    "SensitivityWeightedLoss",
]
