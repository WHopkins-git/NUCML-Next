"""
Threshold Constraint Module
============================

Enforces reaction threshold behavior.
"""

import torch
import torch.nn as nn


class ThresholdConstraint(nn.Module):
    """
    Enforces threshold: σ(E) = 0 for E < E_threshold.

    Physical Meaning:
        Endothermic reactions (n,2n), (n,3n), etc. require minimum energy.
        Below threshold, reaction is kinematically impossible.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize threshold constraint.

        Args:
            weight: Penalty weight
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: torch.Tensor,
        energies: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute threshold penalty.

        Args:
            predictions: Cross sections [batch, seq_len, 1]
            energies: Energies [batch, seq_len]
            thresholds: Threshold energies [batch, 1]

        Returns:
            Penalty term
        """
        below_threshold = energies.unsqueeze(-1) < thresholds.unsqueeze(1)
        violation = predictions * below_threshold.float()
        return self.weight * violation.mean()
