"""
Unitarity Constraint Module
============================

Standalone module for unitarity constraint checking and enforcement.
"""

import torch
import torch.nn as nn
from typing import Dict


class UnitarityConstraint(nn.Module):
    """
    Enforces unitarity: sum of partial cross sections ≤ total cross section.

    Physical Meaning:
        The optical theorem relates total cross section to forward scattering.
        Partial cross sections (elastic, capture, fission, etc.) must sum to ≤ total.
    """

    def __init__(self, weight: float = 0.1):
        """
        Initialize unitarity constraint.

        Args:
            weight: Penalty weight for violations
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        partial_xs: Dict[str, torch.Tensor],
        total_xs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute unitarity penalty.

        Args:
            partial_xs: Dict of partial cross sections
            total_xs: Total cross section

        Returns:
            Penalty term
        """
        sum_partial = sum(partial_xs.values())
        violation = torch.relu(sum_partial - total_xs)
        return self.weight * violation.mean()
