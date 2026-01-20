"""
Physics-Informed Loss Functions
================================

Custom loss functions that encode nuclear physics constraints.

Standard MSE Loss: L = (σ_pred - σ_true)²
Physics-Informed Loss: L = L_MSE + λ₁·L_physics + λ₂·L_sensitivity

This solves the Validation Paradox:
    Minimize MSE → Good fit to data
    Enforce physics → Realistic predictions
    Weight by sensitivity → Prioritize reactor safety

Constraints:
    1. Unitarity: Σ σ_partial ≤ σ_total
    2. Thresholds: σ(E < E_thresh) = 0
    3. Smoothness: |dσ/dE| < M (bounded derivatives)
    4. Positivity: σ(E) ≥ 0 (enforced by architecture)
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function with physics constraints.

    L_total = L_MSE + λ_physics · L_physics + λ_smooth · L_smooth
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        physics_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        use_log_space: bool = True,
    ):
        """
        Initialize physics-informed loss.

        Args:
            mse_weight: Weight for MSE term
            physics_weight: Weight for physics constraints
            smoothness_weight: Weight for smoothness penalty
            use_log_space: Compute MSE in log space (better for cross sections)
        """
        super().__init__()

        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.smoothness_weight = smoothness_weight
        self.use_log_space = use_log_space

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        energies: Optional[torch.Tensor] = None,
        thresholds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.

        Args:
            predictions: Predicted cross sections [batch, seq_len, 1]
            targets: Ground truth cross sections [batch, seq_len, 1]
            energies: Energy values [batch, seq_len] (for threshold/smoothness)
            thresholds: Reaction threshold energies [batch] (optional)

        Returns:
            Dictionary with loss components
        """
        # 1. MSE Loss (data fitting)
        if self.use_log_space:
            log_pred = torch.log10(predictions + 1e-10)
            log_target = torch.log10(targets + 1e-10)
            mse_loss = F.mse_loss(log_pred, log_target)
        else:
            mse_loss = F.mse_loss(predictions, targets)

        # 2. Physics constraints loss
        physics_loss = torch.tensor(0.0, device=predictions.device)

        # Positivity (should be enforced by architecture, but check anyway)
        negative_penalty = F.relu(-predictions).mean()
        physics_loss = physics_loss + negative_penalty

        # Threshold constraint: σ(E < E_thresh) should be small
        if energies is not None and thresholds is not None:
            threshold_mask = energies.unsqueeze(-1) < thresholds.unsqueeze(1).unsqueeze(-1)
            threshold_violation = (predictions * threshold_mask.float()).mean()
            physics_loss = physics_loss + threshold_violation

        # 3. Smoothness constraint
        smoothness_loss = torch.tensor(0.0, device=predictions.device)

        if energies is not None:
            # Compute finite differences dσ/dE
            pred_squeezed = predictions.squeeze(-1)  # [batch, seq_len]
            energy_diffs = energies[:, 1:] - energies[:, :-1]  # [batch, seq_len-1]
            xs_diffs = pred_squeezed[:, 1:] - pred_squeezed[:, :-1]  # [batch, seq_len-1]

            # Avoid division by zero
            derivatives = xs_diffs / (energy_diffs + 1e-10)

            # Penalize large derivatives (resonances should be smooth!)
            smoothness_loss = derivatives.abs().mean()

        # Total loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.physics_weight * physics_loss +
            self.smoothness_weight * smoothness_loss
        )

        return {
            'total': total_loss,
            'mse': mse_loss,
            'physics': physics_loss,
            'smoothness': smoothness_loss,
        }


class UnitarityConstraint(nn.Module):
    """
    Unitarity constraint: Σ σ_partial ≤ σ_total

    The sum of partial cross sections (elastic, inelastic, capture, etc.)
    cannot exceed the total cross section.

    This is a fundamental physics constraint from conservation of probability.
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
        partial_cross_sections: Dict[str, torch.Tensor],
        total_cross_section: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute unitarity violation penalty.

        Args:
            partial_cross_sections: Dictionary of partial XS tensors
                                   {'elastic': [...], 'capture': [...], ...}
            total_cross_section: Total cross section tensor

        Returns:
            Penalty for unitarity violations
        """
        # Sum all partial cross sections
        sum_partials = torch.zeros_like(total_cross_section)
        for partial_xs in partial_cross_sections.values():
            sum_partials = sum_partials + partial_xs

        # Penalize violations: sum_partials > total
        violation = F.relu(sum_partials - total_cross_section)

        penalty = self.weight * violation.mean()

        return penalty


class ThresholdConstraint(nn.Module):
    """
    Threshold constraint: σ(E) = 0 for E < E_threshold

    Reactions like (n,2n) are endothermic - they require minimum energy.
    Below threshold, cross section MUST be zero.
    """

    def __init__(self, weight: float = 1.0, sharpness: float = 10.0):
        """
        Initialize threshold constraint.

        Args:
            weight: Penalty weight
            sharpness: Steepness of threshold function (higher = sharper)
        """
        super().__init__()
        self.weight = weight
        self.sharpness = sharpness

    def forward(
        self,
        predictions: torch.Tensor,
        energies: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute threshold violation penalty.

        Args:
            predictions: Cross sections [batch, seq_len, 1]
            energies: Energies [batch, seq_len]
            thresholds: Threshold energies [batch, 1]

        Returns:
            Penalty for threshold violations
        """
        # Mask for sub-threshold energies
        below_threshold = energies.unsqueeze(-1) < thresholds.unsqueeze(1)

        # Penalty: cross section should be zero below threshold
        violation = predictions * below_threshold.float()

        penalty = self.weight * violation.mean()

        return penalty


class SmoothnessConstraint(nn.Module):
    """
    Smoothness constraint: Bounded derivatives

    Real cross sections are smooth functions (even resonances have smooth shapes).
    Penalize large jumps or oscillations.
    """

    def __init__(self, weight: float = 0.01, max_derivative: float = 100.0):
        """
        Initialize smoothness constraint.

        Args:
            weight: Penalty weight
            max_derivative: Maximum allowed |dσ/dE|
        """
        super().__init__()
        self.weight = weight
        self.max_derivative = max_derivative

    def forward(
        self,
        predictions: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute smoothness penalty.

        Args:
            predictions: Cross sections [batch, seq_len, 1]
            energies: Energies [batch, seq_len]

        Returns:
            Smoothness penalty
        """
        # Compute finite differences
        pred_squeezed = predictions.squeeze(-1)
        energy_diffs = energies[:, 1:] - energies[:, :-1]
        xs_diffs = pred_squeezed[:, 1:] - pred_squeezed[:, :-1]

        # Derivatives
        derivatives = xs_diffs / (energy_diffs + 1e-10)

        # Penalize large derivatives
        violation = F.relu(derivatives.abs() - self.max_derivative)

        penalty = self.weight * violation.mean()

        return penalty


class ConservationLawConstraint(nn.Module):
    """
    Nuclear reaction conservation laws.

    For reactions A(n,X)B:
        - Charge conservation: Z_A + 0 = Z_B + Z_X
        - Mass conservation: A_A + 1 ≈ A_B + A_X (baryon number)
        - Energy-momentum conservation: Encoded in Q-value

    This is more of a data validation constraint than a loss term,
    but can be used to check predictions.
    """

    def __init__(self):
        """Initialize conservation law checker."""
        super().__init__()

    def check_charge_conservation(
        self,
        Z_initial: int,
        Z_final: int,
        Z_ejectile: int,
    ) -> bool:
        """
        Check charge conservation for reaction.

        Args:
            Z_initial: Initial nucleus atomic number
            Z_final: Final nucleus atomic number
            Z_ejectile: Ejectile atomic number

        Returns:
            True if conserved
        """
        return Z_initial + 0 == Z_final + Z_ejectile  # neutron has Z=0

    def check_mass_conservation(
        self,
        A_initial: int,
        A_final: int,
        A_ejectile: int,
    ) -> bool:
        """
        Check mass number conservation.

        Args:
            A_initial: Initial nucleus mass number
            A_final: Final nucleus mass number
            A_ejectile: Ejectile mass number

        Returns:
            True if conserved
        """
        return A_initial + 1 == A_final + A_ejectile  # neutron has A=1
