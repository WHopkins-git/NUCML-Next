"""
Sensitivity-Weighted Loss
=========================

Solves the Validation Paradox by weighting loss according to reactor sensitivity.

The Paradox:
    Model A: MSE = 0.01, but poor on U-235 fission (critical for reactivity)
    Model B: MSE = 0.02, but excellent on U-235 fission

    Standard training picks Model A (lower MSE)
    But Model B is safer for reactor applications!

The Solution:
    L_weighted = Σ_i S_i · (σ_pred,i - σ_true,i)²

    where S_i = |∂k/∂σ_i| = sensitivity coefficient from OpenMC

    This prioritizes accuracy on safety-critical reactions!
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SensitivityWeightedLoss(nn.Module):
    """
    Loss function weighted by reactor sensitivity coefficients.

    Educational Purpose:
        Students learn that:
        1. Not all errors are equal!
        2. 1% error in U-235 fission >> 10% error in minor actinide capture
        3. Reactor safety metrics matter more than geometric MSE
    """

    def __init__(
        self,
        base_loss: str = 'mse',
        use_log_space: bool = True,
        normalize_weights: bool = True,
    ):
        """
        Initialize sensitivity-weighted loss.

        Args:
            base_loss: Base loss type ('mse', 'mae', 'huber')
            use_log_space: Compute loss in log space
            normalize_weights: Normalize sensitivity weights to [0, 1]
        """
        super().__init__()

        self.base_loss = base_loss
        self.use_log_space = use_log_space
        self.normalize_weights = normalize_weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sensitivity_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sensitivity-weighted loss.

        Args:
            predictions: Predicted cross sections [batch, seq_len, 1]
            targets: Ground truth [batch, seq_len, 1]
            sensitivity_weights: Sensitivity coefficients [batch, seq_len, 1]
                                If None, use uniform weights

        Returns:
            Weighted loss scalar
        """
        # Use log space if requested
        if self.use_log_space:
            pred = torch.log10(predictions + 1e-10)
            targ = torch.log10(targets + 1e-10)
        else:
            pred = predictions
            targ = targets

        # Compute base loss
        if self.base_loss == 'mse':
            loss = (pred - targ) ** 2
        elif self.base_loss == 'mae':
            loss = torch.abs(pred - targ)
        elif self.base_loss == 'huber':
            loss = F.smooth_l1_loss(pred, targ, reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        # Apply sensitivity weights
        if sensitivity_weights is not None:
            if self.normalize_weights:
                # Normalize to [0, 1] for numerical stability
                weights = sensitivity_weights / (sensitivity_weights.max() + 1e-10)
            else:
                weights = sensitivity_weights

            weighted_loss = loss * weights
        else:
            weighted_loss = loss

        return weighted_loss.mean()

    def get_sensitivity_weights_from_dict(
        self,
        sensitivity_dict: Dict[tuple, float],
        isotope_ids: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert sensitivity dictionary to weight tensor.

        Args:
            sensitivity_dict: Dict mapping (Z, A, MT, E_idx) -> sensitivity
            isotope_ids: Isotope identifiers [batch]
            energies: Energy values [batch, seq_len]

        Returns:
            Weight tensor [batch, seq_len, 1]
        """
        batch_size, seq_len = energies.shape

        weights = torch.ones(batch_size, seq_len, 1)

        for i in range(batch_size):
            isotope_id = isotope_ids[i].item()
            for j in range(seq_len):
                energy = energies[i, j].item()

                # Look up sensitivity (simplified - in practice, need interpolation)
                key = (isotope_id, energy)
                if key in sensitivity_dict:
                    weights[i, j, 0] = sensitivity_dict[key]

        return weights


class AdaptiveSensitivityWeighting(nn.Module):
    """
    Adaptive weighting that increases emphasis on sensitive reactions during training.

    Training Schedule:
        Early epochs: Focus on all data (uniform weights)
        Later epochs: Emphasize safety-critical reactions (sensitivity weights)

    This prevents overfitting to a few high-sensitivity points early in training.
    """

    def __init__(
        self,
        max_sensitivity_weight: float = 10.0,
        warmup_epochs: int = 10,
    ):
        """
        Initialize adaptive weighting.

        Args:
            max_sensitivity_weight: Maximum weight multiplier
            warmup_epochs: Number of epochs before full sensitivity weighting
        """
        super().__init__()

        self.max_sensitivity_weight = max_sensitivity_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def get_weight_scale(self) -> float:
        """
        Get current weight scale based on training progress.

        Returns:
            Scale factor in [0, 1]
        """
        if self.current_epoch < self.warmup_epochs:
            # Linear ramp-up
            return self.current_epoch / self.warmup_epochs
        else:
            return 1.0

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sensitivity_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptively weighted loss.

        Args:
            predictions: Predictions
            targets: Targets
            sensitivity_weights: Sensitivity coefficients

        Returns:
            Loss
        """
        # Get current weight scale
        scale = self.get_weight_scale()

        # Interpolate between uniform and sensitivity weights
        effective_weights = 1.0 + scale * (sensitivity_weights - 1.0)

        # Clamp to max weight
        effective_weights = torch.clamp(effective_weights, 1.0, self.max_sensitivity_weight)

        # Compute weighted MSE
        loss = ((predictions - targets) ** 2) * effective_weights

        return loss.mean()

    def step_epoch(self):
        """Increment epoch counter."""
        self.current_epoch += 1
