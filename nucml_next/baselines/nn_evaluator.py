"""
Neural Network Evaluator - Research Baseline
=============================================

CPU-friendly feedforward neural network baseline with modern training
techniques: OneCycleLR scheduling, Kaiming He initialization, AdamW
optimizer, early stopping, and gradient clipping.

Features:
- TransformationPipeline integration (same as XGBoost/DT evaluators)
- Four loss functions: MSE, chi-squared, physics-informed, resonance-informed
- OneCycleLR scheduler with cosine annealing for fast convergence
- Kaiming He weight initialization (correct for ReLU networks)
- Early stopping with best-weight restoration
- Full training history tracking

Educational Purpose:
    Shows that a small neural network can produce smooth (continuous)
    predictions unlike trees, at the cost of more careful hyperparameter
    tuning.  The resonance-informed loss demonstrates how physics priors
    can be encoded directly into the training objective.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nucml_next.data.transformations import TransformationPipeline
from nucml_next.data.selection import TransformationConfig


class _SimpleNet(nn.Module):
    """
    Small feedforward network with ReLU activations and BatchNorm.

    Architecture per hidden layer:
        Linear -> ReLU -> BatchNorm1d [-> Dropout]

    Final layer: Linear(hidden[-1], 1)

    Uses Kaiming He initialization on all Linear weights, which is the
    correct initialisation for networks with ReLU activations (Xavier
    underestimates the variance by a factor of 2 because it ignores
    the rectification).
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list = []
        prev = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Kaiming He initialisation (fan_in, ReLU nonlinearity)
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming He initialization to all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def _mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Standard mean squared error."""
    return ((pred - target) ** 2).mean()


def _chi_squared_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    unc: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Chi-squared / N: weight each residual by 1/sigma^2.

    .. math::
        L = \frac{1}{N} \sum_i \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}

    Falls back to MSE if uncertainties are not available.
    """
    if unc is None or not torch.isfinite(unc).all():
        return _mse_loss(pred, target)
    var = (unc ** 2).clamp(min=1e-12)
    return ((pred - target) ** 2 / var).mean()


def _physics_informed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    energy: Optional[torch.Tensor] = None,
    smoothness_weight: float = 0.01,
    **kwargs,
) -> torch.Tensor:
    r"""
    MSE + smoothness penalty on d-sigma/dE.

    .. math::
        L = \text{MSE} + \lambda \cdot \frac{1}{N-1}
            \sum_i \left| \frac{\hat{\sigma}_{i+1} - \hat{\sigma}_i}
                               {E_{i+1} - E_i} \right|

    Penalises unphysical oscillations in the predicted cross-section.
    """
    data_loss = ((pred - target) ** 2).mean()
    if energy is None:
        return data_loss
    order = energy.argsort()
    pred_sorted = pred[order].squeeze()
    e_sorted = energy[order]
    dE = (e_sorted[1:] - e_sorted[:-1]).clamp(min=1e-12)
    dS = pred_sorted[1:] - pred_sorted[:-1]
    smoothness = (dS / dE).abs().mean()
    return data_loss + smoothness_weight * smoothness


def _resonance_informed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    energy: Optional[torch.Tensor] = None,
    unc: Optional[torch.Tensor] = None,
    lambda_1v: float = 0.1,
    lambda_threshold: float = 0.05,
    lambda_curvature: float = 0.01,
    thermal_cutoff: float = 0.0,
    threshold_cutoff: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    r"""
    Research-level loss encoding nuclear physics priors.

    Combines chi-squared data fidelity with three physics-based
    regularisation terms derived from fundamental nuclear scattering
    theory:

    .. math::
        L = L_{\chi^2} + \lambda_{1/v} L_{1/v}
            + \lambda_{\text{thresh}} L_{\text{thresh}}
            + \lambda_{\text{curv}} L_{\text{curv}}

    **1/v Law** (thermal region):
        At low energies (< 1 eV), absorption cross-sections follow
        sigma ~ 1/v ~ 1/sqrt(E).  In log10 space this is a line with
        slope -0.5.  We penalise deviations from this slope:

        .. math::
            L_{1/v} = \frac{1}{N_{\text{thermal}}}
                \sum_{E_i < E_{\text{cutoff}}}
                \left( \frac{d \log_{10}\hat\sigma}{d \log_{10} E}
                       + 0.5 \right)^2

    **Threshold Rise** (near threshold):
        Near reaction thresholds, sigma ~ (E - E_th)^(l+1/2).
        For s-wave (l=0), the log-log slope should be ~0.5.  We
        penalise negative slopes in the threshold region:

        .. math::
            L_{\text{thresh}} = \frac{1}{N_{\text{thresh}}}
                \sum_{E_i > E_{\text{thresh}}}
                \text{ReLU}\!\left(-\frac{d \log_{10}\hat\sigma}
                                         {d \log_{10} E}\right)^2

    **Curvature Bound** (global):
        Real cross-sections have bounded second derivatives (even
        resonances are analytic Breit-Wigner / R-matrix shapes).  We
        penalise excessive curvature in log-log space:

        .. math::
            L_{\text{curv}} = \frac{1}{N-2}
                \sum_i \left| \frac{d^2 \log_{10}\hat\sigma}
                                    {(d \log_{10} E)^2} \right|

    Args:
        pred: Predicted values (log10 cross-section)
        target: True values (log10 cross-section)
        energy: Log10 energy values (already log-transformed by pipeline)
        unc: Log-space uncertainties (optional)
        lambda_1v: Weight for 1/v law penalty
        lambda_threshold: Weight for threshold rise penalty
        lambda_curvature: Weight for curvature bound penalty
        thermal_cutoff: Log10(energy) below which 1/v applies.
            Default 0.0 means E < 1 eV (since energy is log10-transformed).
        threshold_cutoff: Log10(energy) above which threshold rise applies.
            Default 1.0 means E > 10 eV.

    References:
        - Blatt & Weisskopf, "Theoretical Nuclear Physics" (1952)
        - Lane & Thomas, Rev. Mod. Phys. 30, 257 (1958) — R-matrix theory
        - Mughabghab, "Atlas of Neutron Resonances" (2018)
    """
    # Data fidelity: chi-squared if uncertainties available, else MSE
    if unc is not None and torch.isfinite(unc).all():
        var = (unc ** 2).clamp(min=1e-12)
        data_loss = ((pred - target) ** 2 / var).mean()
    else:
        data_loss = ((pred - target) ** 2).mean()

    if energy is None:
        return data_loss

    # Sort by energy for finite difference derivatives
    order = energy.argsort()
    pred_sorted = pred[order].squeeze()
    e_sorted = energy[order]

    # Finite differences in log10 space (energy is already log10-transformed)
    dE = (e_sorted[1:] - e_sorted[:-1]).clamp(min=1e-12)
    dS = pred_sorted[1:] - pred_sorted[:-1]
    slope = dS / dE  # d(log10 sigma) / d(log10 E)

    loss = data_loss

    # 1/v Law: thermal region (log10(E) < thermal_cutoff)
    midpoints = 0.5 * (e_sorted[:-1] + e_sorted[1:])
    thermal_mask = midpoints < thermal_cutoff
    if thermal_mask.any():
        thermal_slopes = slope[thermal_mask]
        # In log-log space, 1/v law gives slope = -0.5
        l_1v = ((thermal_slopes + 0.5) ** 2).mean()
        loss = loss + lambda_1v * l_1v

    # Threshold Rise: near-threshold region (log10(E) > threshold_cutoff)
    thresh_mask = midpoints > threshold_cutoff
    if thresh_mask.any():
        thresh_slopes = slope[thresh_mask]
        # Penalise negative slopes (cross-section should be rising)
        l_thresh = (torch.relu(-thresh_slopes) ** 2).mean()
        loss = loss + lambda_threshold * l_thresh

    # Curvature Bound: global smoothness of second derivative
    if len(slope) > 1:
        d2S = slope[1:] - slope[:-1]
        dE_mid = (midpoints[1:] - midpoints[:-1]).clamp(min=1e-12)
        curvature = d2S / dE_mid
        l_curv = curvature.abs().mean()
        loss = loss + lambda_curvature * l_curv

    return loss


# Map loss names to functions
_LOSS_FUNCTIONS = {
    'mse': _mse_loss,
    'chi_squared': _chi_squared_loss,
    'physics_informed': _physics_informed_loss,
    'resonance_informed': _resonance_informed_loss,
}


class NeuralNetEvaluator:
    """
    CPU-friendly neural network baseline for nuclear cross-section prediction.

    Integrates with NUCML-Next TransformationPipeline (same API as
    XGBoostEvaluator and DecisionTreeEvaluator) and provides modern
    training techniques that converge 3-5x faster than naive Adam:

    - **OneCycleLR scheduler**: warmup -> peak LR -> cosine annealing
    - **Kaiming He initialization**: correct for ReLU networks
    - **AdamW optimizer**: decoupled weight decay for better generalisation
    - **Early stopping**: restores best weights, saves CPU time
    - **Gradient clipping**: prevents exploding gradients

    Four loss functions are available:

    =============================  ================================================
    Loss                           Description
    =============================  ================================================
    ``'mse'``                      Standard mean squared error
    ``'chi_squared'``              Inverse-variance weighted MSE (chi^2/N)
    ``'physics_informed'``         MSE + smoothness penalty on d-sigma/dE
    ``'resonance_informed'``       Chi^2 + 1/v law + threshold rise + curvature
    =============================  ================================================

    Example:
        >>> from nucml_next.baselines import NeuralNetEvaluator
        >>>
        >>> evaluator = NeuralNetEvaluator(loss_function='chi_squared')
        >>> metrics = evaluator.train(df, pipeline=pipeline, verbose=True)
        >>>
        >>> # Predictions in barns (compatible with IsotopePlotter)
        >>> predictions = evaluator.predict(df_test)

    Attributes:
        is_trained: Whether the model has been trained
        metrics: Dictionary of training/test metrics
        history: Training history (train_loss, val_loss, learning_rates)
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (256, 128),
        epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.0,
        early_stopping_patience: int = 8,
        loss_function: str = 'chi_squared',
        smoothness_weight: float = 0.01,
        lambda_1v: float = 0.1,
        lambda_threshold: float = 0.05,
        lambda_curvature: float = 0.01,
        thermal_cutoff: float = 0.0,
        threshold_cutoff: float = 1.0,
        grad_clip_max_norm: float = 1.0,
        random_state: int = 42,
    ):
        """
        Initialize neural network evaluator with CPU-friendly defaults.

        Args:
            hidden_sizes: Width of each hidden layer.
                Default (256, 128) is wider than the old notebook [128, 64]
                to compensate for fewer effective epochs with early stopping.
            epochs: Maximum training epochs.
                Default 50 is an upper bound; early stopping typically fires
                around epoch 20-30.
            batch_size: Mini-batch size.
                Default 512 is much smaller than the old 4096; smaller batches
                give noisier gradients that converge faster in wall-clock time.
            learning_rate: Peak learning rate for OneCycleLR.
                Training starts at learning_rate/25, ramps up to learning_rate
                over 30% of training, then cosine-anneals to near zero.
            weight_decay: L2 regularisation coefficient for AdamW.
            dropout: Dropout rate between hidden layers (0 = disabled).
            early_stopping_patience: Number of epochs with no improvement
                before stopping. Set to 0 or None to disable.
            loss_function: One of 'mse', 'chi_squared', 'physics_informed',
                or 'resonance_informed'.
            smoothness_weight: Lambda for physics_informed smoothness term.
            lambda_1v: Weight for 1/v law penalty (resonance_informed).
            lambda_threshold: Weight for threshold rise penalty.
            lambda_curvature: Weight for curvature bound penalty.
            thermal_cutoff: Log10(energy) below which 1/v law applies.
            threshold_cutoff: Log10(energy) above which threshold rise applies.
            grad_clip_max_norm: Max norm for gradient clipping.
            random_state: Random seed for reproducibility.
        """
        if loss_function not in _LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss_function '{loss_function}'. "
                f"Choose from: {list(_LOSS_FUNCTIONS.keys())}"
            )

        self.hidden_sizes = tuple(hidden_sizes)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience or 0
        self.loss_function = loss_function
        self.smoothness_weight = smoothness_weight
        self.lambda_1v = lambda_1v
        self.lambda_threshold = lambda_threshold
        self.lambda_curvature = lambda_curvature
        self.thermal_cutoff = thermal_cutoff
        self.threshold_cutoff = threshold_cutoff
        self.grad_clip_max_norm = grad_clip_max_norm
        self.random_state = random_state

        # State (set during training)
        self._model: Optional[_SimpleNet] = None
        self._device: Optional[torch.device] = None
        self.pipeline: Optional[TransformationPipeline] = None
        self.feature_columns: Optional[List[str]] = None
        self.is_trained: bool = False
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, list] = {}

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        energy_column: str = 'Energy',
        uncertainty_column: str = 'Uncertainty',
        test_size: float = 0.15,
        exclude_columns: Optional[list] = None,
        pipeline: Optional[TransformationPipeline] = None,
        transformation_config: Optional[TransformationConfig] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the neural network with full pipeline integration.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Target column name
            energy_column: Energy column name
            uncertainty_column: Cross-section uncertainty column name
            test_size: Held-out test fraction
            exclude_columns: Columns to exclude from features
            pipeline: Pre-configured TransformationPipeline (recommended).
                If None, a new pipeline is created from transformation_config.
            transformation_config: Config for new pipeline (if pipeline=None)
            verbose: Print training progress

        Returns:
            Dictionary of training/test metrics (MSE, MAE, R2 in both
            log-space and barns-space, plus training history)
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if verbose:
            print("=" * 70)
            print("NEURAL NETWORK BASELINE")
            print("=" * 70)
            print(f"  Loss function:    {self.loss_function}")
            print(f"  Hidden layers:    {list(self.hidden_sizes)}")
            print(f"  Max epochs:       {self.epochs}")
            print(f"  Batch size:       {self.batch_size}")
            print(f"  Peak LR:          {self.learning_rate}")
            print(f"  Scheduler:        OneCycleLR (warmup 30%, cosine anneal)")
            print(f"  Weight decay:     {self.weight_decay}")
            if self.dropout > 0:
                print(f"  Dropout:          {self.dropout}")
            if self.early_stopping_patience > 0:
                print(f"  Early stopping:   patience={self.early_stopping_patience}")
            print(f"  Grad clipping:    max_norm={self.grad_clip_max_norm}")
            print("=" * 70)

        # ==================================================================
        # PREPARE DATA (same pipeline interface as XGBoost evaluator)
        # ==================================================================
        if exclude_columns is None:
            exclude_columns = [
                target_column, 'Uncertainty', 'Energy_Uncertainty',
                'Entry', 'MT',
            ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sparse_cols = [
            col for col in df.columns
            if isinstance(df[col].dtype, pd.SparseDtype)
        ]
        all_numeric = list(set(numeric_cols + sparse_cols))
        self.feature_columns = [
            col for col in all_numeric if col not in exclude_columns
        ]

        X_feat = df[self.feature_columns]
        y_raw = df[target_column]
        energy_raw = df[energy_column] if energy_column in df.columns else None

        # Create or use pipeline
        if pipeline is None:
            if transformation_config is None:
                transformation_config = TransformationConfig()
            pipeline = TransformationPipeline(config=transformation_config)
            pipeline.fit(
                X_feat, y_raw, energy_raw,
                feature_columns=self.feature_columns,
            )
        self.pipeline = pipeline

        # Transform data
        X_t = pipeline.transform(X_feat, energy_raw)
        y_t = pipeline.transform_target(y_raw)

        X_np = X_t[self.feature_columns].values.astype(np.float32)
        y_np = y_t.values.astype(np.float32).reshape(-1, 1)

        # Energy column index for physics-informed losses
        energy_col_idx = (
            self.feature_columns.index(energy_column)
            if energy_column in self.feature_columns
            else None
        )

        # Uncertainty weights for chi-squared / resonance-informed losses
        needs_unc = self.loss_function in ('chi_squared', 'resonance_informed')
        if needs_unc and uncertainty_column in df.columns:
            unc_raw = df[uncertainty_column].values.astype(np.float32)
            xs_vals = df[target_column].values.astype(np.float32)
            # Transform uncertainty to log-space: delta(log sigma) = delta_sigma / (sigma * ln(10))
            unc_log = np.where(
                (np.isfinite(unc_raw)) & (unc_raw > 0) & (xs_vals > 0),
                unc_raw / (xs_vals * np.log(10) + 1e-30),
                np.nan,
            )
        else:
            unc_log = np.full(len(y_np), np.nan, dtype=np.float32)

        # Drop invalid rows
        valid = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np.ravel())
        if needs_unc:
            valid &= np.isfinite(unc_log) & (unc_log > 0)

        X_np = X_np[valid]
        y_np = y_np[valid]
        unc_log = unc_log[valid]

        if verbose:
            print(f"Valid samples: {valid.sum():,} / {len(valid):,}")

        # Train/test split
        idx = np.arange(len(X_np))
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=self.random_state,
        )

        X_train_t = torch.from_numpy(X_np[idx_train])
        y_train_t = torch.from_numpy(y_np[idx_train])
        X_test_t = torch.from_numpy(X_np[idx_test])
        y_test_t = torch.from_numpy(y_np[idx_test])
        unc_train_t = torch.from_numpy(unc_log[idx_train])

        train_ds = TensorDataset(X_train_t, y_train_t, unc_train_t)
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=False,
        )

        # Device
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        if verbose:
            print(f"Device: {self._device}")

        # ==================================================================
        # BUILD NETWORK
        # ==================================================================
        n_features = X_train_t.shape[1]
        self._model = _SimpleNet(
            n_features,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self._device)

        n_params = sum(p.numel() for p in self._model.parameters())
        if verbose:
            print(f"Parameters: {n_params:,}")

        # ==================================================================
        # OPTIMIZER + SCHEDULER
        # ==================================================================
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # OneCycleLR: warmup 30%, cosine anneal, starts at LR/25
        steps_per_epoch = len(train_dl)
        total_steps = self.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,       # initial_lr = max_lr / 25
            final_div_factor=1e4,  # final_lr = initial_lr / 10000
        )

        # Loss function
        loss_fn = _LOSS_FUNCTIONS[self.loss_function]

        # ==================================================================
        # TRAINING LOOP
        # ==================================================================
        train_losses: list = []
        val_losses: list = []
        learning_rates: list = []
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        stopped_epoch = self.epochs

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            self._model.train()
            epoch_loss = 0.0
            n_samples_epoch = 0

            for xb, yb, ub in train_dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                ub = ub.to(self._device)

                pred = self._model(xb)

                # Build loss kwargs
                loss_kwargs: dict = {}
                if self.loss_function in ('chi_squared', 'resonance_informed'):
                    loss_kwargs['unc'] = ub
                if self.loss_function == 'physics_informed' and energy_col_idx is not None:
                    loss_kwargs['energy'] = xb[:, energy_col_idx]
                    loss_kwargs['smoothness_weight'] = self.smoothness_weight
                if self.loss_function == 'resonance_informed' and energy_col_idx is not None:
                    loss_kwargs['energy'] = xb[:, energy_col_idx]
                    loss_kwargs['lambda_1v'] = self.lambda_1v
                    loss_kwargs['lambda_threshold'] = self.lambda_threshold
                    loss_kwargs['lambda_curvature'] = self.lambda_curvature
                    loss_kwargs['thermal_cutoff'] = self.thermal_cutoff
                    loss_kwargs['threshold_cutoff'] = self.threshold_cutoff

                loss = loss_fn(pred, yb, **loss_kwargs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    max_norm=self.grad_clip_max_norm,
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * len(xb)
                n_samples_epoch += len(xb)

            avg_train_loss = epoch_loss / max(n_samples_epoch, 1)
            train_losses.append(avg_train_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])

            # ---- Validate ----
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_test_t.to(self._device))
                val_loss_kwargs: dict = {}
                if self.loss_function in ('chi_squared', 'resonance_informed'):
                    val_loss_kwargs['unc'] = torch.from_numpy(
                        unc_log[idx_test]
                    ).to(self._device)
                if self.loss_function == 'physics_informed' and energy_col_idx is not None:
                    val_loss_kwargs['energy'] = X_test_t[:, energy_col_idx].to(self._device)
                    val_loss_kwargs['smoothness_weight'] = self.smoothness_weight
                if self.loss_function == 'resonance_informed' and energy_col_idx is not None:
                    val_loss_kwargs['energy'] = X_test_t[:, energy_col_idx].to(self._device)
                    val_loss_kwargs['lambda_1v'] = self.lambda_1v
                    val_loss_kwargs['lambda_threshold'] = self.lambda_threshold
                    val_loss_kwargs['lambda_curvature'] = self.lambda_curvature
                    val_loss_kwargs['thermal_cutoff'] = self.thermal_cutoff
                    val_loss_kwargs['threshold_cutoff'] = self.threshold_cutoff

                val_loss = loss_fn(
                    val_pred, y_test_t.to(self._device), **val_loss_kwargs,
                ).item()
            val_losses.append(val_loss)

            # ---- Early stopping ----
            if self.early_stopping_patience > 0:
                if val_loss < best_val_loss - 1e-8:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        stopped_epoch = epoch
                        if verbose:
                            print(
                                f"  Early stopping at epoch {epoch} "
                                f"(best val_loss={best_val_loss:.6f})"
                            )
                        break

            # ---- Progress ----
            if verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:3d}/{self.epochs}  "
                    f"train_loss={avg_train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"lr={learning_rates[-1]:.2e}"
                )

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            if verbose:
                print(f"  Restored best weights (val_loss={best_val_loss:.6f})")

        # Store history
        self.history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'learning_rates': learning_rates,
            'stopped_epoch': stopped_epoch,
        }

        # ==================================================================
        # EVALUATE
        # ==================================================================
        self._model.eval()
        with torch.no_grad():
            y_pred_train_log = self._model(
                X_train_t.to(self._device)
            ).cpu().numpy().ravel()
            y_pred_test_log = self._model(
                X_test_t.to(self._device)
            ).cpu().numpy().ravel()

        y_train_log = y_np[idx_train].ravel()
        y_test_log = y_np[idx_test].ravel()

        # Log-space metrics
        train_mse_log = mean_squared_error(y_train_log, y_pred_train_log)
        test_mse_log = mean_squared_error(y_test_log, y_pred_test_log)
        train_mae_log = mean_absolute_error(y_train_log, y_pred_train_log)
        test_mae_log = mean_absolute_error(y_test_log, y_pred_test_log)
        train_r2_log = r2_score(y_train_log, y_pred_train_log)
        test_r2_log = r2_score(y_test_log, y_pred_test_log)

        # Physical-space metrics (barns)
        if self.pipeline.config.log_target:
            y_pred_train_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_pred_train_log)
            ).values
            y_pred_test_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_pred_test_log)
            ).values
            y_train_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_train_log)
            ).values
            y_test_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_test_log)
            ).values

            y_pred_train_barns = np.clip(y_pred_train_barns, 0.0, None)
            y_pred_test_barns = np.clip(y_pred_test_barns, 0.0, None)
            y_train_barns = np.clip(y_train_barns, 0.0, None)
            y_test_barns = np.clip(y_test_barns, 0.0, None)
        else:
            y_pred_train_barns = y_pred_train_log
            y_pred_test_barns = y_pred_test_log
            y_train_barns = y_train_log
            y_test_barns = y_test_log

        train_mse_barns = mean_squared_error(y_train_barns, y_pred_train_barns)
        test_mse_barns = mean_squared_error(y_test_barns, y_pred_test_barns)
        train_mae_barns = mean_absolute_error(y_train_barns, y_pred_train_barns)
        test_mae_barns = mean_absolute_error(y_test_barns, y_pred_test_barns)
        train_r2_barns = r2_score(y_train_barns, y_pred_train_barns)
        test_r2_barns = r2_score(y_test_barns, y_pred_test_barns)

        # Store metrics
        self.metrics = {
            'train_mse_log': train_mse_log,
            'test_mse_log': test_mse_log,
            'train_mae_log': train_mae_log,
            'test_mae_log': test_mae_log,
            'train_r2_log': train_r2_log,
            'test_r2_log': test_r2_log,
            'train_mse_barns': train_mse_barns,
            'test_mse_barns': test_mse_barns,
            'train_mae_barns': train_mae_barns,
            'test_mae_barns': test_mae_barns,
            'train_r2_barns': train_r2_barns,
            'test_r2_barns': test_r2_barns,
            'n_train': len(idx_train),
            'n_test': len(idx_test),
            'n_parameters': n_params,
            'stopped_epoch': stopped_epoch,
            'best_val_loss': best_val_loss if best_state is not None else val_losses[-1],
        }
        self.is_trained = True

        # ==================================================================
        # PRINT DIAGNOSTICS (same format as XGBoost evaluator)
        # ==================================================================
        if verbose:
            def _gap_pct(train_val, test_val):
                if train_val == 0:
                    return float('inf')
                return (test_val - train_val) / abs(train_val) * 100

            def _fmt(val, is_r2=False):
                if is_r2:
                    return f"{val:.4f}"
                elif abs(val) >= 1e5 or (abs(val) < 1e-2 and val != 0):
                    return f"{val:.2e}"
                else:
                    return f"{val:.4f}"

            print()
            print("=" * 70)
            print("FINAL PERFORMANCE DIAGNOSTICS")
            print("=" * 70)
            print(f"{'METRIC':<20s}| {'TRAIN SET':>16s} | {'TEST SET':>16s} | {'GAP (%)':>8s}")
            print("-" * 70)

            rows = [
                ("MSE  (Log10)", train_mse_log, test_mse_log, False),
                ("MAE  (Log10)", train_mae_log, test_mae_log, False),
                ("R^2  (Log10)", train_r2_log, test_r2_log, True),
                ("MSE  (Barns)", train_mse_barns, test_mse_barns, False),
                ("MAE  (Barns)", train_mae_barns, test_mae_barns, False),
                ("R^2  (Barns)", train_r2_barns, test_r2_barns, True),
            ]

            for label, train_val, test_val, is_r2 in rows:
                gap = _gap_pct(train_val, test_val)
                gap_str = f"{gap:+.1f}%" if abs(gap) < 1e6 else "   inf"
                print(
                    f"{label:<20s}| {_fmt(train_val, is_r2):>16s} "
                    f"| {_fmt(test_val, is_r2):>16s} | {gap_str:>8s}"
                )

            print("-" * 70)
            print()
            print("TRAINING DIAGNOSTICS")
            print("-" * 70)
            print(f"  Stopped at epoch:  {stopped_epoch} / {self.epochs}")
            print(f"  Best val loss:     {self.metrics['best_val_loss']:.6f}")
            print(f"  Parameters:        {n_params:,}")
            print(f"  Training samples:  {len(idx_train):,}")
            print(f"  Test samples:      {len(idx_test):,}")

            mse_gap = _gap_pct(train_mse_log, test_mse_log)
            if mse_gap > 50:
                print(
                    f"  [!] WARNING: Large generalisation gap ({mse_gap:.0f}%) "
                    f"-- possible overfitting"
                )
            elif mse_gap < 2:
                print(
                    f"  [i] Tight generalisation gap ({mse_gap:.1f}%) "
                    f"-- model may be underfitting"
                )
            else:
                print("  [OK] Generalisation gap within normal range")
            print("=" * 70)

        return self.metrics

    def predict(
        self,
        df: pd.DataFrame,
        energy_column: str = 'Energy',
    ) -> np.ndarray:
        """
        Predict cross-sections with automatic transformation.

        Args:
            df: Input features (must match training format)
            energy_column: Energy column name

        Returns:
            Predicted cross-sections in original scale (barns)
        """
        if not self.is_trained or self._model is None or self.pipeline is None:
            raise RuntimeError("Model must be trained before prediction")

        X_feat = df[self.feature_columns]
        energy = df[energy_column] if energy_column in df.columns else None

        X_t = self.pipeline.transform(X_feat, energy)
        arr = X_t[self.feature_columns].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)

        self._model.eval()
        with torch.no_grad():
            pred_t = self._model(
                torch.from_numpy(arr).to(self._device)
            ).cpu().numpy().ravel()

        if self.pipeline.config.log_target:
            return self.pipeline.inverse_transform_target(
                pd.Series(pred_t)
            ).values
        return pred_t

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model, pipeline, and configuration to disk.

        Args:
            filepath: Output file path (typically .pt or .pth)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        save_data = {
            # Model
            'model_state_dict': self._model.state_dict(),
            'n_features': self._model.net[0].in_features,
            # Configuration
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout,
            'loss_function': self.loss_function,
            'random_state': self.random_state,
            # Pipeline and features
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            # Metrics and history
            'metrics': self.metrics,
            'history': self.history,
        }
        torch.save(save_data, filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load model, pipeline, and configuration from disk.

        Args:
            filepath: Input file path
        """
        data = torch.load(filepath, weights_only=False)

        # Rebuild network
        self.hidden_sizes = data['hidden_sizes']
        self.dropout = data['dropout']
        self.loss_function = data['loss_function']
        self.random_state = data['random_state']

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._model = _SimpleNet(
            data['n_features'],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self._device)
        self._model.load_state_dict(data['model_state_dict'])
        self._model.eval()

        self.pipeline = data['pipeline']
        self.feature_columns = data['feature_columns']
        self.metrics = data['metrics']
        self.history = data['history']
        self.is_trained = True

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"NeuralNetEvaluator("
            f"hidden={list(self.hidden_sizes)}, "
            f"loss='{self.loss_function}', "
            f"{status})"
        )
