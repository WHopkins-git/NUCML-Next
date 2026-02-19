"""
Per-Experiment Exact GP Fitting
===============================

Fits an Exact Gaussian Process to a single EXFOR experiment's data using
heteroscedastic noise from measurement uncertainties.

Key Classes:
    ExactGPExperimentConfig: Configuration dataclass
    ExactGPExperiment: Per-experiment GP fitter

Usage:
    >>> from nucml_next.data.experiment_gp import ExactGPExperiment, ExactGPExperimentConfig
    >>> config = ExactGPExperimentConfig(use_wasserstein_calibration=True)
    >>> gp = ExactGPExperiment(config)
    >>> gp.fit(log_E, log_sigma, log_uncertainties)
    >>> mean, std = gp.predict(log_E_query)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExactGPExperimentConfig:
    """Configuration for per-experiment Exact GP fitting.

    Attributes:
        min_points_for_gp: Minimum points to fit GP (below: cannot fit).
        max_epochs: Maximum epochs for marginal likelihood optimization.
        lr: Learning rate for hyperparameter optimization.
        convergence_tol: Early stopping tolerance.
        patience: Epochs without improvement before stopping.
        default_rel_uncertainty: Default relative uncertainty (10%) for
            points missing uncertainty values.
        use_wasserstein_calibration: If True, optimize lengthscale via
            Wasserstein calibration. If False, use marginal likelihood.
        lengthscale_bounds: (min, max) bounds for lengthscale search.
        device: PyTorch device ('cpu' or 'cuda').
        max_gpu_points: Experiments with more points than this are
            automatically routed to CPU to avoid GPU OOM errors.
            For n points, kernel matrix uses n²×8 bytes (float64):
            - 2000 pts → 32 MB (safe for most GPUs)
            - 5000 pts → 200 MB
            - 10000 pts → 800 MB
    """
    min_points_for_gp: int = 5
    max_epochs: int = 200
    lr: float = 0.1
    convergence_tol: float = 1e-4
    patience: int = 20
    default_rel_uncertainty: float = 0.10
    use_wasserstein_calibration: bool = True
    lengthscale_bounds: Tuple[float, float] = (0.01, 10.0)
    device: str = 'cpu'
    max_gpu_points: int = 2000


class ExactGPExperiment:
    """Fit an Exact GP to a single EXFOR experiment's data.

    Uses heteroscedastic noise from measurement uncertainties and
    optionally calibrates lengthscale via Wasserstein distance.

    Args:
        config: Configuration for GP fitting.

    Example:
        >>> gp = ExactGPExperiment(ExactGPExperimentConfig())
        >>> gp.fit(log_E, log_sigma, log_uncertainties)
        >>> mean, std = gp.predict(log_E_query)
        >>> z_scores = gp.get_point_z_scores(log_E, log_sigma)
    """

    def __init__(self, config: ExactGPExperimentConfig = None):
        if config is None:
            config = ExactGPExperimentConfig()
        self.config = config

        # Will be set after fitting
        self.is_fitted = False
        self.calibration_metric = None
        self.hyperparameters: Dict[str, Any] = {}

        # Training data (stored for predictions)
        self._train_x: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None
        self._noise_variance: Optional[np.ndarray] = None

        # Fitted GP parameters
        self._lengthscale: Optional[float] = None
        self._outputscale: Optional[float] = None
        self._mean_value: Optional[float] = None

        # Cached Cholesky factor for fast predictions
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None  # K^{-1} @ (y - mean)

        # Energy range (for extrapolation detection)
        self._log_E_min: Optional[float] = None
        self._log_E_max: Optional[float] = None

    def fit(
        self,
        log_E: np.ndarray,
        log_sigma: np.ndarray,
        log_uncertainties: np.ndarray,
    ) -> 'ExactGPExperiment':
        """
        Fit GP with heteroscedastic noise and calibrated lengthscale.

        Args:
            log_E: log10(Energy) values, shape (n,).
            log_sigma: log10(CrossSection) values, shape (n,).
            log_uncertainties: Noise std in log-space, shape (n,).
                Typically: 0.434 * (Uncertainty / CrossSection)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If fewer than min_points_for_gp points provided.
        """
        log_E = np.asarray(log_E).ravel()
        log_sigma = np.asarray(log_sigma).ravel()
        log_uncertainties = np.asarray(log_uncertainties).ravel()

        n = len(log_E)
        if n < self.config.min_points_for_gp:
            raise ValueError(
                f"Need >= {self.config.min_points_for_gp} points to fit GP, got {n}"
            )

        # Proactive GPU memory management: route large experiments to CPU
        self._effective_device = self.config.device
        if (self._effective_device == 'cuda' and
                n > self.config.max_gpu_points):
            logger.info(
                f"Experiment with {n} points exceeds max_gpu_points "
                f"({self.config.max_gpu_points}), using CPU"
            )
            self._effective_device = 'cpu'

        # Store training data
        self._train_x = log_E
        self._train_y = log_sigma
        self._noise_variance = log_uncertainties ** 2

        # Store energy range for extrapolation detection
        self._log_E_min = log_E.min()
        self._log_E_max = log_E.max()

        # Estimate initial hyperparameters
        self._mean_value = np.mean(log_sigma)
        data_range = log_E.max() - log_E.min()
        initial_lengthscale = data_range / 5  # Reasonable default

        # Estimate outputscale from data variance minus noise
        signal_var = np.var(log_sigma) - np.mean(self._noise_variance)
        self._outputscale = max(signal_var, 0.1)

        # Optimize lengthscale
        if self.config.use_wasserstein_calibration:
            self._fit_with_wasserstein_calibration()
        else:
            self._fit_with_marginal_likelihood(initial_lengthscale)

        # Build cached quantities for fast prediction
        self._build_prediction_cache()

        self.is_fitted = True

        # Store hyperparameters for diagnostics
        self.hyperparameters = {
            'lengthscale': self._lengthscale,
            'outputscale': self._outputscale,
            'mean': self._mean_value,
            'noise_mean': np.mean(self._noise_variance) ** 0.5,
            'n_points': n,
        }

        return self

    def _fit_with_wasserstein_calibration(self) -> None:
        """Optimize lengthscale via Wasserstein calibration distance.

        Uses PyTorch-accelerated version when device is 'cuda' for GPU speedup.
        Uses _effective_device which may be downgraded from config.device
        for large experiments.
        """
        # Use PyTorch-accelerated version for GPU
        if self._effective_device != 'cpu':
            from nucml_next.data.calibration import optimize_lengthscale_wasserstein_torch

            self._lengthscale, self.calibration_metric = optimize_lengthscale_wasserstein_torch(
                self._train_x,
                self._train_y,
                self._noise_variance,
                lengthscale_bounds=self.config.lengthscale_bounds,
                outputscale=self._outputscale,
                mean_value=self._mean_value,
                device=self._effective_device,
            )
        else:
            # CPU: use NumPy version (slightly faster for small matrices)
            from nucml_next.data.calibration import optimize_lengthscale_wasserstein

            self._lengthscale, self.calibration_metric = optimize_lengthscale_wasserstein(
                self._train_x,
                self._train_y,
                self._noise_variance,
                lengthscale_bounds=self.config.lengthscale_bounds,
                outputscale=self._outputscale,
                mean_value=self._mean_value,
            )

        logger.debug(
            f"Wasserstein calibration: ls={self._lengthscale:.4f}, "
            f"W={self.calibration_metric:.4f}"
        )

    def _fit_with_marginal_likelihood(self, initial_lengthscale: float) -> None:
        """Optimize lengthscale via marginal likelihood (standard approach)."""
        from scipy.optimize import minimize_scalar

        def neg_log_marginal_likelihood(ls: float) -> float:
            if ls <= 0:
                return np.inf

            K = self._compute_kernel_matrix(self._train_x, ls, self._outputscale)
            K += np.diag(self._noise_variance)
            K += np.eye(len(self._train_x)) * 1e-6

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return np.inf

            residuals = self._train_y - self._mean_value
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, residuals))

            # Log marginal likelihood
            n = len(self._train_x)
            nll = 0.5 * np.dot(residuals, alpha)
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * n * np.log(2 * np.pi)

            return nll

        result = minimize_scalar(
            neg_log_marginal_likelihood,
            bounds=self.config.lengthscale_bounds,
            method='bounded',
        )

        self._lengthscale = result.x
        self.calibration_metric = None  # Not computed for MLL

    def _compute_kernel_matrix(
        self,
        x1: np.ndarray,
        lengthscale: float,
        outputscale: float,
        x2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute RBF kernel matrix."""
        if x2 is None:
            x2 = x1

        x1 = np.asarray(x1).ravel()
        x2 = np.asarray(x2).ravel()

        diff = x1[:, None] - x2[None, :]
        K = outputscale * np.exp(-0.5 * diff ** 2 / lengthscale ** 2)

        return K

    def _build_prediction_cache(self) -> None:
        """Build cached quantities for fast prediction.

        Uses PyTorch when device is CUDA for GPU acceleration.
        Uses _effective_device which may be downgraded for large experiments.
        """
        if self._effective_device != 'cpu':
            self._build_prediction_cache_torch()
        else:
            self._build_prediction_cache_numpy()

    def _build_prediction_cache_numpy(self) -> None:
        """Build prediction cache using NumPy (CPU)."""
        K = self._compute_kernel_matrix(
            self._train_x, self._lengthscale, self._outputscale
        )
        K += np.diag(self._noise_variance)
        K += np.eye(len(self._train_x)) * 1e-6

        self._L = np.linalg.cholesky(K)
        residuals = self._train_y - self._mean_value
        self._alpha = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, residuals)
        )

    def _build_prediction_cache_torch(self) -> None:
        """Build prediction cache using PyTorch (GPU-accelerated)."""
        import torch

        device = self._effective_device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for prediction cache")
            self._build_prediction_cache_numpy()
            return

        torch_device = torch.device(device)
        n = len(self._train_x)

        # Convert to tensors
        x = torch.tensor(self._train_x, dtype=torch.float64, device=torch_device)
        noise_var = torch.tensor(self._noise_variance, dtype=torch.float64, device=torch_device)

        # Build kernel matrix
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        K = self._outputscale * torch.exp(-0.5 * diff.pow(2) / (self._lengthscale ** 2))
        K = K + torch.diag(noise_var) + 1e-6 * torch.eye(n, dtype=torch.float64, device=torch_device)

        # Cholesky decomposition
        L = torch.linalg.cholesky(K)

        # Compute alpha = L^{-T} @ L^{-1} @ residuals
        residuals = torch.tensor(
            self._train_y - self._mean_value, dtype=torch.float64, device=torch_device
        )
        z = torch.linalg.solve_triangular(L, residuals.unsqueeze(1), upper=False).squeeze(1)
        alpha = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze(1)

        # Store back as NumPy arrays (predictions will be on CPU)
        self._L = L.cpu().numpy()
        self._alpha = alpha.cpu().numpy()

    def predict(
        self,
        log_E_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at query points.

        Args:
            log_E_query: Query points in log10(Energy) space.

        Returns:
            (mean, std) arrays of shape (n_query,).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")

        log_E_query = np.asarray(log_E_query).ravel()

        # Cross-covariance between query and training points
        K_star = self._compute_kernel_matrix(
            log_E_query, self._lengthscale, self._outputscale, self._train_x
        )

        # Mean prediction: mu + K_* @ alpha
        mean = self._mean_value + K_star @ self._alpha

        # Variance prediction: K_** - K_* @ K^{-1} @ K_*^T
        # K_** is the prior variance at query points (diagonal only)
        K_star_star = self._outputscale  # Diagonal of prior covariance

        # v = L^{-1} @ K_*^T
        v = np.linalg.solve(self._L, K_star.T)
        var = K_star_star - np.sum(v ** 2, axis=0)

        # Ensure non-negative variance
        var = np.clip(var, 1e-10, None)
        std = np.sqrt(var)

        return mean, std

    def get_point_z_scores(
        self,
        log_E: np.ndarray,
        log_sigma: np.ndarray,
    ) -> np.ndarray:
        """
        Compute z-scores for points against this experiment's GP.

        Args:
            log_E: log10(Energy) values.
            log_sigma: log10(CrossSection) values.

        Returns:
            Absolute z-scores: |log_sigma - mean| / std
        """
        mean, std = self.predict(log_E)
        return np.abs(log_sigma - mean) / std

    def get_loo_z_scores(self) -> np.ndarray:
        """
        Compute leave-one-out z-scores for training points.

        Uses efficient computation from cached Cholesky factor.

        Returns:
            LOO z-scores for training points.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_loo_z_scores()")

        from nucml_next.data.calibration import compute_loo_z_scores_from_cholesky

        mean = np.full(len(self._train_y), self._mean_value)
        return compute_loo_z_scores_from_cholesky(
            self._L, self._train_y, mean
        )

    def is_interpolating(self, log_E: np.ndarray, margin: float = 0.1) -> np.ndarray:
        """
        Check if query points are within interpolation range.

        Args:
            log_E: Query points in log10(Energy) space.
            margin: Margin beyond training range (in log10 units).

        Returns:
            Boolean array: True if interpolating, False if extrapolating.
        """
        if self._log_E_min is None or self._log_E_max is None:
            raise RuntimeError("Must call fit() first")

        log_E = np.asarray(log_E).ravel()
        return (log_E >= self._log_E_min - margin) & (log_E <= self._log_E_max + margin)

    @property
    def energy_range(self) -> Tuple[float, float]:
        """Return (log_E_min, log_E_max) of training data."""
        if self._log_E_min is None:
            raise RuntimeError("Must call fit() first")
        return (self._log_E_min, self._log_E_max)


def prepare_log_uncertainties(
    uncertainties: np.ndarray,
    cross_sections: np.ndarray,
    default_rel_uncertainty: float = 0.10,
) -> np.ndarray:
    """
    Convert measurement uncertainties to log-space standard deviations.

    Transform: sigma_log = 0.434 * (Uncertainty / CrossSection)

    Missing or invalid uncertainties are filled with the median valid
    relative uncertainty, or the default if none are valid.

    Args:
        uncertainties: Cross-section uncertainties in barns.
        cross_sections: Cross-section values in barns.
        default_rel_uncertainty: Default relative uncertainty (e.g., 0.10 = 10%).

    Returns:
        Log-space standard deviations, shape same as input.
    """
    uncertainties = np.asarray(uncertainties)
    cross_sections = np.asarray(cross_sections)

    # Identify valid uncertainties
    valid_mask = (
        (uncertainties > 0) &
        (cross_sections > 0) &
        np.isfinite(uncertainties) &
        np.isfinite(cross_sections)
    )

    # Compute relative uncertainties
    rel_unc = np.full_like(cross_sections, np.nan, dtype=float)
    rel_unc[valid_mask] = uncertainties[valid_mask] / cross_sections[valid_mask]

    # Fill missing with median (not max - less conservative)
    if valid_mask.sum() > 0:
        median_rel = np.nanmedian(rel_unc)
        rel_unc = np.where(np.isnan(rel_unc), median_rel, rel_unc)
    else:
        rel_unc = np.full_like(cross_sections, default_rel_uncertainty, dtype=float)

    # Clamp to reasonable range [1%, 100%]
    rel_unc = np.clip(rel_unc, 0.01, 1.0)

    # Convert to log10-space: sigma_log10 = 0.434 * rel_unc
    # (0.434 = 1 / ln(10) from error propagation of log10)
    log_uncertainties = 0.434 * rel_unc

    return log_uncertainties
