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
from typing import Tuple, Optional, Dict, Any, Callable

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
        smooth_mean_config: Configuration for smooth mean function.
            When ``smooth_mean_type='constant'`` (the default), behaviour
            is identical to the original constant-mean GP.  Set to
            ``'spline'`` for a data-driven consensus smooth mean.
        kernel_config: Configuration for the GP kernel.
            ``None`` (default) uses a standard RBF kernel, preserving
            exact pre-Phase-2 behaviour.  Set to a ``KernelConfig`` with
            ``kernel_type='gibbs'`` and an injected RIPL-3 interpolator
            for physics-informed nonstationary lengthscale.
            See ``nucml_next.data.kernels`` for details.
        likelihood_config: Configuration for the GP likelihood model.
            ``None`` (default) uses pure Gaussian noise, preserving
            exact pre-Phase-3 behaviour.  Set to a ``LikelihoodConfig``
            with ``likelihood_type='contaminated'`` for robust outlier
            identification via EM.
            See ``nucml_next.data.likelihood`` for details.
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
    max_gpu_points: int = 40000  # 40k points = 12.8 GB kernel matrix, fits in 16GB GPU
    max_subsample_points: int = 15000  # Subsample large experiments to fit GPU memory
    subsample_random_state: int = 42  # For reproducibility
    smooth_mean_config: Any = None  # SmoothMeanConfig, lazy to avoid circular import
    kernel_config: Any = None  # KernelConfig from kernels.py; None = RBF default
    likelihood_config: Any = None  # LikelihoodConfig; None = Gaussian (no EM)


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
        self._mean_value = None  # float or np.ndarray (array when smooth mean is used)
        self._mean_fn: Optional[Callable] = None  # mean function: log_E -> log_sigma
        self._kernel = None  # Kernel object (built in fit())

        # Cached Cholesky factor for fast predictions
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None  # K^{-1} @ (y - mean)

        # Contaminated normal EM results (Phase 3)
        self._outlier_probabilities: Optional[np.ndarray] = None
        self._noise_variance_effective: Optional[np.ndarray] = None

        # Transient: parameter bounds for hierarchical refit (Phase 4)
        # Set by refit_with_constraints(), cleared after use
        self._refit_param_bounds = None

        # Subsample tracking (set by fit() when subsampling occurs)
        self._subsample_indices: Optional[np.ndarray] = None

        # Energy range (for extrapolation detection)
        self._log_E_min: Optional[float] = None
        self._log_E_max: Optional[float] = None

    def fit(
        self,
        log_E: np.ndarray,
        log_sigma: np.ndarray,
        log_uncertainties: np.ndarray,
        mean_fn: Optional[Callable] = None,
    ) -> 'ExactGPExperiment':
        """
        Fit GP with heteroscedastic noise and calibrated lengthscale.

        Args:
            log_E: log10(Energy) values, shape (n,).
            log_sigma: log10(CrossSection) values, shape (n,).
            log_uncertainties: Noise std in log-space, shape (n,).
                Typically: 0.434 * (Uncertainty / CrossSection)
            mean_fn: Optional pre-computed mean function (e.g. from pooled
                group data).  ``mean_fn(log_E) -> log_sigma_mean``.
                If None, a mean function is computed from this experiment's
                data using ``config.smooth_mean_config``.

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

        # Store energy range for extrapolation detection (from full data)
        self._log_E_min = log_E.min()
        self._log_E_max = log_E.max()

        # Subsample large experiments to fit GPU memory
        # Fit on subsample, but predictions can be made on all points
        self._is_subsampled = False
        self._subsample_indices = None
        if n > self.config.max_subsample_points:
            rng = np.random.RandomState(self.config.subsample_random_state)
            indices = rng.choice(n, size=self.config.max_subsample_points, replace=False)
            indices = np.sort(indices)  # Keep energy ordering for stability

            self._subsample_indices = indices
            log_E = log_E[indices]
            log_sigma = log_sigma[indices]
            log_uncertainties = log_uncertainties[indices]
            self._is_subsampled = True
            logger.info(f"Subsampled {n} points to {len(log_E)} for GP fitting")

        # Store (possibly subsampled) training data
        self._train_x = log_E
        self._train_y = log_sigma
        self._noise_variance = log_uncertainties ** 2

        # Compute mean function: either from provided function, config, or constant
        if mean_fn is not None:
            self._mean_fn = mean_fn
        else:
            from nucml_next.data.smooth_mean import fit_smooth_mean, SmoothMeanConfig
            sm_config = self.config.smooth_mean_config
            if sm_config is None:
                sm_config = SmoothMeanConfig()  # constant mean (default)
            self._mean_fn = fit_smooth_mean(log_E, log_sigma, sm_config)

        # Evaluate mean at training points (scalar for constant, array for spline)
        self._mean_value = self._mean_fn(log_E)

        # Estimate initial hyperparameters
        data_range = log_E.max() - log_E.min()
        initial_lengthscale = data_range / 5  # Reasonable default

        # Estimate outputscale from residual variance minus noise
        residuals_for_var = log_sigma - self._mean_value
        signal_var = np.var(residuals_for_var) - np.mean(self._noise_variance)
        self._outputscale = max(signal_var, 0.1)

        # Build kernel object (outputscale injected, not optimised)
        from nucml_next.data.kernels import build_kernel, KernelConfig
        if self.config.kernel_config is not None:
            self._kernel = build_kernel(self.config.kernel_config)
        else:
            self._kernel = build_kernel(KernelConfig(
                lengthscale=initial_lengthscale,
            ))
        # When outputscale_fn is set, the kernel uses σ(xᵢ)·σ(xⱼ) instead of
        # the scalar outputscale.  Don't overwrite the fn's effect.
        # self._outputscale is still stored for diagnostics regardless.
        if (self.config.kernel_config is None
                or self.config.kernel_config.outputscale_fn is None):
            self._kernel.config.outputscale = self._outputscale

        # Optimize kernel parameters (lengthscale for RBF, or a₀/a₁ for Gibbs)
        if self.config.use_wasserstein_calibration:
            self._fit_with_wasserstein_calibration()
        else:
            self._fit_with_marginal_likelihood(initial_lengthscale)

        # Sync _lengthscale from kernel for backward-compatible diagnostics
        if hasattr(self._kernel.config, 'lengthscale'):
            self._lengthscale = self._kernel.config.lengthscale

        # Build cached quantities for fast prediction
        self._build_prediction_cache()

        # Run contaminated normal EM if configured (updates L, alpha in-place)
        self._run_contaminated_em()

        self.is_fitted = True

        # Store hyperparameters for diagnostics
        # mean: store scalar for JSON-serialisable diagnostics
        mean_for_diag = (
            float(np.mean(self._mean_value))
            if not np.isscalar(self._mean_value)
            else self._mean_value
        )
        self.hyperparameters = {
            'lengthscale': self._lengthscale,
            'outputscale': self._outputscale,
            'mean': mean_for_diag,
            'noise_mean': np.mean(self._noise_variance) ** 0.5,
            'n_points': n,
        }
        # Add kernel-specific params for diagnostics
        self.hyperparameters.update(self._kernel.get_all_params())
        # Add contaminated EM diagnostics if available
        if self._outlier_probabilities is not None:
            self.hyperparameters['n_outliers'] = int(
                np.sum(self._outlier_probabilities > 0.5)
            )
            self.hyperparameters['max_outlier_prob'] = float(
                np.max(self._outlier_probabilities)
            )

        return self

    def _fit_with_wasserstein_calibration(self) -> None:
        """Optimize kernel parameters via Wasserstein calibration distance.

        Delegates to ``optimize_kernel_wasserstein`` (NumPy) or
        ``optimize_kernel_wasserstein_torch`` (GPU), which dispatches
        automatically:
        - RBF (1 param): Grid + Brent search over lengthscale
        - Gibbs (2 params): Grid-initialised Nelder-Mead over (a₀, a₁)

        **Data-driven Gibbs short-circuit:** When the kernel has a
        ``data_lengthscale_interpolator`` (from ``compute_lengthscale_from_residuals``),
        the base profile already captures the physics.  Optimising (a₀, a₁)
        destroys the profile (e.g. a₀≈-3, a₁≈-1.4 collapses ℓ to 10⁻⁶).
        Instead, fix a₀=a₁=0 and compute Wasserstein metric once.

        Uses _effective_device which may be downgraded from config.device
        for large experiments.

        When ``self._refit_param_bounds`` is set (by ``refit_with_constraints``),
        passes the bounds through to the optimiser for constrained search.
        """
        # Data-driven Gibbs: skip (a₀, a₁) optimisation — profile IS the lengthscale
        if (hasattr(self._kernel, 'config')
                and getattr(self._kernel.config, 'data_lengthscale_interpolator', None) is not None):
            self._kernel.set_optimizable_params(
                np.zeros(self._kernel.n_optimizable_params())
            )
            # Compute Wasserstein metric once (for diagnostics / hierarchical stats)
            from nucml_next.data.calibration import (
                compute_loo_z_scores_kernel,
                compute_wasserstein_calibration,
            )
            z = compute_loo_z_scores_kernel(
                self._train_x, self._train_y, self._noise_variance,
                self._kernel, self._mean_value,
            )
            self.calibration_metric = compute_wasserstein_calibration(z)
            # _lengthscale stores a₀=0.0 here — misleading for Gibbs (ℓ varies
            # by energy) but harmless; only used for diagnostics dict.
            self._lengthscale = self._kernel.get_all_params().get(
                'lengthscale', self._kernel.get_optimizable_params()[0]
            )
            logger.debug(
                f"Data-driven Gibbs: a₀=a₁=0, W={self.calibration_metric:.4f}"
            )
            return

        bounds_override = getattr(self, '_refit_param_bounds', None)

        if self._effective_device != 'cpu':
            from nucml_next.data.calibration import optimize_kernel_wasserstein_torch

            self._kernel, self.calibration_metric = optimize_kernel_wasserstein_torch(
                self._train_x,
                self._train_y,
                self._noise_variance,
                kernel=self._kernel,
                mean_value=self._mean_value,
                lengthscale_bounds=self.config.lengthscale_bounds,
                device=self._effective_device,
                param_bounds=bounds_override,
            )
        else:
            from nucml_next.data.calibration import optimize_kernel_wasserstein

            self._kernel, self.calibration_metric = optimize_kernel_wasserstein(
                self._train_x,
                self._train_y,
                self._noise_variance,
                kernel=self._kernel,
                mean_value=self._mean_value,
                lengthscale_bounds=self.config.lengthscale_bounds,
                param_bounds=bounds_override,
            )

        # Sync _lengthscale for backward compatibility
        self._lengthscale = self._kernel.get_all_params().get(
            'lengthscale', self._kernel.get_optimizable_params()[0]
        )

        logger.debug(
            f"Wasserstein calibration: params={self._kernel.get_all_params()}, "
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
        lengthscale: float = None,
        outputscale: float = None,
        x2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute kernel matrix.

        When a kernel object is available (``self._kernel``), delegates to
        ``kernel.compute_matrix()``.  Falls back to inline RBF for the
        marginal-likelihood path (which passes explicit lengthscale/outputscale).

        Args:
            x1: First input array.
            lengthscale: Explicit lengthscale (used by MLL path only).
                If None, uses the kernel object.
            outputscale: Explicit outputscale (used by MLL path only).
                If None, uses the kernel object.
            x2: Second input array. If None, uses x1.

        Returns:
            Kernel matrix of shape (len(x1), len(x2)).
        """
        # If explicit parameters are given (MLL path), use inline RBF
        # to avoid mutating the kernel object during optimisation.
        if lengthscale is not None and outputscale is not None:
            if x2 is None:
                x2 = x1
            x1 = np.asarray(x1).ravel()
            x2 = np.asarray(x2).ravel()
            diff = x1[:, None] - x2[None, :]
            return outputscale * np.exp(-0.5 * diff ** 2 / lengthscale ** 2)

        # Delegate to kernel object
        return self._kernel.compute_matrix(x1, x2)

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
        """Build prediction cache using NumPy (CPU).

        Uses the kernel object to build the covariance matrix.
        """
        K = self._kernel.compute_matrix(self._train_x, self._train_x)
        K += np.diag(self._noise_variance)
        K += np.eye(len(self._train_x)) * 1e-6

        self._L = np.linalg.cholesky(K)
        residuals = self._train_y - self._mean_value
        self._alpha = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, residuals)
        )

    def _build_prediction_cache_torch(self) -> None:
        """Build prediction cache using PyTorch (GPU-accelerated).

        Uses the kernel object's PyTorch path to build the covariance matrix.
        """
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

        # Build kernel matrix via kernel object
        K = self._kernel.compute_matrix_torch(x, x, device=device)
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

    def _run_contaminated_em(self) -> None:
        """Run contaminated normal EM after initial Cholesky build.

        When ``likelihood_config`` is set to ``'contaminated'``, this
        method recomputes the kernel matrix, runs the EM algorithm to
        identify outlier points, and updates ``self._L`` and
        ``self._alpha`` in-place with the EM-refined effective noise.

        When ``likelihood_config`` is None or ``'gaussian'``, this is
        a no-op.
        """
        lc = self.config.likelihood_config
        if lc is None or lc.likelihood_type != 'contaminated':
            return

        from nucml_next.data.likelihood import (
            run_contaminated_em,
            run_contaminated_em_torch,
        )

        if self._effective_device != 'cpu':
            import torch

            device = self._effective_device
            x_t = torch.tensor(
                self._train_x, dtype=torch.float64, device=device
            )
            K_kernel = self._kernel.compute_matrix_torch(
                x_t, x_t, device=device
            )
            sigma_eff_sq, outlier_prob, L = run_contaminated_em_torch(
                K_kernel, self._train_y, self._noise_variance,
                self._mean_value, lc,
            )
        else:
            K_kernel = self._kernel.compute_matrix(
                self._train_x, self._train_x
            )
            sigma_eff_sq, outlier_prob, L = run_contaminated_em(
                K_kernel, self._train_y, self._noise_variance,
                self._mean_value, lc,
            )

        # Update cached Cholesky and alpha with EM-refined noise
        self._L = L
        residuals = self._train_y - np.asarray(self._mean_value)
        self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, residuals))

        self._outlier_probabilities = outlier_prob
        self._noise_variance_effective = sigma_eff_sq

        n_outliers = int(np.sum(outlier_prob > 0.5))
        logger.debug(
            f"Contaminated EM: {n_outliers}/{len(outlier_prob)} points "
            f"flagged (prob > 0.5), max_prob={np.max(outlier_prob):.4f}"
        )

    @property
    def outlier_probabilities(self) -> Optional[np.ndarray]:
        """Per-point outlier probabilities from contaminated normal EM.

        Returns None if contaminated likelihood was not used.
        """
        return self._outlier_probabilities

    @property
    def subsample_indices(self) -> Optional[np.ndarray]:
        """Indices into the original input arrays selected for training.

        Returns None if no subsampling occurred (all points were used).
        When not None, ``len(subsample_indices) == len(outlier_probabilities)``.
        """
        return self._subsample_indices

    def refit_with_constraints(
        self,
        outputscale: Optional[float] = None,
        param_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> 'ExactGPExperiment':
        """Re-fit kernel parameters with constrained bounds (Phase 4).

        Used by the hierarchical refitting pass to re-optimise each experiment's
        kernel parameters within a group-informed feasible region while
        optionally sharing the group median outputscale.

        Skips data preparation (input validation, subsampling, mean function
        estimation — all preserved from Pass 1) and only re-runs: kernel
        parameter optimisation → Cholesky factorisation → optional EM.

        Args:
            outputscale: If not None, override the outputscale before re-fitting.
                Typically set to the group median outputscale.
            param_bounds: If not None, a tuple ``(lower, upper)`` of 1-D arrays
                with shape ``(n_optimizable_params,)``.  The kernel optimiser
                will constrain its search to these bounds.

        Returns:
            self (for chaining).

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Must call fit() before refit_with_constraints()"
            )

        # Update outputscale if requested (shared group outputscale)
        if outputscale is not None:
            self._outputscale = outputscale
            self._kernel.config.outputscale = outputscale

        # Set transient bounds for the optimiser
        self._refit_param_bounds = param_bounds
        try:
            # Re-optimise kernel parameters within constrained bounds
            self._fit_with_wasserstein_calibration()
        finally:
            # Always clear transient bounds
            self._refit_param_bounds = None

        # Sync _lengthscale from kernel for backward-compatible diagnostics
        if hasattr(self._kernel.config, 'lengthscale'):
            self._lengthscale = self._kernel.config.lengthscale

        # Rebuild cached Cholesky factor and alpha
        self._build_prediction_cache()

        # Re-run contaminated normal EM if configured
        self._run_contaminated_em()

        # Rebuild hyperparameters dict (replicating fit() pattern)
        mean_for_diag = (
            float(np.mean(self._mean_value))
            if not np.isscalar(self._mean_value)
            else self._mean_value
        )
        n = len(self._train_x)
        self.hyperparameters = {
            'lengthscale': self._lengthscale,
            'outputscale': self._outputscale,
            'mean': mean_for_diag,
            'noise_mean': np.mean(self._noise_variance) ** 0.5,
            'n_points': n,
        }
        self.hyperparameters.update(self._kernel.get_all_params())
        if self._outlier_probabilities is not None:
            self.hyperparameters['n_outliers'] = int(
                np.sum(self._outlier_probabilities > 0.5)
            )
            self.hyperparameters['max_outlier_prob'] = float(
                np.max(self._outlier_probabilities)
            )

        return self

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
        K_star = self._kernel.compute_matrix(log_E_query, self._train_x)

        # Mean prediction: mu(query) + K_* @ alpha
        query_mean = self._mean_fn(log_E_query)
        mean = query_mean + K_star @ self._alpha

        # Variance prediction: K_** - K_* @ K^{-1} @ K_*^T
        # K_** is the prior variance at query points (diagonal only)
        # Returns array when data_outputscale_interpolator is set, scalar otherwise
        K_star_star = self._kernel.prior_variance(log_E_query)

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

        mean = np.asarray(self._mean_value)
        if mean.ndim == 0:
            mean = np.full(len(self._train_y), float(mean))
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
