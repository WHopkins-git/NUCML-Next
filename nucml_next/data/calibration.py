"""
Wasserstein Calibration for GP Lengthscale Optimization
========================================================

Implements Wasserstein distance-based calibration for Gaussian Process
lengthscale optimization. A well-calibrated GP should have z-scores that
follow a standard normal distribution.

Key Functions:
    compute_wasserstein_calibration: Compare z-scores to half-normal
    compute_loo_z_scores: Efficient leave-one-out z-scores via Cholesky
    optimize_lengthscale_wasserstein: Find calibrated lengthscale

Kernel-Aware Functions (Phase 2):
    compute_loo_z_scores_kernel: LOO z-scores using a Kernel object
    compute_loo_z_scores_kernel_torch: PyTorch version
    optimize_kernel_wasserstein: Optimise any kernel's params via Wasserstein

Usage:
    >>> from nucml_next.data.calibration import optimize_lengthscale_wasserstein
    >>> optimal_ls, wasserstein_dist = optimize_lengthscale_wasserstein(
    ...     log_E, log_sigma, noise_variance
    ... )
"""

import logging
from typing import Tuple, Optional, Callable, Union, TYPE_CHECKING

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize_scalar, minimize

if TYPE_CHECKING:
    from nucml_next.data.kernels import Kernel

logger = logging.getLogger(__name__)


def compute_wasserstein_calibration(
    z_scores: np.ndarray,
    n_theoretical_samples: int = 10000,
    random_state: int = 42,
) -> float:
    """
    Compute Wasserstein distance between |z-scores| and half-normal.

    For a well-calibrated GP, the absolute z-scores should follow a
    half-normal (folded standard normal) distribution. Lower Wasserstein
    distance indicates better calibration.

    Args:
        z_scores: Empirical z-scores from GP, shape (n,). Can be signed
            (will take absolute value) or already absolute.
        n_theoretical_samples: Number of samples from half-normal for
            comparison. More samples = more accurate but slower.
        random_state: Random seed for reproducibility.

    Returns:
        Wasserstein-1 distance (lower = better calibration).
        Returns np.inf if insufficient valid z-scores.

    Example:
        >>> z_scores = np.random.standard_normal(1000)
        >>> w = compute_wasserstein_calibration(z_scores)
        >>> print(f"Wasserstein distance: {w:.4f}")  # Should be close to 0
    """
    # Filter invalid values
    valid_z = z_scores[np.isfinite(z_scores)]
    if len(valid_z) < 3:
        return np.inf

    # Use absolute z-scores (we compute |residual| / std)
    abs_z = np.abs(valid_z)

    # Theoretical: folded standard normal (half-normal)
    # For |Z| where Z ~ N(0,1), the distribution is half-normal
    rng = np.random.default_rng(random_state)
    theoretical_samples = np.abs(rng.standard_normal(n_theoretical_samples))

    # Compute Wasserstein-1 distance
    return wasserstein_distance(abs_z, theoretical_samples)


def compute_loo_z_scores_from_cholesky(
    L: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """
    Compute leave-one-out z-scores efficiently from Cholesky decomposition.

    For an exact GP, LOO predictions can be computed analytically from the
    inverse covariance matrix without refitting N times. This uses the
    Sherman-Morrison-Woodbury formula.

    For point i:
        LOO mean: mu_{-i}(x_i) = y_i - (K^{-1} r)_i / (K^{-1})_{ii}
        LOO var:  var_{-i}(x_i) = 1 / (K^{-1})_{ii}
        LOO z:    z_i = (y_i - mu_{-i}) / sqrt(var_{-i})
                      = (K^{-1} r)_i / sqrt((K^{-1})_{ii})

    where r = y - mean is the residual vector.

    Args:
        L: Lower Cholesky factor of covariance matrix K, shape (n, n).
            K = L @ L.T
        y: Observed values, shape (n,).
        mean: GP mean predictions at training points, shape (n,).

    Returns:
        LOO z-scores, shape (n,).

    Complexity:
        O(N^2) for solving and computing diagonal, given Cholesky.
        The Cholesky decomposition itself is O(N^3).
    """
    n = len(y)
    residuals = y - mean

    # Compute K^{-1} @ residuals efficiently via Cholesky
    # K^{-1} r = L^{-T} @ L^{-1} @ r
    # First solve L @ z = r for z
    z = np.linalg.solve(L, residuals)
    # Then solve L.T @ x = z for x = K^{-1} r
    K_inv_r = np.linalg.solve(L.T, z)

    # Compute diagonal of K^{-1} efficiently
    # K^{-1} = L^{-T} @ L^{-1}
    # (K^{-1})_{ii} = sum_j (L^{-1})_{ji}^2
    # L^{-1} can be computed column by column
    L_inv = np.linalg.solve(L, np.eye(n))
    K_inv_diag = np.sum(L_inv ** 2, axis=0)

    # LOO z-scores
    loo_var = 1.0 / np.clip(K_inv_diag, 1e-10, None)
    z_scores = K_inv_r / np.sqrt(np.clip(K_inv_diag, 1e-10, None))

    return z_scores


def compute_loo_z_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    lengthscale: float,
    outputscale: float = 1.0,
    mean_value: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """
    Compute LOO z-scores for given GP hyperparameters.

    This is a pure NumPy implementation that doesn't require GPyTorch,
    useful for fast lengthscale optimization.

    Args:
        train_x: Training inputs, shape (n,) or (n, 1).
        train_y: Training targets, shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        lengthscale: RBF kernel lengthscale.
        outputscale: RBF kernel outputscale (variance).
        mean_value: Mean value(s). Scalar or array of shape (n,).
            If None, uses mean(train_y).

    Returns:
        LOO z-scores, shape (n,).
    """
    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()
    n = len(train_x)

    if mean_value is None:
        mean_value = np.mean(train_y)

    # Build RBF kernel matrix
    # K(x, x') = outputscale * exp(-0.5 * ||x - x'||^2 / lengthscale^2)
    diff = train_x[:, None] - train_x[None, :]
    K = outputscale * np.exp(-0.5 * diff ** 2 / lengthscale ** 2)

    # Add noise to diagonal
    K += np.diag(noise_variance)

    # Add small jitter for numerical stability
    K += np.eye(n) * 1e-6

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Matrix not positive definite, return infinite z-scores
        logger.warning("Cholesky failed in LOO computation")
        return np.full(n, np.inf)

    # Compute LOO z-scores
    mean = np.full(n, mean_value) if np.isscalar(mean_value) else np.asarray(mean_value)
    return compute_loo_z_scores_from_cholesky(L, train_y, mean)


def compute_loo_z_scores_kernel(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """
    Compute LOO z-scores using a Kernel object.

    Identical algorithm to ``compute_loo_z_scores`` but builds the kernel
    matrix via ``kernel.compute_matrix()`` instead of hardcoded RBF.
    This enables any kernel (RBF, Gibbs, etc.) to be used with the
    Wasserstein calibration framework.

    Args:
        train_x: Training inputs (log₁₀(E [eV])), shape (n,) or (n, 1).
        train_y: Training targets (log₁₀(σ [b])), shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        kernel: Kernel object implementing ``compute_matrix(x1, x2)``.
        mean_value: Mean value(s). Scalar or array of shape (n,).
            If None, uses mean(train_y).

    Returns:
        LOO z-scores, shape (n,).
    """
    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()
    n = len(train_x)

    if mean_value is None:
        mean_value = np.mean(train_y)

    # Build kernel matrix via the kernel object
    K = kernel.compute_matrix(train_x, train_x)

    # Add noise to diagonal
    K += np.diag(noise_variance)

    # Add small jitter for numerical stability
    K += np.eye(n) * 1e-6

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        logger.warning("Cholesky failed in LOO computation (kernel)")
        return np.full(n, np.inf)

    # Compute LOO z-scores
    mean = np.full(n, mean_value) if np.isscalar(mean_value) else np.asarray(mean_value)
    return compute_loo_z_scores_from_cholesky(L, train_y, mean)


def _wasserstein_loss_for_kernel(
    params: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Union[float, np.ndarray],
) -> float:
    """
    Compute Wasserstein calibration loss for given kernel parameters.

    Used as objective function for kernel parameter optimization.
    Sets the kernel's optimisable parameters, computes LOO z-scores,
    then returns the Wasserstein distance.
    """
    kernel.set_optimizable_params(params)

    z_scores = compute_loo_z_scores_kernel(
        train_x, train_y, noise_variance,
        kernel=kernel,
        mean_value=mean_value,
    )

    return compute_wasserstein_calibration(z_scores)


def optimize_kernel_wasserstein(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Optional[Union[float, np.ndarray]] = None,
    lengthscale_bounds: Tuple[float, float] = (0.01, 10.0),
    n_grid: int = 20,
    param_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple['Kernel', float]:
    """
    Optimise a kernel's parameters via Wasserstein calibration.

    Dispatches by ``kernel.n_optimizable_params()``:

    - **1 parameter** (RBF: lengthscale): Grid search + Brent refinement.
      Same algorithm as ``optimize_lengthscale_wasserstein``.
    - **2+ parameters** (Gibbs: a₀, a₁): Grid-initialised Nelder-Mead.

    The kernel's ``outputscale`` is NOT optimised — it must be set
    beforehand (estimated from ``Var(residuals) - mean(noise)``).

    Args:
        train_x: Training inputs (log₁₀(E [eV])), shape (n,).
        train_y: Training targets (log₁₀(σ [b])), shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        kernel: Kernel object.  Its optimisable parameters will be
            modified in-place to the optimal values.
        mean_value: Mean value(s). If None, uses mean(train_y).
        lengthscale_bounds: Search bounds for RBF lengthscale (1-param
            case only).  Ignored for multi-parameter kernels.
        n_grid: Number of grid points for initial search.
        param_bounds: Optional ``(lower, upper)`` arrays of shape
            ``(n_optimizable_params,)`` constraining the search space.
            Used by hierarchical refitting (Phase 4) to pass
            group-informed bounds.  When None, uses default bounds.

    Returns:
        (kernel, wasserstein_distance) — kernel is the same object,
        modified in-place with optimal parameters.

    Examples:
        >>> from nucml_next.data.kernels import build_kernel, KernelConfig
        >>> kernel = build_kernel(KernelConfig(outputscale=1.0))
        >>> kernel, w = optimize_kernel_wasserstein(
        ...     train_x, train_y, noise_var, kernel
        ... )
    """
    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()

    if mean_value is None:
        mean_value = np.mean(train_y)

    n_params = kernel.n_optimizable_params()

    if n_params == 1:
        return _optimize_kernel_1d(
            train_x, train_y, noise_variance, kernel,
            mean_value, lengthscale_bounds, n_grid, param_bounds,
        )
    else:
        return _optimize_kernel_nd(
            train_x, train_y, noise_variance, kernel,
            mean_value, n_grid, param_bounds,
        )


def _optimize_kernel_1d(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Union[float, np.ndarray],
    lengthscale_bounds: Tuple[float, float],
    n_grid: int,
    param_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple['Kernel', float]:
    """1-parameter optimisation: grid + Brent (same as existing RBF path)."""
    # Override bounds when hierarchical refit provides param_bounds
    effective_bounds = lengthscale_bounds
    if param_bounds is not None:
        effective_bounds = (float(param_bounds[0][0]), float(param_bounds[1][0]))

    # Grid search over the single optimisable parameter (log-spaced)
    ls_grid = np.logspace(
        np.log10(effective_bounds[0]),
        np.log10(effective_bounds[1]),
        n_grid,
    )

    wasserstein_values = []
    for ls in ls_grid:
        kernel.set_optimizable_params(np.array([ls]))
        z_scores = compute_loo_z_scores_kernel(
            train_x, train_y, noise_variance, kernel, mean_value
        )
        w = compute_wasserstein_calibration(z_scores)
        wasserstein_values.append(w)

    wasserstein_values = np.array(wasserstein_values)

    # Find best from grid
    best_idx = np.argmin(wasserstein_values)
    best_ls_grid = ls_grid[best_idx]
    best_w_grid = wasserstein_values[best_idx]

    if not np.isfinite(best_w_grid):
        logger.warning("Grid search failed (kernel 1D), using default")
        default_ls = (effective_bounds[0] * effective_bounds[1]) ** 0.5
        kernel.set_optimizable_params(np.array([default_ls]))
        return kernel, np.inf

    # Refine using Brent's method
    if best_idx == 0:
        refine_bounds = (ls_grid[0] / 2, ls_grid[1])
    elif best_idx == len(ls_grid) - 1:
        refine_bounds = (ls_grid[-2], ls_grid[-1] * 2)
    else:
        refine_bounds = (ls_grid[best_idx - 1], ls_grid[best_idx + 1])

    refine_bounds = (
        max(refine_bounds[0], effective_bounds[0]),
        min(refine_bounds[1], effective_bounds[1]),
    )

    def _objective(ls):
        kernel.set_optimizable_params(np.array([ls]))
        z_scores = compute_loo_z_scores_kernel(
            train_x, train_y, noise_variance, kernel, mean_value
        )
        return compute_wasserstein_calibration(z_scores)

    try:
        result = minimize_scalar(
            _objective,
            bounds=refine_bounds,
            method='bounded',
            options={'xatol': 1e-3},
        )
        if result.success and np.isfinite(result.fun) and result.fun < best_w_grid:
            kernel.set_optimizable_params(np.array([result.x]))
            return kernel, result.fun
    except Exception as e:
        logger.debug(f"Brent refinement failed (kernel): {e}")

    # Fall back to grid result
    kernel.set_optimizable_params(np.array([best_ls_grid]))
    return kernel, best_w_grid


def _optimize_kernel_nd(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Union[float, np.ndarray],
    n_grid: int,
    param_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple['Kernel', float]:
    """Multi-parameter optimisation: grid-initialised Nelder-Mead.

    For the Gibbs kernel this searches over (a₀, a₁) — only 2 parameters,
    so Nelder-Mead converges reliably.

    When ``param_bounds`` is provided, parameters are clamped to the
    feasible region inside the objective function.  This is used by
    hierarchical refitting (Phase 4) to constrain parameters to a
    group-informed range.
    """
    n_params = kernel.n_optimizable_params()

    # Generate random starting points around the current values
    # plus a few heuristic starting points
    current_params = kernel.get_optimizable_params().copy()
    rng = np.random.default_rng(42)

    # Grid of starting points: current + perturbations
    starts = [current_params.copy()]
    for _ in range(max(n_grid - 1, 4)):
        starts.append(current_params + rng.normal(0, 0.5, n_params))

    # Also try zero corrections (pure physics)
    starts.append(np.zeros(n_params))

    # Clip starting points to bounds if provided (ensures initial simplex
    # is within feasible region — Pass 1 outlier experiments may have
    # drifted outside the new constrained bounds)
    if param_bounds is not None:
        starts = [np.clip(s, param_bounds[0], param_bounds[1]) for s in starts]

    def _objective(params):
        if param_bounds is not None:
            params = np.clip(params, param_bounds[0], param_bounds[1])
        kernel.set_optimizable_params(params)
        z_scores = compute_loo_z_scores_kernel(
            train_x, train_y, noise_variance, kernel, mean_value
        )
        return compute_wasserstein_calibration(z_scores)

    # Evaluate all starting points
    best_w = np.inf
    best_params = current_params.copy()
    if param_bounds is not None:
        best_params = np.clip(best_params, param_bounds[0], param_bounds[1])

    for start in starts:
        try:
            w = _objective(start)
            if np.isfinite(w) and w < best_w:
                best_w = w
                best_params = start.copy()
        except Exception:
            continue

    # Refine from best starting point using Nelder-Mead
    try:
        result = minimize(
            _objective,
            x0=best_params,
            method='Nelder-Mead',
            options={
                'xatol': 1e-3,
                'fatol': 1e-4,
                'maxiter': 200,
                'adaptive': True,
            },
        )
        if np.isfinite(result.fun) and result.fun < best_w:
            best_w = result.fun
            best_params = result.x.copy()
            logger.debug(
                f"Nelder-Mead converged: W={best_w:.4f}, "
                f"params={best_params}"
            )
    except Exception as e:
        logger.debug(f"Nelder-Mead failed: {e}, using grid result")

    # Final clamp to bounds
    if param_bounds is not None:
        best_params = np.clip(best_params, param_bounds[0], param_bounds[1])

    kernel.set_optimizable_params(best_params)
    return kernel, best_w


def _wasserstein_loss_for_lengthscale(
    lengthscale: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    outputscale: float,
    mean_value: Union[float, np.ndarray],
) -> float:
    """
    Compute Wasserstein calibration loss for a given lengthscale.

    Used as objective function for lengthscale optimization.
    """
    if lengthscale <= 0:
        return np.inf

    z_scores = compute_loo_z_scores(
        train_x, train_y, noise_variance,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_value=mean_value,
    )

    return compute_wasserstein_calibration(z_scores)


def optimize_lengthscale_wasserstein(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    lengthscale_bounds: Tuple[float, float] = (0.01, 10.0),
    n_grid: int = 20,
    outputscale: Optional[float] = None,
    mean_value: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[float, float]:
    """
    Find optimal lengthscale by minimizing Wasserstein calibration distance.

    Algorithm:
    1. Grid search over lengthscales (log-spaced)
    2. For each: compute LOO z-scores, then Wasserstein distance
    3. Refine around minimum using scipy.optimize.minimize_scalar (Brent)

    Args:
        train_x: Training inputs, shape (n,) or (n, 1).
        train_y: Training targets, shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        lengthscale_bounds: (min, max) search bounds for lengthscale.
        n_grid: Number of grid points for initial search.
        outputscale: RBF kernel outputscale. If None, estimated from data.
        mean_value: Mean value(s). Scalar or array of shape (n,).
            If None, uses mean(train_y).

    Returns:
        (optimal_lengthscale, wasserstein_distance)

    Example:
        >>> log_E = np.linspace(0, 7, 100)
        >>> log_sigma = np.sin(log_E) + np.random.normal(0, 0.1, 100)
        >>> noise_var = np.full(100, 0.01)  # 0.1^2
        >>> ls, w = optimize_lengthscale_wasserstein(log_E, log_sigma, noise_var)
        >>> print(f"Optimal lengthscale: {ls:.3f}, Wasserstein: {w:.4f}")
    """
    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()

    # Estimate outputscale from data variance if not provided
    if outputscale is None:
        outputscale = np.var(train_y) - np.mean(noise_variance)
        outputscale = max(outputscale, 0.1)  # Ensure positive

    if mean_value is None:
        mean_value = np.mean(train_y)

    # Grid search
    ls_grid = np.logspace(
        np.log10(lengthscale_bounds[0]),
        np.log10(lengthscale_bounds[1]),
        n_grid
    )

    wasserstein_values = []
    for ls in ls_grid:
        w = _wasserstein_loss_for_lengthscale(
            ls, train_x, train_y, noise_variance, outputscale, mean_value
        )
        wasserstein_values.append(w)

    wasserstein_values = np.array(wasserstein_values)

    # Find best from grid
    best_idx = np.argmin(wasserstein_values)
    best_ls_grid = ls_grid[best_idx]
    best_w_grid = wasserstein_values[best_idx]

    # If grid search failed completely, return conservative default
    if not np.isfinite(best_w_grid):
        logger.warning("Grid search failed, using default lengthscale")
        default_ls = (lengthscale_bounds[0] * lengthscale_bounds[1]) ** 0.5
        return default_ls, np.inf

    # Refine using Brent's method
    # Search in neighborhood of best grid point
    if best_idx == 0:
        refine_bounds = (ls_grid[0] / 2, ls_grid[1])
    elif best_idx == len(ls_grid) - 1:
        refine_bounds = (ls_grid[-2], ls_grid[-1] * 2)
    else:
        refine_bounds = (ls_grid[best_idx - 1], ls_grid[best_idx + 1])

    # Ensure bounds are valid
    refine_bounds = (
        max(refine_bounds[0], lengthscale_bounds[0]),
        min(refine_bounds[1], lengthscale_bounds[1])
    )

    try:
        result = minimize_scalar(
            lambda ls: _wasserstein_loss_for_lengthscale(
                ls, train_x, train_y, noise_variance, outputscale, mean_value
            ),
            bounds=refine_bounds,
            method='bounded',
            options={'xatol': 1e-3}
        )

        if result.success and np.isfinite(result.fun) and result.fun < best_w_grid:
            return result.x, result.fun
    except Exception as e:
        logger.debug(f"Brent refinement failed: {e}")

    # Fall back to grid result
    return best_ls_grid, best_w_grid


def calibration_diagnostic(
    z_scores: np.ndarray,
    sigma_levels: np.ndarray = None,
) -> dict:
    """
    Compute calibration diagnostic metrics.

    Compares empirical coverage at various sigma levels to theoretical
    Gaussian coverage.

    Args:
        z_scores: Empirical z-scores from GP.
        sigma_levels: Sigma levels to evaluate. Default: [1, 2, 3].

    Returns:
        Dictionary with:
            - wasserstein: Overall Wasserstein distance
            - coverage_empirical: Dict of sigma -> empirical coverage
            - coverage_theoretical: Dict of sigma -> theoretical coverage
            - coverage_error: Dict of sigma -> (empirical - theoretical)
    """
    from scipy.stats import norm

    if sigma_levels is None:
        sigma_levels = np.array([1.0, 2.0, 3.0])

    valid_z = z_scores[np.isfinite(z_scores)]
    abs_z = np.abs(valid_z)

    # Wasserstein distance
    wasserstein = compute_wasserstein_calibration(z_scores)

    # Coverage at each sigma level
    coverage_empirical = {}
    coverage_theoretical = {}
    coverage_error = {}

    for sigma in sigma_levels:
        # Empirical: fraction of |z| <= sigma
        empirical = np.mean(abs_z <= sigma)
        # Theoretical: 2 * Phi(sigma) - 1 = erf(sigma / sqrt(2))
        theoretical = 2 * norm.cdf(sigma) - 1

        coverage_empirical[sigma] = empirical
        coverage_theoretical[sigma] = theoretical
        coverage_error[sigma] = empirical - theoretical

    return {
        'wasserstein': wasserstein,
        'coverage_empirical': coverage_empirical,
        'coverage_theoretical': coverage_theoretical,
        'coverage_error': coverage_error,
        'n_valid': len(valid_z),
        'z_mean': np.mean(abs_z),
        'z_std': np.std(valid_z),
    }


# =============================================================================
# PyTorch-Accelerated Calibration Functions (GPU-compatible)
# =============================================================================

def compute_loo_z_scores_torch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    lengthscale: float,
    outputscale: float = 1.0,
    mean_value: Optional[Union[float, np.ndarray]] = None,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Compute LOO z-scores using PyTorch (GPU-accelerated).

    Same algorithm as compute_loo_z_scores but uses PyTorch for GPU support.
    Falls back to NumPy version if PyTorch is not available.

    Args:
        train_x: Training inputs, shape (n,).
        train_y: Training targets, shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        lengthscale: RBF kernel lengthscale.
        outputscale: RBF kernel outputscale.
        mean_value: Mean value(s). Scalar or array of shape (n,).
            If None, uses mean(train_y).
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        LOO z-scores, shape (n,).
    """
    try:
        import torch
    except ImportError:
        # Fall back to NumPy version
        return compute_loo_z_scores(
            train_x, train_y, noise_variance,
            lengthscale, outputscale, mean_value
        )

    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()
    n = len(train_x)

    if mean_value is None:
        mean_value = float(np.mean(train_y))

    # Validate CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'

    torch_device = torch.device(device)

    # Convert to tensors
    x = torch.tensor(train_x, dtype=torch.float64, device=torch_device)
    y = torch.tensor(train_y, dtype=torch.float64, device=torch_device)
    noise_var = torch.tensor(noise_variance, dtype=torch.float64, device=torch_device)

    # Build RBF kernel matrix: K(x, x') = outputscale * exp(-0.5 * ||x - x'||^2 / ls^2)
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n, n)
    K = outputscale * torch.exp(-0.5 * diff.pow(2) / (lengthscale ** 2))

    # Add noise to diagonal + jitter
    K = K + torch.diag(noise_var) + 1e-6 * torch.eye(n, dtype=torch.float64, device=torch_device)

    # Cholesky decomposition
    try:
        L = torch.linalg.cholesky(K)
    except RuntimeError:
        logger.warning("Cholesky failed in LOO computation (torch)")
        return np.full(n, np.inf)

    # Compute K^{-1} @ residuals — mean_value may be scalar or array
    if np.isscalar(mean_value):
        mean_tensor = mean_value
    else:
        mean_tensor = torch.tensor(
            np.asarray(mean_value), dtype=torch.float64, device=torch_device
        )
    residuals = y - mean_tensor
    # Solve L @ z = residuals
    z = torch.linalg.solve_triangular(L, residuals.unsqueeze(1), upper=False).squeeze(1)
    # Solve L.T @ x = z
    K_inv_r = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze(1)

    # Compute diagonal of K^{-1}
    # (K^{-1})_{ii} = sum_j (L^{-1})_{ji}^2
    L_inv = torch.linalg.solve_triangular(
        L, torch.eye(n, dtype=torch.float64, device=torch_device), upper=False
    )
    K_inv_diag = (L_inv ** 2).sum(dim=0)

    # LOO z-scores
    z_scores = K_inv_r / torch.sqrt(torch.clamp(K_inv_diag, min=1e-10))

    return z_scores.cpu().numpy()


def _wasserstein_loss_torch(
    lengthscale: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    outputscale: float,
    mean_value: Union[float, np.ndarray],
    device: str = 'cpu',
) -> float:
    """
    Compute Wasserstein calibration loss using PyTorch LOO z-scores.
    """
    if lengthscale <= 0:
        return np.inf

    z_scores = compute_loo_z_scores_torch(
        train_x, train_y, noise_variance,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_value=mean_value,
        device=device,
    )

    return compute_wasserstein_calibration(z_scores)


def optimize_lengthscale_wasserstein_torch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    lengthscale_bounds: Tuple[float, float] = (0.01, 10.0),
    n_grid: int = 20,
    outputscale: Optional[float] = None,
    mean_value: Optional[Union[float, np.ndarray]] = None,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Find optimal lengthscale by minimizing Wasserstein calibration distance (GPU-accelerated).

    Same algorithm as optimize_lengthscale_wasserstein but uses PyTorch for GPU.

    Args:
        train_x: Training inputs, shape (n,).
        train_y: Training targets, shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        lengthscale_bounds: (min, max) search bounds for lengthscale.
        n_grid: Number of grid points for initial search.
        outputscale: RBF kernel outputscale. If None, estimated from data.
        mean_value: Mean value(s). Scalar or array of shape (n,).
            If None, uses mean(train_y).
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        (optimal_lengthscale, wasserstein_distance)
    """
    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()

    # Estimate outputscale from data variance if not provided
    if outputscale is None:
        outputscale = np.var(train_y) - np.mean(noise_variance)
        outputscale = max(outputscale, 0.1)

    if mean_value is None:
        mean_value = np.mean(train_y)

    # Grid search using PyTorch LOO z-scores
    ls_grid = np.logspace(
        np.log10(lengthscale_bounds[0]),
        np.log10(lengthscale_bounds[1]),
        n_grid
    )

    wasserstein_values = []
    for ls in ls_grid:
        w = _wasserstein_loss_torch(
            ls, train_x, train_y, noise_variance, outputscale, mean_value, device
        )
        wasserstein_values.append(w)

    wasserstein_values = np.array(wasserstein_values)

    # Find best from grid
    best_idx = np.argmin(wasserstein_values)
    best_ls_grid = ls_grid[best_idx]
    best_w_grid = wasserstein_values[best_idx]

    # If grid search failed completely, return conservative default
    if not np.isfinite(best_w_grid):
        logger.warning("Grid search failed, using default lengthscale")
        default_ls = (lengthscale_bounds[0] * lengthscale_bounds[1]) ** 0.5
        return default_ls, np.inf

    # Refine using Brent's method
    if best_idx == 0:
        refine_bounds = (ls_grid[0] / 2, ls_grid[1])
    elif best_idx == len(ls_grid) - 1:
        refine_bounds = (ls_grid[-2], ls_grid[-1] * 2)
    else:
        refine_bounds = (ls_grid[best_idx - 1], ls_grid[best_idx + 1])

    refine_bounds = (
        max(refine_bounds[0], lengthscale_bounds[0]),
        min(refine_bounds[1], lengthscale_bounds[1])
    )

    try:
        result = minimize_scalar(
            lambda ls: _wasserstein_loss_torch(
                ls, train_x, train_y, noise_variance, outputscale, mean_value, device
            ),
            bounds=refine_bounds,
            method='bounded',
            options={'xatol': 1e-3}
        )

        if result.success and np.isfinite(result.fun) and result.fun < best_w_grid:
            best_ls, best_w = result.x, result.fun
        else:
            best_ls, best_w = best_ls_grid, best_w_grid
    except Exception as e:
        logger.debug(f"Brent refinement failed: {e}")
        best_ls, best_w = best_ls_grid, best_w_grid

    # Clear GPU memory after lengthscale optimization
    if device != 'cpu':
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    return best_ls, best_w


# =============================================================================
# Kernel-Aware PyTorch Functions
# =============================================================================

def compute_loo_z_scores_kernel_torch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Optional[Union[float, np.ndarray]] = None,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Compute LOO z-scores using a Kernel object (PyTorch, GPU-accelerated).

    Identical algorithm to ``compute_loo_z_scores_kernel`` but builds the
    kernel matrix via ``kernel.compute_matrix_torch()`` for GPU support.
    Falls back to NumPy version if PyTorch is not available.

    Args:
        train_x: Training inputs (log₁₀(E [eV])), shape (n,).
        train_y: Training targets (log₁₀(σ [b])), shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        kernel: Kernel object implementing ``compute_matrix_torch()``.
        mean_value: Mean value(s). Scalar or array of shape (n,).
            If None, uses mean(train_y).
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        LOO z-scores, shape (n,).
    """
    try:
        import torch
    except ImportError:
        return compute_loo_z_scores_kernel(
            train_x, train_y, noise_variance, kernel, mean_value
        )

    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()
    n = len(train_x)

    if mean_value is None:
        mean_value = float(np.mean(train_y))

    # Validate CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'

    torch_device = torch.device(device)

    # Convert to tensors
    x = torch.tensor(train_x, dtype=torch.float64, device=torch_device)
    y = torch.tensor(train_y, dtype=torch.float64, device=torch_device)
    noise_var = torch.tensor(noise_variance, dtype=torch.float64, device=torch_device)

    # Build kernel matrix via the kernel object (PyTorch path)
    K = kernel.compute_matrix_torch(x, x, device=device)

    # Add noise to diagonal + jitter
    K = K + torch.diag(noise_var) + 1e-6 * torch.eye(n, dtype=torch.float64, device=torch_device)

    # Cholesky decomposition
    try:
        L = torch.linalg.cholesky(K)
    except RuntimeError:
        logger.warning("Cholesky failed in LOO computation (kernel torch)")
        return np.full(n, np.inf)

    # Compute K^{-1} @ residuals
    if np.isscalar(mean_value):
        mean_tensor = mean_value
    else:
        mean_tensor = torch.tensor(
            np.asarray(mean_value), dtype=torch.float64, device=torch_device
        )
    residuals = y - mean_tensor

    z = torch.linalg.solve_triangular(L, residuals.unsqueeze(1), upper=False).squeeze(1)
    K_inv_r = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze(1)

    # Compute diagonal of K^{-1}
    L_inv = torch.linalg.solve_triangular(
        L, torch.eye(n, dtype=torch.float64, device=torch_device), upper=False
    )
    K_inv_diag = (L_inv ** 2).sum(dim=0)

    # LOO z-scores
    z_scores = K_inv_r / torch.sqrt(torch.clamp(K_inv_diag, min=1e-10))

    return z_scores.cpu().numpy()


def optimize_kernel_wasserstein_torch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    kernel: 'Kernel',
    mean_value: Optional[Union[float, np.ndarray]] = None,
    lengthscale_bounds: Tuple[float, float] = (0.01, 10.0),
    n_grid: int = 20,
    device: str = 'cpu',
    param_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple['Kernel', float]:
    """
    Optimise kernel parameters via Wasserstein calibration (GPU-accelerated).

    Same algorithm as ``optimize_kernel_wasserstein`` but uses PyTorch for
    kernel matrix computation.

    Args:
        train_x: Training inputs (log₁₀(E [eV])), shape (n,).
        train_y: Training targets (log₁₀(σ [b])), shape (n,).
        noise_variance: Per-point noise variance, shape (n,).
        kernel: Kernel object.
        mean_value: Mean value(s). If None, uses mean(train_y).
        lengthscale_bounds: Search bounds for 1-param kernels.
        n_grid: Number of grid points for initial search.
        device: PyTorch device ('cpu' or 'cuda').
        param_bounds: Optional ``(lower, upper)`` arrays constraining
            the search space.  Used by hierarchical refitting.

    Returns:
        (kernel, wasserstein_distance) — kernel modified in-place.
    """
    train_x = np.asarray(train_x).ravel()
    train_y = np.asarray(train_y).ravel()
    noise_variance = np.asarray(noise_variance).ravel()

    if mean_value is None:
        mean_value = np.mean(train_y)

    n_params = kernel.n_optimizable_params()

    def _objective(params):
        if param_bounds is not None:
            params = np.clip(params, param_bounds[0], param_bounds[1])
        kernel.set_optimizable_params(params)
        z_scores = compute_loo_z_scores_kernel_torch(
            train_x, train_y, noise_variance, kernel, mean_value, device
        )
        return compute_wasserstein_calibration(z_scores)

    if n_params == 1:
        # 1D: Grid + Brent (same as _optimize_kernel_1d but using torch)
        # Override bounds when hierarchical refit provides param_bounds
        effective_bounds = lengthscale_bounds
        if param_bounds is not None:
            effective_bounds = (float(param_bounds[0][0]), float(param_bounds[1][0]))

        ls_grid = np.logspace(
            np.log10(effective_bounds[0]),
            np.log10(effective_bounds[1]),
            n_grid,
        )

        wasserstein_values = []
        for ls in ls_grid:
            w = _objective(np.array([ls]))
            wasserstein_values.append(w)

        wasserstein_values = np.array(wasserstein_values)
        best_idx = np.argmin(wasserstein_values)
        best_ls_grid = ls_grid[best_idx]
        best_w_grid = wasserstein_values[best_idx]

        if not np.isfinite(best_w_grid):
            default_ls = (effective_bounds[0] * effective_bounds[1]) ** 0.5
            kernel.set_optimizable_params(np.array([default_ls]))
            return kernel, np.inf

        # Brent refinement
        if best_idx == 0:
            refine_bounds = (ls_grid[0] / 2, ls_grid[1])
        elif best_idx == len(ls_grid) - 1:
            refine_bounds = (ls_grid[-2], ls_grid[-1] * 2)
        else:
            refine_bounds = (ls_grid[best_idx - 1], ls_grid[best_idx + 1])

        refine_bounds = (
            max(refine_bounds[0], effective_bounds[0]),
            min(refine_bounds[1], effective_bounds[1]),
        )

        try:
            result = minimize_scalar(
                lambda ls: _objective(np.array([ls])),
                bounds=refine_bounds,
                method='bounded',
                options={'xatol': 1e-3},
            )
            if result.success and np.isfinite(result.fun) and result.fun < best_w_grid:
                kernel.set_optimizable_params(np.array([result.x]))
                return kernel, result.fun
        except Exception as e:
            logger.debug(f"Brent refinement failed (kernel torch): {e}")

        kernel.set_optimizable_params(np.array([best_ls_grid]))
        return kernel, best_w_grid

    else:
        # Multi-parameter: Nelder-Mead
        current_params = kernel.get_optimizable_params().copy()
        rng = np.random.default_rng(42)

        starts = [current_params.copy()]
        for _ in range(max(n_grid - 1, 4)):
            starts.append(current_params + rng.normal(0, 0.5, n_params))
        starts.append(np.zeros(n_params))

        # Clip starting points to bounds if provided
        if param_bounds is not None:
            starts = [np.clip(s, param_bounds[0], param_bounds[1]) for s in starts]

        best_w = np.inf
        best_params = current_params.copy()
        if param_bounds is not None:
            best_params = np.clip(best_params, param_bounds[0], param_bounds[1])

        for start in starts:
            try:
                w = _objective(start)
                if np.isfinite(w) and w < best_w:
                    best_w = w
                    best_params = start.copy()
            except Exception:
                continue

        try:
            result = minimize(
                _objective,
                x0=best_params,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-3,
                    'fatol': 1e-4,
                    'maxiter': 200,
                    'adaptive': True,
                },
            )
            if np.isfinite(result.fun) and result.fun < best_w:
                best_w = result.fun
                best_params = result.x.copy()
        except Exception as e:
            logger.debug(f"Nelder-Mead failed (torch): {e}")

        # Final clamp to bounds
        if param_bounds is not None:
            best_params = np.clip(best_params, param_bounds[0], param_bounds[1])

        kernel.set_optimizable_params(best_params)

        # Clear GPU memory
        if device != 'cpu':
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        return kernel, best_w
