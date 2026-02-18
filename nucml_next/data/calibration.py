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

Usage:
    >>> from nucml_next.data.calibration import optimize_lengthscale_wasserstein
    >>> optimal_ls, wasserstein_dist = optimize_lengthscale_wasserstein(
    ...     log_E, log_sigma, noise_variance
    ... )
"""

import logging
from typing import Tuple, Optional, Callable

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize_scalar

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
    mean_value: Optional[float] = None,
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
        mean_value: Constant mean value. If None, uses mean(train_y).

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
    mean = np.full(n, mean_value)
    return compute_loo_z_scores_from_cholesky(L, train_y, mean)


def _wasserstein_loss_for_lengthscale(
    lengthscale: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    noise_variance: np.ndarray,
    outputscale: float,
    mean_value: float,
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
    mean_value: Optional[float] = None,
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
        mean_value: Constant mean. If None, uses mean(train_y).

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
    mean_value: Optional[float] = None,
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
        mean_value: Constant mean value. If None, uses mean(train_y).
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

    # Compute K^{-1} @ residuals
    residuals = y - mean_value
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
    mean_value: float,
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
    mean_value: Optional[float] = None,
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
        mean_value: Constant mean. If None, uses mean(train_y).
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
            return result.x, result.fun
    except Exception as e:
        logger.debug(f"Brent refinement failed: {e}")

    return best_ls_grid, best_w_grid
