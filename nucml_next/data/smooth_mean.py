"""
Data-Driven Consensus Smooth Mean for GP Outlier Detection
==========================================================

Computes a robust smooth mean from pooled EXFOR data across all experiments
per (Z, A, MT) group BEFORE fitting per-experiment GPs.  The GP then only
models residuals from this trend, dramatically reducing the structure a
single-lengthscale RBF kernel must capture.

**Evaluation-independence principle:** The smooth mean comes from the EXFOR
data itself, NOT from evaluated nuclear data libraries (ENDF/B, JEFF, JENDL),
to avoid circularity in a tool meant to provide independent quality assessment
to evaluators.

Key Classes:
    SmoothMeanConfig: Configuration for smooth mean computation.

Key Functions:
    fit_smooth_mean: Fit a smooth mean function from pooled data.

Usage:
    >>> from nucml_next.data.smooth_mean import SmoothMeanConfig, fit_smooth_mean
    >>> config = SmoothMeanConfig(smooth_mean_type='spline')
    >>> mean_fn = fit_smooth_mean(log_E, log_sigma, config)
    >>> trend = mean_fn(log_E_query)  # Evaluate at any points
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SmoothMeanConfig:
    """Configuration for smooth mean computation.

    Attributes:
        smooth_mean_type: Type of smooth mean to compute.
            'constant' - Uses np.mean(log_sigma), preserving current behaviour.
            'spline'   - Iterative reweighted UnivariateSpline (robust to outliers).
        spline_smoothing_factor: Smoothing factor ``s`` for UnivariateSpline.
            None means scipy chooses automatically (cross-validated).
        spline_degree: B-spline degree (1=linear, 3=cubic, 5=quintic).
        sigma_clip_threshold: Number of standard deviations for sigma-clipping
            during iterative reweighting.  Points with |residual| > threshold * MAD
            are downweighted on each iteration.
        max_iterations: Maximum sigma-clipping iterations.
        convergence_tol: Stop iterating when max absolute change in spline
            predictions between iterations falls below this value.
        min_points_for_spline: Groups with fewer points fall back to constant mean.
    """

    smooth_mean_type: str = 'constant'
    spline_smoothing_factor: Optional[float] = None
    spline_degree: int = 3
    sigma_clip_threshold: float = 3.0
    max_iterations: int = 5
    convergence_tol: float = 1e-4
    min_points_for_spline: int = 10


def fit_smooth_mean(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    config: Optional[SmoothMeanConfig] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a smooth mean function from pooled cross-section data.

    The returned callable maps ``log_E`` arrays to ``log_sigma`` arrays and
    can be evaluated at arbitrary energy points (including outside the
    training range, where it extrapolates linearly via the spline).

    Algorithm (when ``smooth_mean_type='spline'``):
        1. Sort by log_E, filter NaN/inf.
        2. Fit initial ``UnivariateSpline(log_E, log_sigma, k=degree, s=auto)``.
        3. Iterate sigma-clipping: compute residuals, downweight
           ``|r| > threshold * MAD``, refit with weights.
        4. Converge (max iterations) or stop when max change < tol.

    Args:
        log_E: log10(Energy) values, shape (n,).
        log_sigma: log10(CrossSection) values, shape (n,).
        config: Configuration.  ``None`` uses defaults (constant mean).

    Returns:
        Callable ``mean_fn(log_E_array) -> log_sigma_array`` that evaluates
        the smooth mean at given energy points.

    Examples:
        >>> config = SmoothMeanConfig(smooth_mean_type='spline')
        >>> mean_fn = fit_smooth_mean(log_E, log_sigma, config)
        >>> trend = mean_fn(log_E)
    """
    if config is None:
        config = SmoothMeanConfig()

    log_E = np.asarray(log_E, dtype=float).ravel()
    log_sigma = np.asarray(log_sigma, dtype=float).ravel()

    # Filter NaN / inf from both arrays simultaneously
    valid = np.isfinite(log_E) & np.isfinite(log_sigma)
    log_E_clean = log_E[valid]
    log_sigma_clean = log_sigma[valid]
    n = len(log_E_clean)

    # Fall back to constant if requested, too few points, or degenerate data
    if (
        config.smooth_mean_type == 'constant'
        or n < config.min_points_for_spline
        or n < config.spline_degree + 1  # need k+1 points minimum
    ):
        return _fit_constant_mean(log_sigma_clean)

    if config.smooth_mean_type == 'spline':
        return _fit_spline_mean(log_E_clean, log_sigma_clean, config)

    raise ValueError(
        f"Unknown smooth_mean_type: {config.smooth_mean_type!r}. "
        f"Expected 'constant' or 'spline'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fit_constant_mean(log_sigma: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that produces a constant mean array."""
    if len(log_sigma) == 0:
        c = 0.0
    else:
        c = float(np.mean(log_sigma))

    def _constant(log_E: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(log_E).shape, c)

    return _constant


def _fit_spline_mean(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    config: SmoothMeanConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit an iterative reweighted UnivariateSpline.

    Returns a callable that evaluates the spline at arbitrary points.
    Falls back to constant mean if the spline fit fails for any reason.
    """
    from scipy.interpolate import UnivariateSpline

    # Sort by energy (UnivariateSpline requires sorted x)
    order = np.argsort(log_E)
    x = log_E[order]
    y = log_sigma[order]

    # Handle duplicate x values: UnivariateSpline requires strictly
    # increasing x.  Average y for duplicate x groups.
    x, y = _deduplicate_sorted(x, y)
    n = len(x)

    if n < config.spline_degree + 1:
        logger.debug(
            f"Only {n} unique energies after dedup (need {config.spline_degree + 1}), "
            f"falling back to constant mean"
        )
        return _fit_constant_mean(y)

    # Check energy range: if all energies are identical, fall back
    x_range = x[-1] - x[0]
    if x_range < 1e-12:
        logger.debug("Degenerate energy range, falling back to constant mean")
        return _fit_constant_mean(y)

    # Initial uniform weights
    weights = np.ones(n)

    # Determine smoothing factor.
    # scipy auto-selection (s ≈ n) allows ~n knots which can be too flexible
    # for small groups (50–200 points), letting the spline trace individual
    # resonances instead of the gross energy-dependent envelope.  Override
    # with s = n × Var(y) to force heavy smoothing (~O(10) knots).
    s = config.spline_smoothing_factor
    if s is None:
        s = n * max(np.var(y), 1e-6)

    try:
        spl = UnivariateSpline(x, y, w=weights, k=config.spline_degree, s=s)
    except Exception as e:
        logger.debug(f"Initial spline fit failed: {e}, falling back to constant mean")
        return _fit_constant_mean(y)

    # Iterative sigma-clipping reweighting
    prev_pred = spl(x)
    for iteration in range(config.max_iterations):
        residuals = y - prev_pred

        # Robust scale estimate: MAD (median absolute deviation)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad < 1e-12:
            # All residuals essentially zero — perfect fit, stop
            break
        sigma_est = mad * 1.4826  # Consistency constant for normal distribution

        # Downweight outliers (Huber-like soft clipping)
        abs_r = np.abs(residuals) / sigma_est
        weights = np.where(
            abs_r <= config.sigma_clip_threshold,
            1.0,
            config.sigma_clip_threshold / abs_r,
        )

        # Refit with new weights
        try:
            spl = UnivariateSpline(x, y, w=weights, k=config.spline_degree, s=s)
        except Exception as e:
            logger.debug(
                f"Spline refit failed at iteration {iteration}: {e}, "
                f"using previous iteration"
            )
            break

        new_pred = spl(x)

        # Convergence check
        max_change = np.max(np.abs(new_pred - prev_pred))
        if max_change < config.convergence_tol:
            logger.debug(
                f"Smooth mean converged after {iteration + 1} iterations "
                f"(max_change={max_change:.2e})"
            )
            break

        prev_pred = new_pred

    # Wrap the fitted spline in a clean callable
    def _spline_mean(log_E_query: np.ndarray) -> np.ndarray:
        log_E_query = np.asarray(log_E_query, dtype=float).ravel()
        return spl(log_E_query)

    return _spline_mean


def _deduplicate_sorted(x: np.ndarray, y: np.ndarray) -> tuple:
    """Average y values for duplicate x entries in a sorted array.

    Args:
        x: Sorted array of x values (may contain duplicates).
        y: Corresponding y values.

    Returns:
        (x_unique, y_averaged) with strictly increasing x.
    """
    if len(x) <= 1:
        return x.copy(), y.copy()

    # Find indices where x changes
    diff = np.diff(x)
    # Treat values within 1e-14 relative tolerance as duplicates
    is_new = diff > np.maximum(np.abs(x[:-1]) * 1e-14, 1e-30)
    split_indices = np.where(is_new)[0] + 1

    if len(split_indices) == len(x) - 1:
        # No duplicates
        return x.copy(), y.copy()

    # Split into groups and average
    x_groups = np.split(x, split_indices)
    y_groups = np.split(y, split_indices)

    x_unique = np.array([g[0] for g in x_groups])
    y_averaged = np.array([np.mean(g) for g in y_groups])

    return x_unique, y_averaged
