"""
Data-Driven Smooth Mean and Rolling MAD for Outlier Detection
=============================================================

Computes a robust smooth mean from pooled EXFOR data across all experiments
per (Z, A, MT) group, then estimates energy-dependent scatter (MAD) to
produce z-scores for outlier detection.

**Evaluation-independence principle:** The smooth mean comes from the EXFOR
data itself, NOT from evaluated nuclear data libraries (ENDF/B, JEFF, JENDL),
to avoid circularity in a tool meant to provide independent quality assessment
to evaluators.

Key Classes:
    SmoothMeanConfig: Configuration for smooth mean computation.

Key Functions:
    fit_smooth_mean: Fit a smooth mean function from pooled data.
    compute_rolling_mad_interpolator: Energy-dependent MAD interpolator
        for outlier scoring.

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
    weights: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a smooth mean function from pooled cross-section data.

    The returned callable maps ``log_E`` arrays to ``log_sigma`` arrays and
    can be evaluated at arbitrary energy points (including outside the
    training range, where it extrapolates linearly via the spline).

    Algorithm (when ``smooth_mean_type='spline'``):
        1. Sort by log_E, filter NaN/inf.
        2. Fit initial ``UnivariateSpline(log_E, log_sigma, k=degree, s=auto)``.
        3. Iterate sigma-clipping: compute residuals, downweight
           ``|r| > threshold * MAD``, refit with combined weights.
        4. Converge (max iterations) or stop when max change < tol.

    Args:
        log_E: log10(Energy) values, shape (n,).
        log_sigma: log10(CrossSection) values, shape (n,).
        config: Configuration.  ``None`` uses defaults (constant mean).
        weights: Optional per-point fitting weights, shape (n,).  Typically
            inverse-variance weights from reported measurement uncertainties:
            ``w = 1 / σ_log²``.  Combined multiplicatively with sigma-clipping
            weights during iterative reweighting.  ``None`` = equal weights.

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
    weights_clean = weights[valid] if weights is not None else None
    n = len(log_E_clean)

    # Fall back to constant if requested, too few points, or degenerate data
    if (
        config.smooth_mean_type == 'constant'
        or n < config.min_points_for_spline
        or n < config.spline_degree + 1  # need k+1 points minimum
    ):
        return _fit_constant_mean(log_sigma_clean)

    if config.smooth_mean_type == 'spline':
        return _fit_spline_mean(log_E_clean, log_sigma_clean, config,
                                weights=weights_clean)

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
    weights: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit an iterative reweighted UnivariateSpline.

    Returns a callable that evaluates the spline at arbitrary points.
    Falls back to constant mean if the spline fit fails for any reason.

    Args:
        log_E: Sorted-clean log10(Energy), shape (n,).
        log_sigma: Corresponding log10(CrossSection), shape (n,).
        config: Spline configuration.
        weights: Optional per-point fitting weights (e.g. 1/σ²). Combined
            multiplicatively with sigma-clipping weights each iteration.
    """
    from scipy.interpolate import UnivariateSpline

    # Sort by energy (UnivariateSpline requires sorted x)
    order = np.argsort(log_E)
    x = log_E[order]
    y = log_sigma[order]
    user_w = weights[order] if weights is not None else None

    # Handle duplicate x values: UnivariateSpline requires strictly
    # increasing x.  Average y (and weights) for duplicate x groups.
    x, y, user_w = _deduplicate_sorted(x, y, w=user_w)
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

    # Normalise user weights so they sum to n (preserves effective sample
    # size for the smoothing parameter s)
    if user_w is not None:
        user_w = user_w * n / max(user_w.sum(), 1e-10)

    # Initial weights: user weights or uniform
    w = user_w.copy() if user_w is not None else np.ones(n)

    # Determine smoothing factor.
    # scipy auto-selection (s ≈ n) allows ~n knots which can be too flexible
    # for small groups (50–200 points), letting the spline trace individual
    # resonances instead of the gross energy-dependent envelope.  Override
    # with s = n × Var(y) to force heavy smoothing (~O(10) knots).
    s = config.spline_smoothing_factor
    if s is None:
        s = n * max(np.var(y), 1e-6)

    try:
        spl = UnivariateSpline(x, y, w=w, k=config.spline_degree, s=s)
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
        clip_weights = np.where(
            abs_r <= config.sigma_clip_threshold,
            1.0,
            config.sigma_clip_threshold / abs_r,
        )

        # Combine clip weights with user weights multiplicatively
        if user_w is not None:
            w = clip_weights * user_w
            # Re-normalise so combined weights sum to n
            w = w * n / max(w.sum(), 1e-10)
        else:
            w = clip_weights

        # Refit with new weights
        try:
            spl = UnivariateSpline(x, y, w=w, k=config.spline_degree, s=s)
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


def _deduplicate_sorted(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> tuple:
    """Average y (and w) values for duplicate x entries in a sorted array.

    Args:
        x: Sorted array of x values (may contain duplicates).
        y: Corresponding y values.
        w: Optional weights array to average along with y.

    Returns:
        ``(x_unique, y_averaged)`` with strictly increasing x when ``w``
        is None, or ``(x_unique, y_averaged, w_averaged)`` when ``w``
        is provided.
    """
    if len(x) <= 1:
        if w is not None:
            return x.copy(), y.copy(), w.copy()
        return x.copy(), y.copy(), None

    # Find indices where x changes
    diff = np.diff(x)
    # Treat values within 1e-14 relative tolerance as duplicates
    is_new = diff > np.maximum(np.abs(x[:-1]) * 1e-14, 1e-30)
    split_indices = np.where(is_new)[0] + 1

    if len(split_indices) == len(x) - 1:
        # No duplicates
        if w is not None:
            return x.copy(), y.copy(), w.copy()
        return x.copy(), y.copy(), None

    # Split into groups and average
    x_groups = np.split(x, split_indices)
    y_groups = np.split(y, split_indices)

    x_unique = np.array([g[0] for g in x_groups])
    y_averaged = np.array([np.mean(g) for g in y_groups])

    if w is not None:
        w_groups = np.split(w, split_indices)
        w_averaged = np.array([np.mean(g) for g in w_groups])
        return x_unique, y_averaged, w_averaged

    return x_unique, y_averaged, None


# ---------------------------------------------------------------------------
# Rolling MAD computation
# ---------------------------------------------------------------------------


def _compute_rolling_mad(
    x: np.ndarray,
    r: np.ndarray,
    window_fraction: float = 0.1,
    min_window_points: int = 15,
) -> np.ndarray:
    """Rolling MAD of residuals in adaptive energy windows.

    Used by ``compute_rolling_mad_interpolator()``.

    Args:
        x: Sorted energy values, shape (n,).
        r: Residuals from smooth mean (same order as x), shape (n,).
        window_fraction: Half-width of the rolling window as a fraction
            of the total energy range.
        min_window_points: Minimum number of neighbours in a window.

    Returns:
        Array of MAD * 1.4826 (robust std estimate) at each point, shape (n,).
    """
    n = len(x)
    energy_range = x[-1] - x[0]
    half_width = max(energy_range * window_fraction, 0.1)

    mad = np.empty(n)
    for i in range(n):
        lo = x[i] - half_width
        hi = x[i] + half_width
        lo_idx = np.searchsorted(x, lo, side='left')    # O(log n)
        hi_idx = np.searchsorted(x, hi, side='right')   # O(log n)
        n_in_window = hi_idx - lo_idx

        if n_in_window < min_window_points:
            distances = np.abs(x - x[i])
            k = min(min_window_points, n)
            if k >= n:
                r_local = r  # Use all points
            else:
                idx = np.argpartition(distances, k)[:k]
                r_local = r[idx]
        else:
            r_local = r[lo_idx:hi_idx]  # contiguous slice, no copy

        med_local = np.median(r_local)
        mad[i] = np.median(np.abs(r_local - med_local)) * 1.4826

    return mad


def compute_rolling_mad_interpolator(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    window_fraction: float = 0.1,
    min_window_points: int = 15,
    mad_floor: float = 0.02,  # ~5% relative — minimum plausible scatter
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Compute energy-dependent MAD of residuals from smooth mean.

    Returns a callable that maps ``log₁₀(E)`` → ``local_MAD(log₁₀(E))``.
    The MAD is computed in sliding energy windows on the pooled residuals,
    giving an energy-local measure of cross-section scatter.

    In regions where data varies wildly (resonance region), MAD is large.
    In smooth regions (thermal, continuum), MAD is small.

    This is designed for the ``local_mad`` scoring method in
    ``ExperimentOutlierDetector``, where z-scores are computed as
    ``|residual| / local_MAD``.

    Args:
        log_E: log₁₀(Energy/eV) values from pooled group data.
        log_sigma: log₁₀(CrossSection) values from pooled group data.
        mean_fn: Smooth mean function (from ``fit_smooth_mean()``).
        window_fraction: Fraction of energy range for rolling window.
            Default 0.1 = 10% of the total energy range.
        min_window_points: Minimum points per window. If the window
            contains fewer points, it expands to include k-nearest.
        mad_floor: Minimum MAD value. Prevents division by zero in
            z-score computation and avoids flagging in very sparse regions.

    Returns:
        Callable ``log_E_query → MAD values``, or ``None`` if fewer
        than 10 data points.
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import median_filter

    log_E = np.asarray(log_E, dtype=float).ravel()
    log_sigma = np.asarray(log_sigma, dtype=float).ravel()

    if len(log_E) < 10:
        return None

    # Filter NaN/inf
    valid = np.isfinite(log_E) & np.isfinite(log_sigma)
    log_E = log_E[valid]
    log_sigma = log_sigma[valid]

    if len(log_E) < 10:
        return None

    # Sort by energy
    order = np.argsort(log_E)
    x = log_E[order]
    r = (log_sigma - mean_fn(log_E))[order]

    n = len(x)

    # Use existing _compute_rolling_mad() helper
    mad = _compute_rolling_mad(x, r, window_fraction, min_window_points)

    # Apply floor
    mad = np.maximum(mad, mad_floor)

    # Smooth with median filter to avoid sharp jumps
    smooth_window = max(5, n // 20)
    if smooth_window % 2 == 0:
        smooth_window += 1
    mad_smooth = median_filter(mad, size=smooth_window, mode='nearest')
    mad_smooth = np.maximum(mad_smooth, mad_floor)

    # Deduplicate x values (interp1d requires strictly increasing x)
    # Average MAD values at duplicate x positions
    x_unique, indices = np.unique(x, return_inverse=True)
    mad_unique = np.zeros(len(x_unique))
    counts = np.zeros(len(x_unique))
    np.add.at(mad_unique, indices, mad_smooth)
    np.add.at(counts, indices, 1)
    mad_unique /= counts

    if len(x_unique) < 2:
        const_mad = float(mad_unique[0])
        return lambda log_E_q: np.full(
            np.asarray(log_E_q).shape, max(const_mad, mad_floor)
        )

    # Build interpolator
    interp = interp1d(
        x_unique, mad_unique, kind='linear',
        bounds_error=False,
        fill_value=(mad_unique[0], mad_unique[-1]),
    )

    floor = mad_floor  # capture in closure

    def _mad_fn(log_E_query: np.ndarray) -> np.ndarray:
        result = interp(np.asarray(log_E_query, dtype=float).ravel())
        return np.maximum(result, floor)

    return _mad_fn
