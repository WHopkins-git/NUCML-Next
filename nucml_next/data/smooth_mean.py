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


# ---------------------------------------------------------------------------
# Data-driven lengthscale estimation
# ---------------------------------------------------------------------------

def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    """Inverse of softplus: log(exp(x) - 1).  Numerically stable.

    For large x, softplus(x) ≈ x, so softplus_inverse(x) ≈ x.
    For small x, use log(expm1(x)) directly.

    Args:
        x: Input values (must be > 0 for valid output).

    Returns:
        Array of same shape with softplus_inverse(x).
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 20, x, np.log(np.expm1(np.clip(x, 1e-10, 20))))


def compute_lengthscale_from_residuals(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    window_fraction: float = 0.1,
    min_window_points: int = 15,
    smoothing_factor: float = 1.0,
    min_lengthscale: float = 0.02,
    max_lengthscale: float = 2.0,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Compute energy-dependent lengthscale from local residual variability.

    Estimates the Gibbs kernel lengthscale at each energy from how volatile
    the smooth-mean residuals are locally.  Where residuals are noisy
    (e.g. the resolved resonance region), the lengthscale is short; where
    residuals are smooth (thermal, continuum), the lengthscale is long.

    The returned callable maps ``log₁₀(E [eV])`` to a *signal* value such
    that ``softplus(signal + a₀ + a₁·log_E)`` yields the desired
    lengthscale.  With default Gibbs corrections ``a₀ = a₁ = 0``, this
    gives ``softplus(softplus_inverse(ℓ)) = ℓ`` exactly.

    Args:
        log_E: log₁₀(Energy) values, shape (n,).
        log_sigma: log₁₀(CrossSection) values, shape (n,).
        mean_fn: Smooth mean function (from ``fit_smooth_mean``).
        window_fraction: Half-width of the rolling window as a fraction
            of the total energy range (in log₁₀ decades).
        min_window_points: Minimum number of neighbours in a window.
            If the energy window contains fewer, expand to k-nearest.
        smoothing_factor: Controls the MAD → lengthscale mapping steepness.
            Higher values make the lengthscale drop faster with variability.
        min_lengthscale: Floor for the output lengthscale.
        max_lengthscale: Ceiling for the output lengthscale.

    Returns:
        Callable ``log_E_query → signal_values`` compatible with
        ``GibbsKernel._compute_lengthscales()``, or ``None`` if the data
        is too sparse (fewer than 10 points).
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import median_filter

    log_E = np.asarray(log_E, dtype=float).ravel()
    log_sigma = np.asarray(log_sigma, dtype=float).ravel()

    # Edge case: too few points
    if len(log_E) < 10:
        return None

    # Filter NaN/inf
    valid = np.isfinite(log_E) & np.isfinite(log_sigma)
    log_E = log_E[valid]
    log_sigma = log_sigma[valid]

    if len(log_E) < 10:
        return None

    # 1. Sort by energy
    order = np.argsort(log_E)
    x = log_E[order]
    y = log_sigma[order]

    # 2. Compute residuals from smooth mean
    r = y - mean_fn(x)

    n = len(x)

    # 3. Rolling MAD in adaptive energy window
    energy_range = x[-1] - x[0]
    half_width = max(energy_range * window_fraction, 0.1)

    mad = np.empty(n)
    for i in range(n):
        # Energy-based window
        lo = x[i] - half_width
        hi = x[i] + half_width
        mask = (x >= lo) & (x <= hi)
        n_in_window = mask.sum()

        if n_in_window < min_window_points:
            # Expand to k-nearest neighbours
            distances = np.abs(x - x[i])
            k = min(min_window_points, n)
            idx = np.argpartition(distances, k)[:k]
            r_local = r[idx]
        else:
            r_local = r[mask]

        # MAD with consistency constant
        med_local = np.median(r_local)
        mad[i] = np.median(np.abs(r_local - med_local)) * 1.4826

    # 4. MAD → lengthscale (inverse relationship)
    positive_mad = mad[mad > 0]
    if len(positive_mad) == 0:
        # All identical residuals → uniform max lengthscale
        signal = np.full(n, _softplus_inverse(np.array([max_lengthscale]))[0])
        interpolator = interp1d(
            x, signal,
            kind='linear',
            fill_value=(signal[0], signal[-1]),
            bounds_error=False,
        )
        return interpolator

    median_mad = np.median(positive_mad)
    ell = max_lengthscale * np.exp(-smoothing_factor * mad / median_mad)
    ell = np.clip(ell, min_lengthscale, max_lengthscale)

    # 5. Smooth the lengthscale profile with median filter
    # Use a wide window (≈10% of points, min 3, max 51, must be odd)
    filter_size = max(3, min(51, n // 10))
    if filter_size % 2 == 0:
        filter_size += 1
    ell_smooth = median_filter(ell, size=filter_size)

    # 6. Convert to softplus-compatible signal
    signal = _softplus_inverse(ell_smooth)

    # 7. Build interpolator
    # De-duplicate x to avoid interp1d errors with repeated energies
    x_unique, indices = np.unique(x, return_index=True)
    signal_unique = signal[indices]

    if len(x_unique) < 2:
        # Degenerate: only one unique energy
        const_signal = float(signal_unique[0])
        return lambda log_E_q: np.full(np.asarray(log_E_q).shape, const_signal)

    interpolator = interp1d(
        x_unique, signal_unique,
        kind='linear',
        fill_value=(signal_unique[0], signal_unique[-1]),
        bounds_error=False,
    )

    return interpolator
