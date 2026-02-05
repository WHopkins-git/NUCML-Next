"""
Shared uncertainty-based sample weight computation for evaluators.

Supports three weighting modes:
    - None: No weighting (equal weight for all samples)
    - 'xs': Inverse-variance weighting using cross-section uncertainty only
    - 'both': Combined weighting using both cross-section AND energy uncertainty

When the training pipeline uses a log-transform on the target (log_target=True),
uncertainties are propagated into log-space via error propagation before computing
inverse-variance weights. This prevents raw-space uncertainties from creating
extreme weight ratios that collapse tree-based models.

Log-space propagation
---------------------
For a measurement σ ± d_sigma, the uncertainty in log10(σ) is:

    δ(log10 σ) = d_sigma / (σ · ln 10)

So the inverse-variance weight in log-space is:

    w = 1 / δ(log10 σ)² = (σ · ln 10)² / d_sigma²

This is equivalent to weighting by *relative* precision (d_sigma/σ)⁻², which
is the statistically correct weight when the loss is computed in log-space.
"""

from typing import Optional
import numpy as np
import pandas as pd


# Valid values for use_uncertainty_weights parameter
VALID_WEIGHT_MODES = {None, 'xs', 'both'}


def normalize_weight_mode(mode: Optional[str]) -> Optional[str]:
    """
    Normalize use_uncertainty_weights to a canonical string or None.

    Args:
        mode: Weight mode – None, 'xs', or 'both'

    Returns:
        None, 'xs', or 'both'

    Raises:
        ValueError: If mode is not valid
    """
    if mode is None:
        return None
    if mode in ('xs', 'both'):
        return mode
    raise ValueError(
        f"use_uncertainty_weights must be one of {VALID_WEIGHT_MODES}, "
        f"got {mode!r}"
    )


def compute_sample_weights(
    df: pd.DataFrame,
    mode: Optional[str],
    uncertainty_column: str = 'Uncertainty',
    energy_uncertainty_column: str = 'Energy_Uncertainty',
    missing_handling: str = 'median',
    target_column: str = 'CrossSection',
    log_target: bool = False,
    energy_column: str = 'Energy',
    log_energy: bool = False,
) -> Optional[np.ndarray]:
    """
    Compute sample weights from experimental uncertainties.

    When *log_target* is True, cross-section uncertainties are propagated
    into log-space before computing inverse-variance weights. This ensures
    weights reflect precision in the space where the loss is actually
    computed, preventing extreme weight ratios that collapse tree-based
    models.

    Args:
        df: DataFrame containing the data
        mode: Weighting mode (None, 'xs', or 'both').
              Use :func:`normalize_weight_mode` first.
        uncertainty_column: Cross-section uncertainty column name
        energy_uncertainty_column: Energy uncertainty column name
        missing_handling: Strategy for missing uncertainties
            ('median', 'equal', 'exclude')
        target_column: Cross-section column (needed for log-space propagation)
        log_target: Whether the training pipeline log-transforms the target.
            When True, uncertainties are propagated to log-space.
        energy_column: Energy column (needed for log-space propagation of
            energy uncertainty)
        log_energy: Whether the training pipeline log-transforms energy.

    Returns:
        Normalized sample weight array (mean=1), or None if mode is None

    Notes:
        Without log_target (raw-space training):
            - 'xs' mode:   w_i = 1 / d_sigma_i²
            - 'both' mode:  w_i = 1 / (d_sigma_i² · δE_i²)

        With log_target (log-space training):
            - 'xs' mode:   w_i = (σ_i · ln10)² / d_sigma_i²  =  1 / (d_sigma_i/σ_i)²·(1/ln10²)
            - 'both' mode:  product of XS and energy log-space weights

        Weights are normalized to mean=1 for numerical stability.
        Extreme outlier weights are clipped at the 99th percentile.
    """
    if mode is None:
        return None

    n_total = len(df)

    # --- Cross-section uncertainty weights ---
    xs_weights = _compute_xs_weight(
        df, uncertainty_column, target_column, n_total,
        missing_handling, log_target,
    )

    if xs_weights is None:
        print(f"  WARNING: No valid cross-section uncertainties found. "
              f"Disabling sample weighting.")
        return None

    if mode == 'xs':
        return _normalize_weights(xs_weights, label="Cross-section")

    # --- 'both' mode: also compute energy uncertainty weights ---
    energy_weights = _compute_energy_weight(
        df, energy_uncertainty_column, energy_column, n_total,
        missing_handling, log_energy,
    )

    if energy_weights is None:
        print(f"  WARNING: No valid energy uncertainties found. "
              f"Using cross-section uncertainty only.")
        return _normalize_weights(xs_weights, label="Cross-section")

    # Combine: product of independent inverse-variance weights
    combined = xs_weights * energy_weights
    return _normalize_weights(combined, label="Combined (XS + Energy)")


def _compute_xs_weight(
    df: pd.DataFrame,
    uncertainty_column: str,
    target_column: str,
    n_total: int,
    missing_handling: str,
    log_target: bool,
) -> Optional[np.ndarray]:
    """
    Compute inverse-variance weights for cross-section uncertainty.

    When log_target is True, propagates absolute uncertainty into log10 space:
        δ(log10 σ) = d_sigma / (σ · ln 10)
        w = 1 / δ(log10 σ)² = (σ · ln 10)² / d_sigma²
    """
    if uncertainty_column not in df.columns:
        print(f"  WARNING: Cross-section uncertainty column "
              f"'{uncertainty_column}' not found.")
        return None

    unc = df[uncertainty_column].values.copy().astype(float)
    valid_mask = np.isfinite(unc) & (unc > 0)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return None

    weights = np.ones(n_total, dtype=float)

    if log_target and target_column in df.columns:
        # Propagate uncertainty into log10 space
        xs = df[target_column].values.copy().astype(float)
        # Need valid cross-section AND valid uncertainty
        both_valid = valid_mask & np.isfinite(xs) & (xs > 0)
        n_both = both_valid.sum()

        if n_both > 0:
            # δ(log10 σ) = d_sigma / (σ · ln10)
            # w = 1 / δ(log10 σ)² = (σ · ln10)² / d_sigma²
            ln10 = np.log(10.0)
            log_unc = unc[both_valid] / (xs[both_valid] * ln10)
            weights[both_valid] = 1.0 / (log_unc ** 2)

            # Samples with valid uncertainty but invalid XS get default weight
            unc_only = valid_mask & ~both_valid
            if unc_only.any():
                weights[unc_only] = 1.0 / (unc[unc_only] ** 2)

            print(f"  Cross-section uncertainty propagated to log10 space")
            print(f"    d(log10 xs) = d_xs / (xs * ln10)")
            valid_mask = both_valid  # update for counting
            n_valid = n_both
        else:
            # Fallback: raw inverse-variance
            weights[valid_mask] = 1.0 / (unc[valid_mask] ** 2)
    else:
        # Raw-space: w = 1 / d_sigma²
        weights[valid_mask] = 1.0 / (unc[valid_mask] ** 2)

    # Handle missing values
    _handle_missing(weights, valid_mask, n_total, missing_handling,
                    label="cross-section")

    print(f"  Cross-section uncertainty: "
          f"{n_valid:,} / {n_total:,} valid ({100*n_valid/n_total:.1f}%)")

    return weights


def _compute_energy_weight(
    df: pd.DataFrame,
    energy_uncertainty_column: str,
    energy_column: str,
    n_total: int,
    missing_handling: str,
    log_energy: bool,
) -> Optional[np.ndarray]:
    """
    Compute inverse-variance weights for energy uncertainty.

    When log_energy is True, propagates absolute uncertainty into log10 space:
        δ(log10 E) = δE / (E · ln 10)
        w = 1 / δ(log10 E)²
    """
    if energy_uncertainty_column not in df.columns:
        print(f"  WARNING: Energy uncertainty column "
              f"'{energy_uncertainty_column}' not found.")
        return None

    unc = df[energy_uncertainty_column].values.copy().astype(float)
    valid_mask = np.isfinite(unc) & (unc > 0)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return None

    weights = np.ones(n_total, dtype=float)

    if log_energy and energy_column in df.columns:
        energy = df[energy_column].values.copy().astype(float)
        both_valid = valid_mask & np.isfinite(energy) & (energy > 0)
        n_both = both_valid.sum()

        if n_both > 0:
            ln10 = np.log(10.0)
            log_unc = unc[both_valid] / (energy[both_valid] * ln10)
            weights[both_valid] = 1.0 / (log_unc ** 2)

            unc_only = valid_mask & ~both_valid
            if unc_only.any():
                weights[unc_only] = 1.0 / (unc[unc_only] ** 2)

            print(f"  Energy uncertainty propagated to log10 space")
            valid_mask = both_valid
            n_valid = n_both
        else:
            weights[valid_mask] = 1.0 / (unc[valid_mask] ** 2)
    else:
        weights[valid_mask] = 1.0 / (unc[valid_mask] ** 2)

    _handle_missing(weights, valid_mask, n_total, missing_handling,
                    label="energy")

    print(f"  Energy uncertainty: "
          f"{n_valid:,} / {n_total:,} valid ({100*n_valid/n_total:.1f}%)")

    return weights


def _handle_missing(
    weights: np.ndarray,
    valid_mask: np.ndarray,
    n_total: int,
    missing_handling: str,
    label: str,
) -> None:
    """Apply missing-value strategy to weight array (in-place)."""
    n_missing = n_total - valid_mask.sum()
    if n_missing > 0:
        if missing_handling == 'median':
            median_w = np.median(weights[valid_mask])
            weights[~valid_mask] = median_w
            print(f"  Missing {label} uncertainties ({n_missing:,}): "
                  f"assigned median weight")
        elif missing_handling == 'equal':
            weights[~valid_mask] = 1.0
            print(f"  Missing {label} uncertainties ({n_missing:,}): "
                  f"assigned equal weight (1.0)")
        elif missing_handling == 'exclude':
            weights[~valid_mask] = np.nan
            print(f"  Missing {label} uncertainties ({n_missing:,}): "
                  f"will be excluded")


def _normalize_weights(
    weights: np.ndarray,
    label: str = "Sample",
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """Normalize weights to mean=1 with percentile clipping.

    Inverse-variance weights (1/σ²) can span many orders of magnitude
    because EXFOR uncertainties range from microbarns to kilobarns. Without
    clipping, a handful of extremely-high-weight samples dominate the
    splitting criterion in tree-based models, collapsing a max_depth=80
    tree to depth ~7.

    Clipping at the *clip_percentile* (default 99th) caps outlier weights
    before normalization, ensuring the tree sees a representative sample
    distribution while still up-weighting precise measurements.

    Args:
        weights: Raw weight array (may contain NaN for excluded rows).
        label: Label for log messages.
        clip_percentile: Upper percentile at which to cap weights.
            Set to 100.0 to disable clipping.

    Returns:
        Normalized weight array (mean=1 among finite entries).
    """
    valid_mask = np.isfinite(weights)
    n_valid = valid_mask.sum()

    if n_valid > 0:
        valid_w = weights[valid_mask]

        # Clip extreme weights at the upper percentile
        if clip_percentile < 100.0:
            cap = np.percentile(valid_w, clip_percentile)
            n_clipped = (valid_w > cap).sum()
            if n_clipped > 0:
                valid_w = np.clip(valid_w, None, cap)
                weights[valid_mask] = valid_w
                print(f"  {label} weights: clipped {n_clipped:,} values "
                      f"above {clip_percentile}th percentile (cap={cap:.4e})")

        # Normalize to mean=1
        mean_w = valid_w.mean()
        if mean_w > 0:
            weights[valid_mask] = valid_w / mean_w

        final_w = weights[valid_mask]
        print(f"  {label} weights: range [{final_w.min():.4f}, "
              f"{final_w.max():.4f}], mean=1.0")

    return weights
