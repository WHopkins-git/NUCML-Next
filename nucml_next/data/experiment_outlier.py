"""
Outlier Detection for EXFOR Cross-Section Data
===============================================

Scores every data point using a smooth mean + rolling MAD approach:

    z_score = |residual from smooth mean| / local_MAD

The smooth mean can optionally be weighted by reported EXFOR measurement
uncertainties (1/σ²) to pull the consensus toward precise measurements.
The z-score denominator is always the local MAD — no measurement uncertainty
in the denominator.

No GP fitting, O(n log n) per group, parallelised across CPU cores.

Key Classes:
    ExperimentOutlierConfig: Configuration dataclass (thresholds, parallelism)
    ExperimentOutlierDetector: Main detector with score_dataframe() API

Output columns:
    - experiment_outlier: bool - Entire EXFOR Entry flagged as discrepant
    - point_outlier: bool - Individual point anomalous
    - z_score: float - Continuous anomaly score
    - gp_mean: float - Smooth mean value (column name kept for compatibility)
    - gp_std: float - Local MAD (column name kept for compatibility)
    - calibration_metric: float - NaN (reserved for compatibility)
    - outlier_probability: float - NaN (reserved for compatibility)
    - experiment_id: str - EXFOR Entry identifier

Usage:
    >>> from nucml_next.data.experiment_outlier import (
    ...     ExperimentOutlierDetector, ExperimentOutlierConfig,
    ... )
    >>> config = ExperimentOutlierConfig()
    >>> detector = ExperimentOutlierDetector(config)
    >>> df_scored = detector.score_dataframe(df)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_weights_for_worker(group_df):
    """Compute inverse-variance weights for smooth mean fitting (worker version).

    Module-level function for pickle compatibility with ProcessPoolExecutor.
    Returns None if <10% of points have uncertainty data.
    """
    if 'Uncertainty' not in group_df.columns or 'CrossSection' not in group_df.columns:
        return None

    xs = group_df['CrossSection'].values
    unc = group_df['Uncertainty'].values

    with np.errstate(divide='ignore', invalid='ignore'):
        has_unc = np.isfinite(unc) & (unc > 0) & (xs > 0)
        rel_unc = np.where(has_unc, unc / xs, np.nan)

    if has_unc.mean() < 0.10:
        return None

    log_unc = 0.434 * rel_unc
    log_unc_clipped = np.where(
        np.isfinite(log_unc),
        np.clip(log_unc, 0.005, 0.5),
        np.nan,
    )
    weights = np.where(
        np.isfinite(log_unc_clipped),
        1.0 / log_unc_clipped**2,
        1.0,
    )
    median_w = np.median(weights)
    weights = np.clip(weights, median_w / 100, median_w * 100)
    return weights


def _local_mad_worker(args):
    """Process one (Z,A,MT) group with local_mad scoring.

    Module-level function (not a method) so it can be pickled by
    ``concurrent.futures.ProcessPoolExecutor``.

    Args:
        args: Tuple of (group_df, experiment_ids, config_dict) where
            group_df is a DataFrame for one group, experiment_ids is a
            list of Entry strings, and config_dict holds threshold values.

    Returns:
        Tuple of (index, scored_dict) where index is the DataFrame index
        and scored_dict maps column names to numpy arrays.
    """
    (group_df, experiment_ids, config_dict) = args

    from nucml_next.data.smooth_mean import (
        fit_smooth_mean, compute_rolling_mad_interpolator, SmoothMeanConfig,
    )

    log_E_all = group_df['log_E'].values
    log_sigma_all = group_df['log_sigma'].values

    # 1. Fit smooth mean (always spline for local_mad)
    #    Optionally weight by 1/σ² from reported uncertainties
    weights = None
    if config_dict.get('use_uncertainty_weights', True):
        weights = _compute_weights_for_worker(group_df)

    mean_fn = fit_smooth_mean(
        log_E_all, log_sigma_all,
        SmoothMeanConfig(smooth_mean_type='spline'),
        weights=weights,
    )

    # 2. Rolling MAD
    mad_fn = compute_rolling_mad_interpolator(
        log_E_all, log_sigma_all, mean_fn,
        window_fraction=config_dict['mad_window_fraction'],
        min_window_points=config_dict['mad_min_window_points'],
        mad_floor=config_dict['mad_floor'],
    )

    n = len(group_df)

    if mad_fn is None:
        # Too few points — simple MAD fallback
        center = np.median(log_sigma_all)
        scale = np.median(np.abs(log_sigma_all - center)) * 1.4826
        if scale < 1e-10:
            scale = 1e-6
        z_scores = np.abs(log_sigma_all - center) / scale
        return group_df.index, {
            'gp_mean': np.full(n, center),
            'gp_std': np.full(n, scale),
            'z_score': z_scores,
            'point_outlier': z_scores > config_dict['point_z_threshold'],
            'experiment_outlier': np.zeros(n, dtype=bool),
            'calibration_metric': np.full(n, np.nan),
            'outlier_probability': np.full(n, np.nan),
        }

    # 3. Score every point: z = |residual| / local_MAD
    residuals = log_sigma_all - mean_fn(log_E_all)
    local_mad = mad_fn(log_E_all)
    z_scores = np.abs(residuals) / local_mad
    point_outlier = z_scores > config_dict['point_z_threshold']

    # 4. Experiment discrepancy
    experiment_outlier = np.zeros(n, dtype=bool)
    n_experiments = len(experiment_ids)

    if 'experiment_id' in group_df.columns:
        exp_id_col = group_df['experiment_id'].values
    else:
        exp_id_col = group_df.get('Entry', pd.Series(['unknown'] * n)).values

    for entry_id in experiment_ids:
        exp_mask = exp_id_col == entry_id
        exp_z = z_scores[exp_mask]
        if len(exp_z) == 0:
            continue
        n_bad = np.sum(exp_z > config_dict['exp_z_threshold'])
        fraction_bad = n_bad / len(exp_z)
        if n_experiments > 1 and fraction_bad > config_dict['exp_fraction_threshold']:
            experiment_outlier[exp_mask] = True

    return group_df.index, {
        'gp_mean': mean_fn(log_E_all),
        'gp_std': local_mad,
        'z_score': z_scores,
        'point_outlier': point_outlier,
        'experiment_outlier': experiment_outlier,
        'calibration_metric': np.full(n, np.nan),
        'outlier_probability': np.full(n, np.nan),
    }


@dataclass
class ExperimentOutlierConfig:
    """Configuration for outlier detection on EXFOR cross-section data.

    Attributes:
        point_z_threshold: Z-score threshold for individual point outliers.
        min_group_size: Minimum points in (Z, A, MT) group for rolling MAD.
            Below this, uses simple MAD fallback.
        entry_column: Column name containing EXFOR Entry identifier.
        n_workers: Number of parallel workers. None or -1 = auto (half of
            logical cores). 0 = all logical cores. Positive int = exact count.
        mad_window_fraction: Rolling window as fraction of energy range.
        mad_min_window_points: Minimum points per MAD window.
        mad_floor: Minimum MAD value (prevents div-by-zero in flat regions).
        use_uncertainty_weights: When True, weight the smooth mean fit by
            1/σ² from reported EXFOR uncertainties (improves consensus line).
        exp_z_threshold: Z-score threshold for counting "bad" points
            within an experiment.
        exp_fraction_threshold: Fraction of bad points above which an
            entire experiment is flagged as discrepant.
    """
    point_z_threshold: float = 3.0
    min_group_size: int = 10
    entry_column: str = 'Entry'
    n_workers: Optional[int] = None
    # --- Local MAD scoring ---
    mad_window_fraction: float = 0.1
    mad_min_window_points: int = 15
    mad_floor: float = 0.02  # ~5% relative — minimum plausible measurement scatter
    use_uncertainty_weights: bool = True  # Weight smooth mean by 1/σ² when available
    # --- Experiment discrepancy thresholds ---
    exp_z_threshold: float = 3.0
    exp_fraction_threshold: float = 0.30


class ExperimentOutlierDetector:
    """Outlier detector using smooth mean + local MAD scoring.

    Scores every data point in a DataFrame grouped by (Z, A, MT):
    1. Fits a smooth spline mean on pooled data (optionally uncertainty-weighted)
    2. Computes rolling MAD of residuals in energy windows
    3. Assigns z-scores: |residual| / local_MAD
    4. Flags point outliers by z-score threshold
    5. Flags experiment discrepancy by fraction of flagged points

    Args:
        config: Configuration for detector parameters.

    Example:
        >>> detector = ExperimentOutlierDetector()
        >>> df_scored = detector.score_dataframe(df)
        >>> discrepant = df_scored[df_scored['experiment_outlier']]
        >>> point_outliers = df_scored[df_scored['point_outlier']]
    """

    def __init__(self, config: ExperimentOutlierConfig = None):
        if config is None:
            config = ExperimentOutlierConfig()
        self.config = config

        # Statistics tracking
        self._stats = {
            'mad_groups': 0,
            'discrepant_experiments': 0,
            'total_points': 0,
            'total_groups': 0,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all data points with local MAD z-scores.

        Groups data by (Z, A, MT), fits smooth mean + rolling MAD per group,
        and flags discrepant experiments.

        Adds columns:
        - log_E: log10(Energy)
        - log_sigma: log10(CrossSection)
        - gp_mean: Smooth mean in log10 space (name kept for compatibility)
        - gp_std: Local MAD (name kept for compatibility)
        - z_score: |log_sigma - gp_mean| / local_MAD
        - experiment_outlier: bool - entire experiment is discrepant
        - point_outlier: bool - individual point is anomalous
        - calibration_metric: float - NaN (compatibility column)
        - outlier_probability: float - NaN (compatibility column)
        - experiment_id: str - EXFOR Entry identifier

        Args:
            df: DataFrame with columns: Z, A, MT, Energy, CrossSection
                Optional: Entry (or entry_column), Uncertainty

        Returns:
            DataFrame with additional scoring columns.
        """
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # Validate required columns
        required = ['Z', 'A', 'MT', 'Energy', 'CrossSection']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Pre-compute log values
        result = df.copy()
        result['log_E'] = np.log10(result['Energy'].clip(lower=1e-30))
        result['log_sigma'] = np.log10(result['CrossSection'].clip(lower=1e-30))

        # Initialize output columns
        result['gp_mean'] = np.nan
        result['gp_std'] = np.nan
        result['z_score'] = np.nan
        result['experiment_outlier'] = False
        result['point_outlier'] = False
        result['calibration_metric'] = np.nan
        result['outlier_probability'] = np.nan
        result['experiment_id'] = ''

        # Determine experiment identifier column
        entry_col = self._get_entry_column(df)
        if entry_col:
            result['experiment_id'] = df[entry_col].astype(str)
        else:
            result['experiment_id'] = 'unknown'

        # Group by (Z, A, MT) — include Projectile when available
        group_cols = ['Z', 'A', 'MT']
        if 'Projectile' in result.columns and result['Projectile'].notna().any():
            group_cols.append('Projectile')

        groups = list(result.groupby(group_cols))
        n_groups = len(groups)
        self._stats['total_groups'] = n_groups
        self._stats['total_points'] = len(df)

        logger.info(
            f"Outlier detection: {n_groups:,} groups "
            f"(groupby {group_cols}), {len(df):,} points"
        )

        import os
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        n_workers = self._resolve_n_workers(self.config.n_workers)

        # Only use multiprocessing when it's worth the overhead
        # (process spawn on Windows is expensive; need enough groups)
        use_parallel = n_workers > 1 and n_groups >= 4

        config_dict = {
            'mad_window_fraction': self.config.mad_window_fraction,
            'mad_min_window_points': self.config.mad_min_window_points,
            'mad_floor': self.config.mad_floor,
            'use_uncertainty_weights': self.config.use_uncertainty_weights,
            'point_z_threshold': self.config.point_z_threshold,
            'exp_z_threshold': self.config.exp_z_threshold,
            'exp_fraction_threshold': self.config.exp_fraction_threshold,
        }

        # Partition groups into MAD-fallback (tiny) and main work items
        work_items = []
        for group_key, group_df in groups:
            if len(group_df) < self.config.min_group_size:
                self._stats['mad_groups'] += 1
                scored = self._score_with_mad(group_df.copy())
                for col in ['gp_mean', 'gp_std', 'z_score', 'experiment_outlier',
                           'point_outlier', 'calibration_metric',
                           'outlier_probability', 'experiment_id']:
                    if col in scored.columns:
                        result.loc[scored.index, col] = scored[col].values
            else:
                experiments = self._partition_by_experiment(group_df)
                experiment_ids = list(experiments.keys())
                work_items.append((group_df, experiment_ids, config_dict))

        n_work = len(work_items)
        total_cores = os.cpu_count() or 4
        if use_parallel:
            logger.info(
                f"Local MAD parallel: {n_work} groups on {n_workers} workers "
                f"(from {total_cores} logical cores)"
            )
        else:
            logger.info(
                f"Local MAD sequential: {n_work} groups "
                f"(n_workers={n_workers}, n_groups={n_groups})"
            )

        if n_work > 0:
            if use_parallel:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = pool.map(_local_mad_worker, work_items)
                    if has_tqdm:
                        futures = tqdm(futures, total=n_work,
                                       desc="Local MAD scoring")
                    for idx, scored_dict in futures:
                        for col, values in scored_dict.items():
                            result.loc[idx, col] = values
            else:
                # Sequential fallback (small jobs / n_workers=1)
                seq_iter = (
                    _local_mad_worker(item) for item in work_items
                )
                if has_tqdm:
                    seq_iter = tqdm(seq_iter, total=n_work,
                                    desc="Local MAD scoring")
                for idx, scored_dict in seq_iter:
                    for col, values in scored_dict.items():
                        result.loc[idx, col] = values

        # Log summary and return
        n_discrepant = int(result['experiment_outlier'].sum())
        n_point_outliers = int(result['point_outlier'].sum())
        logger.info(
            f"Local MAD scoring complete: "
            f"{n_work} groups scored, "
            f"{n_point_outliers:,} point outliers, "
            f"{n_discrepant:,} experiment-outlier points"
        )
        return result

    def _get_entry_column(self, df: pd.DataFrame) -> Optional[str]:
        """Determine which column contains the EXFOR Entry identifier."""
        if self.config.entry_column in df.columns:
            return self.config.entry_column

        for col in ['Entry', 'entry', 'ENTRY', 'ExforEntry', 'exfor_entry']:
            if col in df.columns:
                return col

        return None

    def _partition_by_experiment(
        self,
        df_group: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Partition group by experiment (Entry)."""
        if 'experiment_id' not in df_group.columns:
            return {'single': df_group}

        experiments = {}
        for entry_id, exp_df in df_group.groupby('experiment_id'):
            experiments[str(entry_id)] = exp_df

        return experiments

    def _score_with_mad(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """Score using Median Absolute Deviation (fallback for small groups)."""
        result = df_group.copy()
        log_sigma = df_group['log_sigma'].values

        center = np.median(log_sigma)
        mad = np.median(np.abs(log_sigma - center))
        scale = mad * 1.4826  # Consistency constant for normal

        if scale < 1e-10:
            scale = 1e-6

        result['gp_mean'] = center
        result['gp_std'] = scale
        result['z_score'] = np.abs(log_sigma - center) / scale
        result['experiment_outlier'] = False
        result['point_outlier'] = result['z_score'] > self.config.point_z_threshold
        result['calibration_metric'] = np.nan

        return result

    def _score_group_local_mad(
        self,
        df_group: pd.DataFrame,
        experiments: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Score all points in a group using smooth mean + local MAD.

        This method:
        1. Fits a smooth mean on ALL pooled data in the group
           (optionally weighted by 1/σ² from reported uncertainties)
        2. Computes rolling MAD of residuals in energy windows
        3. Assigns z-scores: |residual| / local_MAD
        4. Flags point outliers by z-score threshold
        5. Flags experiment discrepancy by fraction of flagged points

        Args:
            df_group: Full DataFrame for this (Z, A, MT[, Projectile]) group.
            experiments: Dict mapping Entry ID to experiment DataFrames.

        Returns:
            DataFrame with scoring columns populated.
        """
        result = df_group.copy()

        log_E_all = result['log_E'].values
        log_sigma_all = result['log_sigma'].values

        # 1. Fit smooth mean on pooled data (always spline)
        #    Optionally weight by 1/σ² from reported uncertainties
        from nucml_next.data.smooth_mean import (
            fit_smooth_mean, compute_rolling_mad_interpolator, SmoothMeanConfig,
        )

        if self.config.use_uncertainty_weights:
            weights = self._compute_smooth_mean_weights(result)
        else:
            weights = None

        mean_fn = fit_smooth_mean(
            log_E_all, log_sigma_all,
            SmoothMeanConfig(smooth_mean_type='spline'),
            weights=weights,
        )

        # 2. Compute residuals and rolling MAD
        mad_fn = compute_rolling_mad_interpolator(
            log_E_all, log_sigma_all, mean_fn,
            window_fraction=self.config.mad_window_fraction,
            min_window_points=self.config.mad_min_window_points,
            mad_floor=self.config.mad_floor,
        )

        if mad_fn is None:
            return self._score_with_mad(result)

        # 3. Score every point: z = |residual| / local_MAD
        residuals = log_sigma_all - mean_fn(log_E_all)
        local_mad = mad_fn(log_E_all)
        z_scores = np.abs(residuals) / local_mad

        result['gp_mean'] = mean_fn(log_E_all)
        result['gp_std'] = local_mad
        result['z_score'] = z_scores

        # 4. Flag point outliers
        result['point_outlier'] = z_scores > self.config.point_z_threshold

        # 5. Flag experiment discrepancy
        for entry_id, exp_df in experiments.items():
            exp_mask = result.index.isin(exp_df.index)
            exp_z_scores = result.loc[exp_mask, 'z_score'].values

            n_total = len(exp_z_scores)
            if n_total == 0:
                continue

            n_bad = np.sum(exp_z_scores > self.config.exp_z_threshold)
            fraction_bad = n_bad / n_total

            # Single-experiment groups can't be flagged as discrepant
            if len(experiments) == 1:
                is_discrepant = False
            else:
                is_discrepant = fraction_bad > self.config.exp_fraction_threshold

            result.loc[exp_mask, 'experiment_outlier'] = is_discrepant

            if is_discrepant:
                self._stats['discrepant_experiments'] += 1

        # 6. Ensure compatibility columns
        result['calibration_metric'] = np.nan
        result['outlier_probability'] = np.nan

        return result

    @staticmethod
    def _resolve_n_workers(n_workers: Optional[int]) -> int:
        """Resolve worker count for parallel dispatch.

        Args:
            n_workers: None or -1 = auto (half of logical cores).
                       0 = all logical cores.
                       Positive int = exact count.

        Returns:
            Resolved number of workers (always >= 1).
        """
        import os
        total = os.cpu_count() or 4
        if n_workers is None or n_workers == -1:
            return max(1, total // 2)
        elif n_workers == 0:
            return total
        else:
            return max(1, n_workers)

    def _compute_smooth_mean_weights(
        self, df: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Compute inverse-variance weights from EXFOR uncertainties.

        Returns weights proportional to 1/σ² in log₁₀ space, or None if
        fewer than 10% of points have reported uncertainties.

        Points without reported uncertainty get weight = 1.0 (neutral).
        Points with reported uncertainty get weight = 1/σ_log² (capped).

        Returns:
            Array of weights shape (n,), or None if too few points have
            uncertainty to make weighting meaningful.
        """
        if 'Uncertainty' not in df.columns or 'CrossSection' not in df.columns:
            return None

        xs = df['CrossSection'].values
        unc = df['Uncertainty'].values

        with np.errstate(divide='ignore', invalid='ignore'):
            has_unc = np.isfinite(unc) & (unc > 0) & (xs > 0)
            rel_unc = np.where(has_unc, unc / xs, np.nan)

        # If fewer than 10% have uncertainty, don't weight (not enough info)
        if has_unc.mean() < 0.10:
            return None

        # Convert to log₁₀ space: σ_log = 0.434 * (dσ/σ)
        log_unc = 0.434 * rel_unc

        # Inverse variance weights, capped to avoid extreme values
        # Floor σ_log at 0.005 (0.5% relative) — prevents absurd weights
        # Ceiling σ_log at 0.5 (100% relative) — prevents zero weights
        log_unc_clipped = np.where(
            np.isfinite(log_unc),
            np.clip(log_unc, 0.005, 0.5),
            np.nan,
        )

        weights = np.where(
            np.isfinite(log_unc_clipped),
            1.0 / log_unc_clipped**2,
            1.0,  # Default weight for missing uncertainty
        )

        # Cap extreme weight ratios to prevent single points dominating
        median_w = np.median(weights)
        weights = np.clip(weights, median_w / 100, median_w * 100)

        return weights

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return self._stats.copy()
