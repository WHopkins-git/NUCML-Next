"""
Per-Experiment GP Outlier Detection for EXFOR Cross-Section Data
================================================================

Fits independent Exact GPs to each EXFOR experiment (Entry) within a (Z, A, MT)
group, builds consensus from multiple experiment posteriors, and identifies
discrepant experiments.

Key Classes:
    ExperimentOutlierConfig: Configuration dataclass
    ExperimentOutlierDetector: Main detector with score_dataframe() API

Compared to SVGPOutlierDetector:
    - Fits per-experiment (not pooled across all experiments)
    - Uses heteroscedastic noise from measurement uncertainties
    - Calibrates lengthscale via Wasserstein distance
    - Flags entire experiments as discrepant (not just individual points)
    - More robust to resonance structure (no over-smoothing)

Output columns:
    - experiment_outlier: bool - Entire EXFOR Entry flagged as discrepant
    - point_outlier: bool - Individual point anomalous within its experiment
    - z_score: float - Continuous anomaly score (backward compat)
    - calibration_metric: float - Per-experiment Wasserstein distance
    - outlier_probability: float - Per-point outlier probability from
        contaminated normal EM (NaN when contaminated likelihood not used)
    - experiment_id: str - EXFOR Entry identifier
    - log_E, log_sigma, gp_mean, gp_std: float - Backward compat columns

Usage:
    >>> from nucml_next.data.experiment_outlier import ExperimentOutlierDetector
    >>> detector = ExperimentOutlierDetector()
    >>> df_scored = detector.score_dataframe(df)
    >>> # df_scored has experiment_outlier, point_outlier, z_score columns
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from nucml_next.data.experiment_gp import (
    ExactGPExperiment,
    ExactGPExperimentConfig,
    prepare_log_uncertainties,
)
from nucml_next.data.consensus import ConsensusBuilder, ConsensusConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentOutlierConfig:
    """Configuration for per-experiment GP outlier detection.

    Attributes:
        gp_config: Configuration for per-experiment GP fitting.
        consensus_config: Configuration for consensus building.
        point_z_threshold: Z-score threshold for individual point outliers.
        min_group_size: Minimum points in (Z, A, MT) group for GP fitting.
            Below this, uses MAD fallback.
        entry_column: Column name containing EXFOR Entry identifier.
            If not present, falls back to 'Entry' or treats all points
            as single experiment.
        checkpoint_dir: Directory for saving checkpoints (None = no checkpointing).
        checkpoint_interval: Save checkpoint every N groups processed.
        n_workers: Number of parallel workers (None = sequential processing).
        clear_caches_after_group: If True (default), clear _fitted_gps and
            _consensus_builders after each group to save memory. Set False
            for post-hoc diagnostics.
        streaming_output: Optional path to write results incrementally to Parquet
            instead of holding in memory. Significantly reduces peak memory for
            large datasets (>5M points).
        ripl_data_path: Path to RIPL-3 ``levels-param.data`` file.
            Required for Gibbs kernel (``kernel_config.kernel_type='gibbs'``).
            If None, the Gibbs kernel will fall back to RBF.
            Only used when ``gp_config.kernel_config`` is set.
        s_n_column: DataFrame column name for neutron separation energy
            (MeV). If present, used directly; otherwise estimated from
            AME2020 data or a built-in fallback table.
        gibbs_lengthscale_source: Lengthscale source for Gibbs kernel.
            ``'data'`` computes from local residual variability (requires
            ``smooth_mean_type='spline'``).  ``'ripl'`` uses RIPL-3 level
            density (requires ``ripl_data_path``).  ``'auto'`` tries data
            first, falls back to RIPL-3.  Default ``'data'``.
        hierarchical_refitting: Enable two-pass hierarchical fitting (Phase 4).
            When True, Pass 2 extracts group-level hyperparameter statistics
            from Pass 1 fitted GPs, then re-fits each experiment with
            constrained bounds and shared outputscale. Default False.
        min_experiments_for_refit: Minimum number of successfully fitted GPs
            required in a group before hierarchical refitting is attempted.
            With fewer GPs, IQR-based bounds are unreliable. Default 3.
        refit_bounds_iqr_margin: IQR multiplier for computing parameter bounds
            in Pass 2.  Bounds are ``[Q1 - margin*IQR, Q3 + margin*IQR]``.
            Default 1.0.
        refit_share_outputscale: When True (default), Pass 2 sets each GP's
            outputscale to the group median before re-fitting.
    """
    gp_config: ExactGPExperimentConfig = field(default_factory=ExactGPExperimentConfig)
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)
    point_z_threshold: float = 3.0
    min_group_size: int = 10
    entry_column: str = 'Entry'
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 100
    n_workers: Optional[int] = None
    clear_caches_after_group: bool = True
    streaming_output: Optional[str] = None
    ripl_data_path: Optional[str] = None
    s_n_column: Optional[str] = None
    # Data-driven lengthscale source for Gibbs kernel
    gibbs_lengthscale_source: str = 'data'  # 'data' | 'ripl' | 'auto'
    # Phase 4: Hierarchical experiment structure
    hierarchical_refitting: bool = False
    min_experiments_for_refit: int = 3
    refit_bounds_iqr_margin: float = 1.0
    refit_share_outputscale: bool = True
    # --- Scoring method selection ---
    scoring_method: str = 'gp'  # 'gp' | 'local_mad'
    # --- Local MAD scoring (scoring_method='local_mad') ---
    mad_window_fraction: float = 0.1        # Rolling window as fraction of energy range
    mad_min_window_points: int = 15          # Minimum points per MAD window
    mad_floor: float = 0.01                  # Minimum MAD (prevents div-by-zero)
    # --- Experiment discrepancy thresholds ---
    exp_z_threshold: float = 3.0             # z-score threshold for counting "bad" points
    exp_fraction_threshold: float = 0.30     # Fraction of bad points to flag experiment


class ExperimentOutlierDetector:
    """Per-experiment GP outlier detector with consensus-based flagging.

    This detector addresses limitations of pooled SVGP by:
    1. Fitting independent GPs to each EXFOR experiment (Entry)
    2. Building consensus from multiple experiment posteriors
    3. Using heteroscedastic noise from measurement uncertainties
    4. Flagging entire experiments that deviate from consensus

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
            'gp_experiments': 0,          # Experiments fitted with GP
            'small_experiments': 0,        # Experiments too small for GP
            'mad_groups': 0,              # Groups using MAD fallback
            'single_experiment_groups': 0, # Groups with only one experiment
            'consensus_groups': 0,         # Groups with multi-experiment consensus
            'discrepant_experiments': 0,   # Experiments flagged as discrepant
            'total_points': 0,
            'total_groups': 0,
            # Phase 4: Hierarchical refitting
            'hierarchical_refits': 0,          # Experiments successfully refit
            'hierarchical_groups': 0,          # Groups where Pass 2 ran
            'hierarchical_skipped_groups': 0,  # Groups skipped (too few GPs)
        }

        # Cached fitted GPs per group (for diagnostics)
        self._fitted_gps: Dict[Tuple, Dict[str, ExactGPExperiment]] = {}
        self._consensus_builders: Dict[Tuple, ConsensusBuilder] = {}

        # RIPL-3 loader (lazy, only when Gibbs kernel is configured)
        self._ripl_loader = None

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all data points with per-experiment GP z-scores.

        Groups data by (Z, A, MT), partitions by Entry, fits GPs per experiment,
        builds consensus, and flags discrepant experiments.

        Adds columns:
        - log_E: log10(Energy)
        - log_sigma: log10(CrossSection)
        - gp_mean: GP predicted mean in log10 space
        - gp_std: GP predicted std in log10 space
        - z_score: |log_sigma - gp_mean| / gp_std (backward compat)
        - experiment_outlier: bool - entire experiment is discrepant
        - point_outlier: bool - individual point is anomalous
        - calibration_metric: float - per-experiment Wasserstein distance
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
            # No Entry column - assign unique ID per group
            result['experiment_id'] = 'unknown'

        # Group by (Z, A, MT) — include Projectile when available so that
        # different projectile-induced reactions get separate GP fits.
        group_cols = ['Z', 'A', 'MT']
        if 'Projectile' in result.columns and result['Projectile'].notna().any():
            group_cols.append('Projectile')

        groups = list(result.groupby(group_cols))
        n_groups = len(groups)
        self._stats['total_groups'] = n_groups
        self._stats['total_points'] = len(df)

        logger.info(
            f"Experiment outlier detection: {n_groups:,} groups "
            f"(groupby {group_cols}), {len(df):,} points"
        )

        # Check for checkpoint to resume from
        start_idx = 0
        partial_results: Dict[Tuple, pd.DataFrame] = {}
        if self.config.checkpoint_dir:
            start_idx, partial_results = self._load_checkpoint()
            if start_idx > 0:
                logger.info(f"Resuming from checkpoint: group {start_idx}/{n_groups}")

        # Streaming mode setup
        streaming_writer = None
        streaming_chunks: List[pd.DataFrame] = []
        streaming_mode = self.config.streaming_output is not None

        if streaming_mode:
            logger.info(f"Streaming mode enabled: writing to {self.config.streaming_output}")

        # Process groups
        iterator = enumerate(groups)
        if has_tqdm:
            iterator = tqdm(iterator, total=n_groups, desc="Experiment scoring",
                           initial=start_idx)

        for i, (group_key, group_df) in iterator:
            if i < start_idx:
                continue

            # Check if already processed (from checkpoint)
            if group_key in partial_results:
                scored = partial_results[group_key]
            else:
                scored = self._score_group(group_df, group_key)
                # Only accumulate partial_results if checkpointing is enabled
                # (results go directly to result DataFrame on lines below)
                # This prevents unbounded memory growth when checkpointing is disabled
                if not streaming_mode and self.config.checkpoint_dir:
                    partial_results[group_key] = scored

            if streaming_mode:
                # Accumulate scored chunks for streaming write
                streaming_chunks.append(scored)

                # Write to disk every 100 groups to bound memory
                if len(streaming_chunks) >= 100:
                    self._flush_streaming_chunks(streaming_chunks)
                    streaming_chunks.clear()
            else:
                # Update result DataFrame (traditional mode)
                for col in ['gp_mean', 'gp_std', 'z_score', 'experiment_outlier',
                           'point_outlier', 'calibration_metric',
                           'outlier_probability', 'experiment_id']:
                    if col in scored.columns:
                        result.loc[scored.index, col] = scored[col].values

            # Checkpoint (only in non-streaming mode)
            if (not streaming_mode and self.config.checkpoint_dir and
                    (i + 1) % self.config.checkpoint_interval == 0):
                self._save_checkpoint(i + 1, partial_results)

            # GPU memory cleanup (every 10 groups when using CUDA)
            if (self.config.gp_config.device == 'cuda' and
                    (i + 1) % 10 == 0):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

            # Periodic garbage collection to reclaim memory from released objects
            # (every 100 groups to avoid overhead)
            if (i + 1) % 100 == 0:
                import gc
                gc.collect()

            # Progress logging (every 10%)
            if not has_tqdm and n_groups >= 10 and (i + 1) % max(1, n_groups // 10) == 0:
                pct = 100 * (i + 1) / n_groups
                logger.info(
                    f"  Progress: {pct:.0f}% ({i+1}/{n_groups} groups) | "
                    f"GP exps: {self._stats['gp_experiments']}, "
                    f"Discrepant: {self._stats['discrepant_experiments']}"
                )

        # Final operations
        if streaming_mode:
            # Flush any remaining chunks
            if streaming_chunks:
                self._flush_streaming_chunks(streaming_chunks)
                streaming_chunks.clear()

            # Finalize streaming output
            self._finalize_streaming_output()

            # Log summary
            logger.info(
                f"Experiment scoring complete (streaming mode): "
                f"{self._stats['gp_experiments']} GP experiments, "
                f"{self._stats['small_experiments']} small experiments, "
                f"{self._stats['consensus_groups']} consensus groups, "
                f"{self._stats['discrepant_experiments']} discrepant experiments"
            )

            # In streaming mode, return None since results are on disk
            return None
        else:
            # Final checkpoint
            if self.config.checkpoint_dir:
                self._save_checkpoint(n_groups, partial_results)

            # Log summary
            logger.info(
                f"Experiment scoring complete: "
                f"{self._stats['gp_experiments']} GP experiments, "
                f"{self._stats['small_experiments']} small experiments, "
                f"{self._stats['consensus_groups']} consensus groups, "
                f"{self._stats['discrepant_experiments']} discrepant experiments"
            )

            return result

    def _get_entry_column(self, df: pd.DataFrame) -> Optional[str]:
        """Determine which column contains the EXFOR Entry identifier."""
        # Check configured column name
        if self.config.entry_column in df.columns:
            return self.config.entry_column

        # Common alternatives
        for col in ['Entry', 'entry', 'ENTRY', 'ExforEntry', 'exfor_entry']:
            if col in df.columns:
                return col

        return None

    def _score_group(
        self,
        df_group: pd.DataFrame,
        group_key: Tuple[int, int, int],
    ) -> pd.DataFrame:
        """Score a single (Z, A, MT) group.

        Processing logic:
        1. If n < min_group_size: MAD fallback
        2. If only 1 experiment: Fit GP if large enough, else MAD
        3. If >= 2 experiments: Build consensus, flag discrepant ones

        Args:
            df_group: DataFrame for one (Z, A, MT) group
            group_key: (Z, A, MT) tuple for caching

        Returns:
            DataFrame with scoring columns filled
        """
        n = len(df_group)
        result = df_group.copy()

        # Get experiment partitions
        experiments = self._partition_by_experiment(df_group)
        n_experiments = len(experiments)

        log_E = df_group['log_E'].values
        log_sigma = df_group['log_sigma'].values

        # Case 0: Local MAD scoring (bypasses GP entirely)
        if self.config.scoring_method == 'local_mad':
            if n < self.config.min_group_size:
                self._stats['mad_groups'] += 1
                return self._score_with_mad(result)
            return self._score_group_local_mad(df_group, experiments)

        # Case 1: Very small group - MAD fallback
        if n < self.config.min_group_size:
            self._stats['mad_groups'] += 1
            return self._score_with_mad(result)

        # Case 2: Single experiment
        if n_experiments == 1:
            self._stats['single_experiment_groups'] += 1
            return self._score_single_experiment(result, experiments, group_key)

        # Case 3: Multiple experiments - build consensus
        self._stats['consensus_groups'] += 1
        return self._score_multi_experiment(result, experiments, group_key)

    def _partition_by_experiment(
        self,
        df_group: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Partition group by experiment (Entry)."""
        if 'experiment_id' not in df_group.columns:
            # No experiment info - treat as single experiment
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

    def _assign_outlier_probabilities(
        self,
        result: pd.DataFrame,
        exp_df: pd.DataFrame,
        gp: 'ExactGPExperiment',
        also_set_point_outlier: bool = False,
    ) -> None:
        """Assign outlier_probability (and optionally point_outlier) from GP.

        Handles subsampled GPs by assigning only to the subsampled rows.
        Non-subsampled rows retain NaN for outlier_probability and their
        existing value for point_outlier (typically z-score-based).
        """
        if gp.outlier_probabilities is None:
            return

        if gp.subsample_indices is not None:
            idx = exp_df.index[gp.subsample_indices]
        else:
            idx = exp_df.index

        result.loc[idx, 'outlier_probability'] = gp.outlier_probabilities
        if also_set_point_outlier:
            result.loc[idx, 'point_outlier'] = gp.outlier_probabilities > 0.5

    def _build_data_lengthscale(self, log_E, log_sigma, mean_fn):
        """Compute data-driven lengthscale interpolator from residuals.

        Returns ``None`` if:
        - ``gibbs_lengthscale_source`` is ``'ripl'`` (explicitly skip data)
        - Smooth mean is not spline (cannot compute residuals)
        - Kernel is not Gibbs
        - Data is too sparse (< 10 points)

        Args:
            log_E: log₁₀(Energy) values.
            log_sigma: log₁₀(CrossSection) values.
            mean_fn: Smooth mean function (from ``fit_smooth_mean``).

        Returns:
            Callable or None.
        """
        source = self.config.gibbs_lengthscale_source
        if source == 'ripl':
            return None  # User explicitly wants RIPL-3 only

        sm_config = self.config.gp_config.smooth_mean_config
        kc = self.config.gp_config.kernel_config
        if (mean_fn is None
                or sm_config is None
                or sm_config.smooth_mean_type != 'spline'
                or kc is None
                or kc.kernel_type != 'gibbs'):
            return None

        from nucml_next.data.smooth_mean import compute_lengthscale_from_residuals
        return compute_lengthscale_from_residuals(log_E, log_sigma, mean_fn)

    def _build_data_outputscale(self, log_E, log_sigma, mean_fn):
        """Compute data-driven outputscale interpolator from residuals.

        Returns ``None`` if:
        - ``gibbs_lengthscale_source`` is ``'ripl'`` (explicitly skip data)
        - Smooth mean is not spline (cannot compute residuals)
        - Kernel type is not 'gibbs' or 'rbf'
        - Data is too sparse (< 10 points)

        Unlike ``_build_data_lengthscale()``, this works with **both** RBF
        and Gibbs kernels — the energy-dependent outputscale is orthogonal
        to the kernel choice.

        The returned callable maps ``log₁₀(E) → σ(E)`` (standard deviation)
        so the kernel uses ``K(xᵢ,xⱼ) = σ(xᵢ)·σ(xⱼ)·K_unit(xᵢ,xⱼ)``,
        giving energy-appropriate prior variance.
        """
        source = self.config.gibbs_lengthscale_source
        if source == 'ripl':
            return None

        sm_config = self.config.gp_config.smooth_mean_config
        kc = self.config.gp_config.kernel_config
        if (mean_fn is None
                or sm_config is None
                or sm_config.smooth_mean_type != 'spline'
                or kc is None
                or kc.kernel_type not in ('gibbs', 'rbf')):
            return None

        from nucml_next.data.smooth_mean import compute_outputscale_from_residuals
        return compute_outputscale_from_residuals(log_E, log_sigma, mean_fn)

    def _score_group_local_mad(
        self,
        df_group: pd.DataFrame,
        experiments: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Score all points in a group using smooth mean + local MAD.

        This method:
        1. Fits a smooth mean on ALL pooled data in the group
        2. Computes rolling MAD of residuals in energy windows
        3. Assigns z-scores to every point: |residual| / local_MAD
        4. Flags point outliers by z-score threshold
        5. Flags experiment discrepancy by fraction of flagged points

        Works for both single-experiment and multi-experiment groups.
        No GP fitting, no subsampling, no Cholesky decomposition.

        Args:
            df_group: Full DataFrame for this (Z, A, MT[, Projectile]) group.
            experiments: Dict mapping Entry ID to experiment DataFrames.

        Returns:
            DataFrame with scoring columns populated.
        """
        result = df_group.copy()

        log_E_all = result['log_E'].values
        log_sigma_all = result['log_sigma'].values

        # 1. Fit smooth mean on pooled data (always spline for local_mad)
        from nucml_next.data.smooth_mean import (
            fit_smooth_mean, compute_rolling_mad_interpolator, SmoothMeanConfig,
        )

        sm_config = self.config.gp_config.smooth_mean_config
        if sm_config is not None and sm_config.smooth_mean_type == 'spline':
            mean_fn = fit_smooth_mean(log_E_all, log_sigma_all, sm_config)
        else:
            # Force spline for local_mad even if config says constant
            mean_fn = fit_smooth_mean(
                log_E_all, log_sigma_all,
                SmoothMeanConfig(smooth_mean_type='spline'),
            )

        # 2. Compute residuals and rolling MAD
        mad_fn = compute_rolling_mad_interpolator(
            log_E_all, log_sigma_all, mean_fn,
            window_fraction=self.config.mad_window_fraction,
            min_window_points=self.config.mad_min_window_points,
            mad_floor=self.config.mad_floor,
        )

        if mad_fn is None:
            # Too few points — fall back to simple MAD scoring
            return self._score_with_mad(result)

        # 3. Score every point
        residuals = log_sigma_all - mean_fn(log_E_all)
        local_mad = mad_fn(log_E_all)
        z_scores = np.abs(residuals) / local_mad

        result['gp_mean'] = mean_fn(log_E_all)   # Reuse column name for compat
        result['gp_std'] = local_mad               # Local MAD fills "uncertainty"
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

    def _score_single_experiment(
        self,
        df_group: pd.DataFrame,
        experiments: Dict[str, pd.DataFrame],
        group_key: Tuple[int, int, int] = None,
    ) -> pd.DataFrame:
        """Score when only one experiment exists (no consensus possible).

        **Limitation:** When ``smooth_mean_type='spline'`` is configured, the
        smooth mean for a single-experiment group is fitted to just that
        experiment.  If the experiment itself is biased, the mean absorbs the
        bias and outlier detection power is reduced.  For multi-experiment
        groups the pooled mean dilutes any single experiment's bias.  This is
        inherent to having only one data source and can only be addressed with
        cross-isotope information (Phase 4+).
        """
        result = df_group.copy()
        entry_id = list(experiments.keys())[0]
        exp_df = experiments[entry_id]
        n = len(exp_df)

        # Build kernel config for this group (if Gibbs kernel configured)
        group_kernel_config = None
        if group_key is not None:
            Z, A = group_key[0], group_key[1]
            S_n = None
            if self.config.s_n_column and self.config.s_n_column in df_group.columns:
                S_n = float(df_group[self.config.s_n_column].iloc[0])
            group_kernel_config = self._build_kernel_config_for_group(Z, A, S_n=S_n)

        # Compute smooth mean for single experiment
        single_mean_fn = None
        sm_config = self.config.gp_config.smooth_mean_config
        if sm_config is not None:
            from nucml_next.data.smooth_mean import fit_smooth_mean
            _log_E = exp_df['log_E'].values
            _log_sigma = exp_df['log_sigma'].values
            single_mean_fn = fit_smooth_mean(_log_E, _log_sigma, sm_config)

        # Inject data-driven lengthscale and outputscale if available
        if group_kernel_config is not None:
            _le = exp_df['log_E'].values
            _ls = exp_df['log_sigma'].values
            data_ls_fn = self._build_data_lengthscale(_le, _ls, single_mean_fn)
            if data_ls_fn is not None:
                from dataclasses import replace
                group_kernel_config = replace(
                    group_kernel_config,
                    data_lengthscale_interpolator=data_ls_fn,
                )
            data_os_fn = self._build_data_outputscale(_le, _ls, single_mean_fn)
            if data_os_fn is not None:
                from dataclasses import replace
                group_kernel_config = replace(
                    group_kernel_config,
                    outputscale_fn=data_os_fn,
                )

        # Cannot flag experiment as discrepant with no comparison
        result['experiment_outlier'] = False

        if n >= self.config.gp_config.min_points_for_gp:
            # Fit GP to single experiment
            try:
                gp = self._fit_experiment_gp(
                    exp_df, mean_fn=single_mean_fn,
                    kernel_config=group_kernel_config,
                )
                self._stats['gp_experiments'] += 1

                mean, std = gp.predict(exp_df['log_E'].values)
                z_scores = np.abs(exp_df['log_sigma'].values - mean) / np.clip(std, 1e-10, None)

                result.loc[exp_df.index, 'gp_mean'] = mean
                result.loc[exp_df.index, 'gp_std'] = std
                result.loc[exp_df.index, 'z_score'] = z_scores
                result.loc[exp_df.index, 'point_outlier'] = z_scores > self.config.point_z_threshold
                result.loc[exp_df.index, 'calibration_metric'] = gp.calibration_metric or np.nan

                # Populate outlier probability from contaminated EM
                self._assign_outlier_probabilities(
                    result, exp_df, gp, also_set_point_outlier=True,
                )

            except Exception as e:
                logger.warning(f"GP fit failed for single experiment {entry_id}: {e}")
                return self._score_with_mad(result)
        else:
            self._stats['small_experiments'] += 1
            return self._score_with_mad(result)

        return result

    # ------------------------------------------------------------------
    # Phase 4: Hierarchical refitting helpers
    # ------------------------------------------------------------------

    def _extract_group_hyperparameters(
        self,
        fitted_gps: Dict[str, 'ExactGPExperiment'],
    ) -> Optional[Dict[str, Any]]:
        """Extract group-level hyperparameter statistics from fitted GPs.

        Collects outputscale and optimisable kernel parameters from each GP,
        then computes median, Q1 and Q3 per parameter.

        Returns:
            Dict with keys ``n_experiments``, ``outputscale_median``,
            ``param_medians``, ``param_q1``, ``param_q3``.
            Returns ``None`` if fewer than ``min_experiments_for_refit`` GPs.
        """
        gps = list(fitted_gps.values())
        if len(gps) < self.config.min_experiments_for_refit:
            return None

        outputscales = np.array([gp._outputscale for gp in gps])

        # Collect per-kernel optimisable params (shape: n_experiments × n_params)
        param_arrays = np.array([
            gp._kernel.get_optimizable_params() for gp in gps
        ])  # (n_experiments, n_params)

        return {
            'n_experiments': len(gps),
            'outputscale_median': float(np.median(outputscales)),
            'param_medians': np.median(param_arrays, axis=0),
            'param_q1': np.percentile(param_arrays, 25, axis=0),
            'param_q3': np.percentile(param_arrays, 75, axis=0),
            'n_params': param_arrays.shape[1],
        }

    def _compute_refit_bounds(
        self,
        group_stats: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IQR-based parameter bounds for hierarchical refit.

        Bounds are ``[Q1 - margin*IQR, Q3 + margin*IQR]`` per parameter.

        **Kernel-type-aware clipping:**
        - RBF (``n_params == 1``): lower clipped to max(lower, 1e-3) —
          lengthscale must be strictly positive.
        - Gibbs (``n_params >= 2``): NO clipping — a₀, a₁ can be negative.

        **Degenerate IQR handling:** When all experiments produce identical
        parameters (IQR = 0), bounds are expanded to [median*0.5, median*2.0]
        for parameters far from zero, or [median-0.5, median+0.5] for
        parameters near zero.

        Returns:
            ``(lower, upper)`` arrays of shape ``(n_params,)``.
        """
        margin = self.config.refit_bounds_iqr_margin
        q1 = group_stats['param_q1']
        q3 = group_stats['param_q3']
        medians = group_stats['param_medians']
        n_params = group_stats['n_params']

        iqr = q3 - q1
        lower = q1 - margin * iqr
        upper = q3 + margin * iqr

        # Handle zero-IQR (all identical params) — expand to avoid degenerate bounds
        for i in range(n_params):
            if iqr[i] < 1e-12:
                if np.abs(medians[i]) > 1.0:
                    # Parameter far from zero: ±50% around median
                    lower[i] = medians[i] * 0.5
                    upper[i] = medians[i] * 2.0
                    # Handle negative medians
                    if lower[i] > upper[i]:
                        lower[i], upper[i] = upper[i], lower[i]
                else:
                    # Parameter near zero: ±0.5
                    lower[i] = medians[i] - 0.5
                    upper[i] = medians[i] + 0.5

        # Kernel-type-aware clipping
        if n_params == 1:
            # RBF: lengthscale must be positive
            lower = np.maximum(lower, 1e-3)

        # Gibbs (n_params >= 2): no clipping — a₀, a₁ can be negative

        return lower, upper

    def _hierarchical_refit(
        self,
        fitted_gps: Dict[str, 'ExactGPExperiment'],
        group_stats: Dict[str, Any],
        result: 'pd.DataFrame',
        large_exps: Dict[str, 'pd.DataFrame'],
    ) -> None:
        """Re-fit experiments with group-informed constrained bounds.

        For each GP, calls ``refit_with_constraints()`` with:
        - Shared outputscale (group median) when ``refit_share_outputscale`` is True
        - Parameter bounds from ``_compute_refit_bounds()``

        Updates the result DataFrame in-place with new predictions.
        Failures are non-fatal: a warning is logged and Pass 1 results
        are preserved.

        Args:
            fitted_gps: Dict of entry_id -> fitted ExactGPExperiment from Pass 1.
            group_stats: Group-level statistics from ``_extract_group_hyperparameters``.
            result: The result DataFrame to update in-place.
            large_exps: Dict of entry_id -> experiment DataFrame (for index lookup).
        """
        param_bounds = self._compute_refit_bounds(group_stats)
        outputscale = (
            group_stats['outputscale_median']
            if self.config.refit_share_outputscale
            else None
        )

        self._stats['hierarchical_groups'] += 1

        for entry_id, gp in fitted_gps.items():
            try:
                gp.refit_with_constraints(
                    outputscale=outputscale,
                    param_bounds=param_bounds,
                )
                self._stats['hierarchical_refits'] += 1

                # Update predictions in result DataFrame
                exp_df = large_exps[entry_id]
                mean, std = gp.predict(exp_df['log_E'].values)
                z_scores = np.abs(
                    exp_df['log_sigma'].values - mean
                ) / np.clip(std, 1e-10, None)

                result.loc[exp_df.index, 'gp_mean'] = mean
                result.loc[exp_df.index, 'gp_std'] = std
                result.loc[exp_df.index, 'z_score'] = z_scores
                result.loc[exp_df.index, 'calibration_metric'] = (
                    gp.calibration_metric or np.nan
                )

                # Update outlier probability from EM refit
                self._assign_outlier_probabilities(result, exp_df, gp)

            except Exception as e:
                logger.warning(
                    f"Hierarchical refit failed for experiment {entry_id}: {e}"
                )
                # Keep Pass 1 results — non-fatal

    def _score_multi_experiment(
        self,
        df_group: pd.DataFrame,
        experiments: Dict[str, pd.DataFrame],
        group_key: Tuple[int, int, int],
    ) -> pd.DataFrame:
        """Score with multiple experiments - build consensus and flag discrepant."""
        result = df_group.copy()

        # Partition into large and small experiments
        large_exps = {}
        small_exps = {}
        min_pts = self.config.gp_config.min_points_for_gp

        for entry_id, exp_df in experiments.items():
            if len(exp_df) >= min_pts:
                large_exps[entry_id] = exp_df
            else:
                small_exps[entry_id] = exp_df

        # Compute group-level smooth mean from ALL pooled data (before
        # per-experiment split).  When smooth_mean_type='constant' (default)
        # this is a no-op that returns np.mean, preserving existing behaviour.
        group_mean_fn = None
        sm_config = self.config.gp_config.smooth_mean_config
        if sm_config is not None:
            from nucml_next.data.smooth_mean import fit_smooth_mean
            _log_E_pool = df_group['log_E'].values
            _log_sigma_pool = df_group['log_sigma'].values
            group_mean_fn = fit_smooth_mean(_log_E_pool, _log_sigma_pool, sm_config)

        # Build kernel config with RIPL-3 interpolator for this (Z, A) group.
        # group_key is (Z, A, MT) or (Z, A, MT, Projectile).
        Z, A = group_key[0], group_key[1]

        # Try to get S_n from the data if available
        S_n = None
        if self.config.s_n_column and self.config.s_n_column in df_group.columns:
            S_n = float(df_group[self.config.s_n_column].iloc[0])

        group_kernel_config = self._build_kernel_config_for_group(Z, A, S_n=S_n)

        # Inject data-driven lengthscale and outputscale if available
        if group_kernel_config is not None:
            _log_E_pool = df_group['log_E'].values
            _log_sigma_pool = df_group['log_sigma'].values
            data_ls_fn = self._build_data_lengthscale(
                _log_E_pool, _log_sigma_pool, group_mean_fn,
            )
            if data_ls_fn is not None:
                from dataclasses import replace
                group_kernel_config = replace(
                    group_kernel_config,
                    data_lengthscale_interpolator=data_ls_fn,
                )
            data_os_fn = self._build_data_outputscale(
                _log_E_pool, _log_sigma_pool, group_mean_fn,
            )
            if data_os_fn is not None:
                from dataclasses import replace
                group_kernel_config = replace(
                    group_kernel_config,
                    outputscale_fn=data_os_fn,
                )

        # Fit GPs to large experiments
        fitted_gps: Dict[str, ExactGPExperiment] = {}
        for entry_id, exp_df in large_exps.items():
            try:
                gp = self._fit_experiment_gp(
                    exp_df, mean_fn=group_mean_fn,
                    kernel_config=group_kernel_config,
                )
                fitted_gps[entry_id] = gp
                self._stats['gp_experiments'] += 1

                # Store per-experiment predictions
                mean, std = gp.predict(exp_df['log_E'].values)
                z_scores = np.abs(exp_df['log_sigma'].values - mean) / np.clip(std, 1e-10, None)

                result.loc[exp_df.index, 'gp_mean'] = mean
                result.loc[exp_df.index, 'gp_std'] = std
                result.loc[exp_df.index, 'z_score'] = z_scores
                result.loc[exp_df.index, 'calibration_metric'] = gp.calibration_metric or np.nan

                # Populate outlier probability from contaminated EM
                self._assign_outlier_probabilities(result, exp_df, gp)

            except Exception as e:
                logger.warning(f"GP fit failed for experiment {entry_id}: {e}")
                # Fall back to MAD for this experiment
                self._stats['small_experiments'] += 1
                small_exps[entry_id] = exp_df

        # Pass 2: Hierarchical refitting (opt-in, Phase 4)
        if self.config.hierarchical_refitting:
            group_stats = self._extract_group_hyperparameters(fitted_gps)
            if group_stats is not None:
                self._hierarchical_refit(
                    fitted_gps, group_stats, result, large_exps,
                )
            else:
                self._stats['hierarchical_skipped_groups'] += 1

        # Build consensus if we have >= 2 fitted GPs
        if len(fitted_gps) >= self.config.consensus_config.min_experiments_for_consensus:
            # Determine energy range from all experiments
            all_log_E = df_group['log_E'].values
            energy_range = (all_log_E.min(), all_log_E.max())

            # Build consensus
            consensus = ConsensusBuilder(self.config.consensus_config)
            try:
                _, cons_mean, cons_std = consensus.build_consensus(fitted_gps, energy_range)

                # Flag discrepant experiments
                exp_flags = consensus.flag_discrepant_experiments()

                for entry_id, is_discrepant in exp_flags.items():
                    if entry_id in large_exps:
                        exp_df = large_exps[entry_id]
                        result.loc[exp_df.index, 'experiment_outlier'] = is_discrepant
                        if is_discrepant:
                            self._stats['discrepant_experiments'] += 1

                # Evaluate small experiments against consensus
                for entry_id, exp_df in small_exps.items():
                    self._stats['small_experiments'] += 1
                    log_E = exp_df['log_E'].values
                    log_sigma = exp_df['log_sigma'].values

                    # Get uncertainties
                    log_unc = self._get_log_uncertainties(exp_df)

                    z_scores, is_discrepant = consensus.evaluate_small_experiment(
                        log_E, log_sigma, log_unc
                    )

                    # Use consensus predictions for small experiments
                    cons_mean_pts, cons_std_pts = consensus.predict_at_points(log_E)

                    result.loc[exp_df.index, 'gp_mean'] = cons_mean_pts
                    result.loc[exp_df.index, 'gp_std'] = cons_std_pts
                    result.loc[exp_df.index, 'z_score'] = z_scores
                    result.loc[exp_df.index, 'experiment_outlier'] = is_discrepant
                    if is_discrepant:
                        self._stats['discrepant_experiments'] += 1

                # Cache for diagnostics (only if not clearing caches for memory efficiency)
                if not self.config.clear_caches_after_group:
                    self._fitted_gps[group_key] = fitted_gps
                    self._consensus_builders[group_key] = consensus

            except Exception as e:
                logger.warning(f"Consensus building failed for group {group_key}: {e}")
                # Fall back to individual GP scoring without consensus
                for entry_id, exp_df in small_exps.items():
                    scored = self._score_with_mad(exp_df)
                    for col in ['gp_mean', 'gp_std', 'z_score', 'point_outlier']:
                        result.loc[exp_df.index, col] = scored[col].values

        elif len(fitted_gps) == 1:
            # Only one large experiment - use as reference for small ones
            ref_gp = list(fitted_gps.values())[0]

            for entry_id, exp_df in small_exps.items():
                self._stats['small_experiments'] += 1
                log_E = exp_df['log_E'].values
                log_sigma = exp_df['log_sigma'].values

                # Predict from reference GP
                mean, std = ref_gp.predict(log_E)
                z_scores = np.abs(log_sigma - mean) / np.clip(std, 1e-10, None)

                result.loc[exp_df.index, 'gp_mean'] = mean
                result.loc[exp_df.index, 'gp_std'] = std
                result.loc[exp_df.index, 'z_score'] = z_scores

                # Flag experiment if median z-score is high
                median_z = np.median(z_scores[np.isfinite(z_scores)])
                is_discrepant = median_z > 3.0
                result.loc[exp_df.index, 'experiment_outlier'] = is_discrepant
                if is_discrepant:
                    self._stats['discrepant_experiments'] += 1

        else:
            # All experiments are small - MAD within each
            for entry_id, exp_df in experiments.items():
                self._stats['small_experiments'] += 1
                scored = self._score_with_mad(exp_df)
                for col in ['gp_mean', 'gp_std', 'z_score', 'point_outlier']:
                    result.loc[exp_df.index, col] = scored[col].values

        # Compute point outliers:
        # If contaminated likelihood produced outlier_probability, use that.
        # Otherwise fall back to z-score threshold.
        has_outlier_prob = np.isfinite(result['outlier_probability'])
        if has_outlier_prob.any():
            result.loc[has_outlier_prob, 'point_outlier'] = (
                result.loc[has_outlier_prob, 'outlier_probability'] > 0.5
            )
        # z-score fallback for points without outlier probability
        no_outlier_prob = ~has_outlier_prob & np.isfinite(result['z_score'])
        result.loc[no_outlier_prob, 'point_outlier'] = (
            result.loc[no_outlier_prob, 'z_score'] > self.config.point_z_threshold
        )

        return result

    def _ensure_ripl_loaded(self) -> None:
        """Lazy-load RIPL-3 data on first use (only if Gibbs kernel configured)."""
        if self._ripl_loader is not None:
            return

        kc = self.config.gp_config.kernel_config
        if kc is None or kc.kernel_type != 'gibbs':
            return

        from nucml_next.data.ripl_loader import RIPL3LevelDensity

        path = self.config.ripl_data_path
        self._ripl_loader = RIPL3LevelDensity(path)

        if self._ripl_loader.n_nuclides > 0:
            logger.info(
                f"RIPL-3 loaded: {self._ripl_loader.n_nuclides} nuclides "
                f"(for Gibbs kernel)"
            )
        else:
            logger.warning(
                "RIPL-3 data not found or empty; Gibbs kernel will "
                "fall back to RBF for all groups"
            )

    def _build_kernel_config_for_group(
        self,
        Z: int,
        A: int,
        S_n: Optional[float] = None,
    ) -> Optional[Any]:
        """Build a kernel config with injected RIPL-3 interpolator for (Z, A).

        If the base kernel_config is None (default RBF), returns None
        (preserving exact backward-compatible behaviour).

        If Gibbs kernel is configured, injects the RIPL-3 log_D interpolator
        for the compound nucleus (Z, A+1) — since neutron capture on target
        (Z, A) produces compound nucleus (Z, A+1).

        Args:
            Z: Proton number of the **target** nucleus.
            A: Mass number of the **target** nucleus.
            S_n: Neutron separation energy in MeV.  If None, a rough
                estimate is used.

        Returns:
            KernelConfig with RIPL-3 interpolator injected, or None.
        """
        from dataclasses import replace

        base_kc = self.config.gp_config.kernel_config
        if base_kc is None:
            return None

        if base_kc.kernel_type != 'gibbs':
            return base_kc

        # Ensure RIPL-3 is loaded
        self._ensure_ripl_loaded()
        if self._ripl_loader is None:
            return base_kc

        # Compound nucleus = target + neutron
        Z_compound = Z
        A_compound = A + 1

        # Get S_n (neutron separation energy)
        if S_n is None:
            # Default: use a rough estimate (~8 MeV for medium nuclei,
            # ~6 MeV for actinides). This is used only if S_n is not
            # provided in the DataFrame.
            S_n = self._estimate_s_n(Z, A)

        # Get RIPL-3 interpolator for compound nucleus
        interpolator = self._ripl_loader.get_log_D_interpolator(
            Z_compound, A_compound, S_n=S_n
        )

        if interpolator is None:
            logger.debug(
                f"No RIPL-3 data for compound ({Z_compound}, {A_compound}); "
                f"falling back to RBF for this group"
            )
            return base_kc

        # Inject interpolator into a copy of the kernel config
        return replace(base_kc, ripl_log_D_interpolator=interpolator)

    def _estimate_s_n(self, Z: int, A: int) -> float:
        """Rough estimate of neutron separation energy S_n in MeV.

        Uses a simple empirical formula.  This is only a fallback when
        S_n is not provided.  For accurate results, pass S_n from AME2020.

        Returns S_n in MeV.
        """
        # Simple parabolic approximation:
        # S_n ≈ 15.5 - 0.017*A for beta-stable nuclei (decent for A > 30)
        # with an even-odd correction
        S_n = 15.5 - 0.017 * (A + 1)  # A+1 is compound nucleus

        # Even-odd pairing correction: even-N compound nuclei have higher S_n
        N_compound = (A + 1) - Z
        if N_compound % 2 == 0:
            S_n += 1.0  # Even-N: higher S_n (paired neutron)
        else:
            S_n -= 0.5  # Odd-N: lower S_n

        # Clamp to physically reasonable range
        S_n = max(3.0, min(S_n, 12.0))
        return S_n

    def _fit_experiment_gp(
        self,
        exp_df: pd.DataFrame,
        mean_fn=None,
        kernel_config=None,
    ) -> ExactGPExperiment:
        """Fit ExactGP to a single experiment.

        Args:
            exp_df: DataFrame for one experiment.
            mean_fn: Optional pre-computed mean function from pooled group
                data.  Passed through to ``ExactGPExperiment.fit()``.
            kernel_config: Optional kernel config with RIPL-3 interpolator
                injected for this specific (Z, A) group.  If provided,
                overrides ``gp_config.kernel_config`` for this experiment.

        If CUDA OOM occurs, automatically retries on CPU.
        """
        log_E = exp_df['log_E'].values
        log_sigma = exp_df['log_sigma'].values
        log_unc = self._get_log_uncertainties(exp_df)

        # Build per-experiment GP config (with optional kernel override)
        gp_config = self.config.gp_config
        if kernel_config is not None and kernel_config is not gp_config.kernel_config:
            from dataclasses import replace as dc_replace
            gp_config = dc_replace(gp_config, kernel_config=kernel_config)

        gp = ExactGPExperiment(gp_config)

        try:
            gp.fit(log_E, log_sigma, log_unc, mean_fn=mean_fn)
        except RuntimeError as e:
            # Check if this is a CUDA OOM error
            error_msg = str(e).lower()
            if 'cuda out of memory' in error_msg or 'out of memory' in error_msg:
                logger.warning(
                    f"CUDA OOM for experiment with {len(log_E)} points, "
                    f"retrying on CPU"
                )
                # Clear GPU memory before retry
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

                # Create new GP with CPU config
                cpu_config = ExactGPExperimentConfig(
                    device='cpu',
                    min_points_for_gp=self.config.gp_config.min_points_for_gp,
                    use_wasserstein_calibration=self.config.gp_config.use_wasserstein_calibration,
                    lengthscale_bounds=self.config.gp_config.lengthscale_bounds,
                    default_rel_uncertainty=self.config.gp_config.default_rel_uncertainty,
                    smooth_mean_config=self.config.gp_config.smooth_mean_config,
                    kernel_config=kernel_config or self.config.gp_config.kernel_config,
                )
                gp = ExactGPExperiment(cpu_config)
                gp.fit(log_E, log_sigma, log_unc, mean_fn=mean_fn)
            else:
                raise  # Re-raise non-OOM errors

        return gp

    def _get_log_uncertainties(self, exp_df: pd.DataFrame) -> np.ndarray:
        """Extract log-space uncertainties from experiment data."""
        if 'Uncertainty' in exp_df.columns and 'CrossSection' in exp_df.columns:
            uncertainties = exp_df['Uncertainty'].values
            cross_sections = exp_df['CrossSection'].values
            return prepare_log_uncertainties(
                uncertainties,
                cross_sections,
                self.config.gp_config.default_rel_uncertainty
            )
        else:
            # No uncertainties - use default
            n = len(exp_df)
            return np.full(n, 0.434 * self.config.gp_config.default_rel_uncertainty)

    def _save_checkpoint(
        self, group_idx: int, results: Dict[Tuple, pd.DataFrame]
    ) -> None:
        """Save processing checkpoint for resume capability."""
        if not self.config.checkpoint_dir:
            return

        import pickle

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'experiment_outlier_checkpoint.pkl'

        # Serialize results
        serializable_results = {}
        for key, df in results.items():
            data = {
                'index': df.index.tolist(),
                'gp_mean': df['gp_mean'].values.tolist(),
                'gp_std': df['gp_std'].values.tolist(),
                'z_score': df['z_score'].values.tolist(),
                'experiment_outlier': df['experiment_outlier'].values.tolist(),
                'point_outlier': df['point_outlier'].values.tolist(),
                'calibration_metric': df['calibration_metric'].values.tolist(),
                'experiment_id': df['experiment_id'].values.tolist(),
            }
            if 'outlier_probability' in df.columns:
                data['outlier_probability'] = df['outlier_probability'].values.tolist()
            serializable_results[key] = data

        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'group_idx': group_idx,
                'results': serializable_results,
                'stats': self._stats.copy(),
            }, f)

        logger.info(f"Checkpoint saved: group {group_idx} -> {checkpoint_path}")

        # Clear results dict to free memory after checkpoint
        # Results are now persisted to disk - no need to keep in RAM
        results.clear()

    def _load_checkpoint(self) -> Tuple[int, Dict[Tuple, pd.DataFrame]]:
        """Load checkpoint if available."""
        if not self.config.checkpoint_dir:
            return 0, {}

        import pickle

        checkpoint_path = Path(self.config.checkpoint_dir) / 'experiment_outlier_checkpoint.pkl'
        if not checkpoint_path.exists():
            return 0, {}

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            group_idx = checkpoint['group_idx']
            self._stats = checkpoint.get('stats', self._stats)

            # Deserialize results
            partial_results = {}
            for key, data in checkpoint['results'].items():
                df_data = {
                    'gp_mean': data['gp_mean'],
                    'gp_std': data['gp_std'],
                    'z_score': data['z_score'],
                    'experiment_outlier': data['experiment_outlier'],
                    'point_outlier': data['point_outlier'],
                    'calibration_metric': data['calibration_metric'],
                    'experiment_id': data['experiment_id'],
                }
                if 'outlier_probability' in data:
                    df_data['outlier_probability'] = data['outlier_probability']
                scored_df = pd.DataFrame(df_data, index=data['index'])
                partial_results[key] = scored_df

            logger.info(f"Loaded checkpoint: group {group_idx}")
            return group_idx, partial_results

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
            return 0, {}

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return self._stats.copy()

    def get_fitted_gps(self) -> Dict[Tuple, Dict[str, ExactGPExperiment]]:
        """Return fitted GPs per group for diagnostics."""
        return self._fitted_gps

    def get_consensus_builders(self) -> Dict[Tuple, ConsensusBuilder]:
        """Return consensus builders per group for diagnostics."""
        return self._consensus_builders

    def _flush_streaming_chunks(self, chunks: List[pd.DataFrame]) -> None:
        """Write accumulated chunks to streaming output file.

        Uses Parquet row groups for efficient append-style writes.
        """
        if not chunks or not self.config.streaming_output:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        combined = pd.concat(chunks, ignore_index=True)
        table = pa.Table.from_pandas(combined)

        output_path = Path(self.config.streaming_output)

        # Append to existing file or create new
        if output_path.exists():
            # Read existing and combine
            existing = pq.read_table(output_path)
            combined_table = pa.concat_tables([existing, table])
            pq.write_table(combined_table, output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_path)

        logger.debug(f"Flushed {len(combined)} rows to {output_path}")

    def _finalize_streaming_output(self) -> None:
        """Finalize streaming output (placeholder for future optimizations)."""
        if not self.config.streaming_output:
            return

        output_path = Path(self.config.streaming_output)
        if output_path.exists():
            logger.info(f"Streaming output complete: {output_path}")
