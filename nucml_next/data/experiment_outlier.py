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
        }

        # Cached fitted GPs per group (for diagnostics)
        self._fitted_gps: Dict[Tuple, Dict[str, ExactGPExperiment]] = {}
        self._consensus_builders: Dict[Tuple, ConsensusBuilder] = {}

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
        result['experiment_id'] = ''

        # Determine experiment identifier column
        entry_col = self._get_entry_column(df)
        if entry_col:
            result['experiment_id'] = df[entry_col].astype(str)
        else:
            # No Entry column - assign unique ID per group
            result['experiment_id'] = 'unknown'

        # Group by (Z, A, MT)
        groups = list(result.groupby(['Z', 'A', 'MT']))
        n_groups = len(groups)
        self._stats['total_groups'] = n_groups
        self._stats['total_points'] = len(df)

        logger.info(
            f"Experiment outlier detection: {n_groups:,} groups, "
            f"{len(df):,} points"
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

        for i, ((z, a, mt), group_df) in iterator:
            if i < start_idx:
                continue

            group_key = (z, a, mt)

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
                           'point_outlier', 'calibration_metric', 'experiment_id']:
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

        # Case 1: Very small group - MAD fallback
        if n < self.config.min_group_size:
            self._stats['mad_groups'] += 1
            return self._score_with_mad(result)

        # Case 2: Single experiment
        if n_experiments == 1:
            self._stats['single_experiment_groups'] += 1
            return self._score_single_experiment(result, experiments)

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

    def _score_single_experiment(
        self,
        df_group: pd.DataFrame,
        experiments: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Score when only one experiment exists (no consensus possible)."""
        result = df_group.copy()
        entry_id = list(experiments.keys())[0]
        exp_df = experiments[entry_id]
        n = len(exp_df)

        # Cannot flag experiment as discrepant with no comparison
        result['experiment_outlier'] = False

        if n >= self.config.gp_config.min_points_for_gp:
            # Fit GP to single experiment
            try:
                gp = self._fit_experiment_gp(exp_df)
                self._stats['gp_experiments'] += 1

                mean, std = gp.predict(exp_df['log_E'].values)
                z_scores = np.abs(exp_df['log_sigma'].values - mean) / np.clip(std, 1e-10, None)

                result.loc[exp_df.index, 'gp_mean'] = mean
                result.loc[exp_df.index, 'gp_std'] = std
                result.loc[exp_df.index, 'z_score'] = z_scores
                result.loc[exp_df.index, 'point_outlier'] = z_scores > self.config.point_z_threshold
                result.loc[exp_df.index, 'calibration_metric'] = gp.calibration_metric or np.nan

            except Exception as e:
                logger.warning(f"GP fit failed for single experiment {entry_id}: {e}")
                return self._score_with_mad(result)
        else:
            self._stats['small_experiments'] += 1
            return self._score_with_mad(result)

        return result

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

        # Fit GPs to large experiments
        fitted_gps: Dict[str, ExactGPExperiment] = {}
        for entry_id, exp_df in large_exps.items():
            try:
                gp = self._fit_experiment_gp(exp_df)
                fitted_gps[entry_id] = gp
                self._stats['gp_experiments'] += 1

                # Store per-experiment predictions
                mean, std = gp.predict(exp_df['log_E'].values)
                z_scores = np.abs(exp_df['log_sigma'].values - mean) / np.clip(std, 1e-10, None)

                result.loc[exp_df.index, 'gp_mean'] = mean
                result.loc[exp_df.index, 'gp_std'] = std
                result.loc[exp_df.index, 'z_score'] = z_scores
                result.loc[exp_df.index, 'calibration_metric'] = gp.calibration_metric or np.nan

            except Exception as e:
                logger.warning(f"GP fit failed for experiment {entry_id}: {e}")
                # Fall back to MAD for this experiment
                self._stats['small_experiments'] += 1
                small_exps[entry_id] = exp_df

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

        # Compute point outliers based on z-scores
        valid_z = np.isfinite(result['z_score'])
        result.loc[valid_z, 'point_outlier'] = (
            result.loc[valid_z, 'z_score'] > self.config.point_z_threshold
        )

        return result

    def _fit_experiment_gp(self, exp_df: pd.DataFrame) -> ExactGPExperiment:
        """Fit ExactGP to a single experiment.

        If CUDA OOM occurs, automatically retries on CPU.
        """
        log_E = exp_df['log_E'].values
        log_sigma = exp_df['log_sigma'].values
        log_unc = self._get_log_uncertainties(exp_df)

        gp = ExactGPExperiment(self.config.gp_config)

        try:
            gp.fit(log_E, log_sigma, log_unc)
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
                )
                gp = ExactGPExperiment(cpu_config)
                gp.fit(log_E, log_sigma, log_unc)
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
            serializable_results[key] = {
                'index': df.index.tolist(),
                'gp_mean': df['gp_mean'].values.tolist(),
                'gp_std': df['gp_std'].values.tolist(),
                'z_score': df['z_score'].values.tolist(),
                'experiment_outlier': df['experiment_outlier'].values.tolist(),
                'point_outlier': df['point_outlier'].values.tolist(),
                'calibration_metric': df['calibration_metric'].values.tolist(),
                'experiment_id': df['experiment_id'].values.tolist(),
            }

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
                scored_df = pd.DataFrame({
                    'gp_mean': data['gp_mean'],
                    'gp_std': data['gp_std'],
                    'z_score': data['z_score'],
                    'experiment_outlier': data['experiment_outlier'],
                    'point_outlier': data['point_outlier'],
                    'calibration_metric': data['calibration_metric'],
                    'experiment_id': data['experiment_id'],
                }, index=data['index'])
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
