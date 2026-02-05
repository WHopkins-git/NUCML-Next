"""
Consensus Building from Multiple Experiment Posteriors
=======================================================

Builds a consensus cross-section distribution from multiple EXFOR experiments'
GP posteriors and identifies discrepant experiments.

Key Classes:
    ConsensusConfig: Configuration dataclass
    ConsensusBuilder: Multi-experiment consensus builder

Usage:
    >>> from nucml_next.data.consensus import ConsensusBuilder, ConsensusConfig
    >>> consensus = ConsensusBuilder(ConsensusConfig())
    >>> consensus.build_consensus(fitted_gps, energy_range)
    >>> flags = consensus.flag_discrepant_experiments()
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class ConsensusConfig:
    """Configuration for consensus building.

    Attributes:
        n_grid_points: Number of points in common energy grid.
        extrapolation_margin: Margin (in log10 units) beyond experiment's
            training range to still consider as "interpolating".
        discrepancy_z_threshold: Z-score threshold for flagging discrepancy
            at individual grid points.
        discrepancy_fraction_threshold: Fraction of grid points that must
            be discrepant to flag entire experiment.
        min_experiments_for_consensus: Minimum experiments needed to build
            consensus (below this, cannot compare).
        min_overlap_fraction: Minimum fraction of an experiment's range that
            must overlap with consensus grid to evaluate it.
    """
    n_grid_points: int = 200
    extrapolation_margin: float = 0.1
    discrepancy_z_threshold: float = 2.0
    discrepancy_fraction_threshold: float = 0.2
    min_experiments_for_consensus: int = 2
    min_overlap_fraction: float = 0.1


class ConsensusBuilder:
    """
    Build consensus distribution from multiple experiment posteriors.

    The consensus is computed analytically from GP posterior means and
    variances (no sampling required since ExactGP gives Gaussian posteriors).

    Algorithm:
    1. Create common energy grid spanning all experiments
    2. For each experiment, compute (mean, std) at grid points within its range
    3. At each grid point, compute weighted median of means (weights = 1/std^2)
    4. Compute consensus uncertainty from weighted scatter
    5. Flag experiments whose posteriors deviate from consensus

    Example:
        >>> consensus = ConsensusBuilder(ConsensusConfig())
        >>> consensus.build_consensus(experiments, (0.0, 7.0))
        >>> flags = consensus.flag_discrepant_experiments()
        >>> for entry, is_outlier in flags.items():
        ...     if is_outlier:
        ...         print(f"Experiment {entry} is discrepant")
    """

    def __init__(self, config: ConsensusConfig = None):
        if config is None:
            config = ConsensusConfig()
        self.config = config

        # Will be set after build_consensus()
        self.energy_grid: Optional[np.ndarray] = None
        self.consensus_mean: Optional[np.ndarray] = None
        self.consensus_std: Optional[np.ndarray] = None

        # Per-experiment predictions and masks
        self._experiment_means: Dict[str, np.ndarray] = {}
        self._experiment_stds: Dict[str, np.ndarray] = {}
        self._experiment_masks: Dict[str, np.ndarray] = {}  # True where interpolating

        # Discrepancy scores
        self._experiment_discrepancy_fraction: Dict[str, float] = {}

    def build_consensus(
        self,
        experiments: Dict[str, 'ExactGPExperiment'],
        energy_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build consensus from multiple experiment posteriors.

        Args:
            experiments: Dictionary mapping Entry ID to fitted ExactGPExperiment.
            energy_range: (log_E_min, log_E_max) for common grid.

        Returns:
            (energy_grid, consensus_mean, consensus_std)

        Raises:
            ValueError: If fewer than min_experiments_for_consensus provided.
        """
        if len(experiments) < self.config.min_experiments_for_consensus:
            raise ValueError(
                f"Need >= {self.config.min_experiments_for_consensus} experiments, "
                f"got {len(experiments)}"
            )

        # Create common energy grid
        self.energy_grid = np.linspace(
            energy_range[0], energy_range[1], self.config.n_grid_points
        )

        # Get predictions from each experiment at grid points
        self._experiment_means = {}
        self._experiment_stds = {}
        self._experiment_masks = {}

        for entry_id, gp in experiments.items():
            mean, std = gp.predict(self.energy_grid)
            mask = gp.is_interpolating(
                self.energy_grid, margin=self.config.extrapolation_margin
            )

            self._experiment_means[entry_id] = mean
            self._experiment_stds[entry_id] = std
            self._experiment_masks[entry_id] = mask

        # Build consensus at each grid point
        self._compute_consensus()

        return self.energy_grid, self.consensus_mean, self.consensus_std

    def _compute_consensus(self) -> None:
        """Compute weighted consensus mean and std at each grid point."""
        n_grid = len(self.energy_grid)
        self.consensus_mean = np.full(n_grid, np.nan)
        self.consensus_std = np.full(n_grid, np.nan)

        for i in range(n_grid):
            # Collect predictions from experiments that are interpolating at this point
            means = []
            stds = []
            weights = []

            for entry_id in self._experiment_means:
                if self._experiment_masks[entry_id][i]:
                    m = self._experiment_means[entry_id][i]
                    s = self._experiment_stds[entry_id][i]

                    if np.isfinite(m) and np.isfinite(s) and s > 0:
                        means.append(m)
                        stds.append(s)
                        weights.append(1.0 / s ** 2)

            if len(means) >= 2:
                means = np.array(means)
                stds = np.array(stds)
                weights = np.array(weights)

                # Weighted median (more robust than weighted mean)
                self.consensus_mean[i] = self._weighted_median(means, weights)

                # Consensus uncertainty: combine measurement uncertainty and scatter
                # weighted_std^2 = (weighted variance of means) + (mean of variances)
                residuals = means - self.consensus_mean[i]
                weighted_scatter = np.sum(weights * residuals ** 2) / np.sum(weights)
                mean_variance = np.mean(stds ** 2)

                # Total variance is scatter between experiments + typical within-experiment
                total_var = weighted_scatter + mean_variance
                self.consensus_std[i] = np.sqrt(total_var)

            elif len(means) == 1:
                # Only one experiment at this point - use its values
                self.consensus_mean[i] = means[0]
                self.consensus_std[i] = stds[0]

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted median."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumsum = np.cumsum(sorted_weights)
        cutoff = 0.5 * cumsum[-1]

        return sorted_values[cumsum >= cutoff][0]

    def flag_discrepant_experiments(self) -> Dict[str, bool]:
        """
        Flag experiments that deviate significantly from consensus.

        An experiment is flagged if more than discrepancy_fraction_threshold
        of its grid points have z-score > discrepancy_z_threshold.

        Returns:
            Dictionary mapping Entry ID to boolean (True = discrepant).
        """
        if self.consensus_mean is None:
            raise RuntimeError("Must call build_consensus() first")

        flags = {}
        self._experiment_discrepancy_fraction = {}

        for entry_id in self._experiment_means:
            mask = self._experiment_masks[entry_id]
            n_valid = mask.sum()

            if n_valid < self.config.n_grid_points * self.config.min_overlap_fraction:
                # Not enough overlap to evaluate
                flags[entry_id] = False
                self._experiment_discrepancy_fraction[entry_id] = 0.0
                continue

            # Compute z-scores at valid grid points
            exp_mean = self._experiment_means[entry_id][mask]
            exp_std = self._experiment_stds[entry_id][mask]
            cons_mean = self.consensus_mean[mask]
            cons_std = self.consensus_std[mask]

            # Combined uncertainty
            total_std = np.sqrt(exp_std ** 2 + cons_std ** 2)
            total_std = np.clip(total_std, 1e-6, None)

            z_scores = np.abs(exp_mean - cons_mean) / total_std

            # Fraction of points that are discrepant
            discrepant = z_scores > self.config.discrepancy_z_threshold
            discrepancy_fraction = discrepant.sum() / n_valid

            self._experiment_discrepancy_fraction[entry_id] = discrepancy_fraction
            flags[entry_id] = discrepancy_fraction > self.config.discrepancy_fraction_threshold

            if flags[entry_id]:
                logger.debug(
                    f"Experiment {entry_id}: discrepancy_fraction={discrepancy_fraction:.2%} "
                    f"(threshold={self.config.discrepancy_fraction_threshold:.0%})"
                )

        return flags

    def evaluate_small_experiment(
        self,
        log_E: np.ndarray,
        log_sigma: np.ndarray,
        log_uncertainties: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Evaluate a small experiment (< min_points) against consensus.

        For experiments too small to fit their own GP, we evaluate their
        points directly against the consensus distribution.

        Args:
            log_E: log10(Energy) values for the experiment.
            log_sigma: log10(CrossSection) values.
            log_uncertainties: Measurement uncertainties in log-space.

        Returns:
            (z_scores, is_discrepant): Z-scores for each point and overall
                experiment discrepancy flag.
        """
        if self.consensus_mean is None:
            raise RuntimeError("Must call build_consensus() first")

        log_E = np.asarray(log_E).ravel()
        log_sigma = np.asarray(log_sigma).ravel()
        log_uncertainties = np.asarray(log_uncertainties).ravel()

        n_points = len(log_E)

        # Interpolate consensus at experiment's energy points
        # Handle NaN in consensus gracefully
        valid_consensus = np.isfinite(self.consensus_mean) & np.isfinite(self.consensus_std)

        if valid_consensus.sum() < 2:
            # Not enough consensus data - return neutral z-scores
            return np.zeros(n_points, dtype=float), False

        # Get valid data for interpolation
        grid_valid = self.energy_grid[valid_consensus]
        mean_valid = self.consensus_mean[valid_consensus]
        std_valid = self.consensus_std[valid_consensus]

        # Check for degenerate case (all same energy)
        if np.ptp(grid_valid) < 1e-10:
            return np.zeros(n_points, dtype=float), False

        try:
            mean_interp = interp1d(
                grid_valid,
                mean_valid,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )
            std_interp = interp1d(
                grid_valid,
                std_valid,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )

            consensus_mean_at_points = mean_interp(log_E)
            consensus_std_at_points = std_interp(log_E)
        except Exception as e:
            logger.warning(f"Interpolation failed in evaluate_small_experiment: {e}")
            return np.zeros(n_points, dtype=float), False

        # Combine consensus uncertainty with measurement uncertainty
        total_std = np.sqrt(consensus_std_at_points ** 2 + log_uncertainties ** 2)
        total_std = np.clip(total_std, 1e-6, None)

        # Compute z-scores
        z_scores = np.abs(log_sigma - consensus_mean_at_points) / total_std

        # Handle any remaining NaN/inf
        z_scores = np.where(np.isfinite(z_scores), z_scores, 0.0)

        # Flag experiment if median z-score is high
        valid_z = z_scores[np.isfinite(z_scores)]
        if len(valid_z) == 0:
            return z_scores, False

        median_z = np.median(valid_z)
        is_discrepant = bool(median_z > 3.0)  # Ensure native Python bool

        return z_scores, is_discrepant

    def get_experiment_discrepancy_fractions(self) -> Dict[str, float]:
        """Return discrepancy fraction for each experiment."""
        return self._experiment_discrepancy_fraction.copy()

    def predict_at_points(
        self,
        log_E: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict consensus mean and std at arbitrary energy points.

        Args:
            log_E: Query points in log10(Energy) space.

        Returns:
            (mean, std) interpolated from consensus grid.
        """
        if self.consensus_mean is None:
            raise RuntimeError("Must call build_consensus() first")

        log_E = np.asarray(log_E).ravel()
        n_points = len(log_E)

        valid = np.isfinite(self.consensus_mean) & np.isfinite(self.consensus_std)
        if valid.sum() < 2:
            return np.full(n_points, np.nan, dtype=float), np.full(n_points, np.nan, dtype=float)

        # Get valid data for interpolation
        grid_valid = self.energy_grid[valid]
        mean_valid = self.consensus_mean[valid]
        std_valid = self.consensus_std[valid]

        # Check for degenerate case (all same energy)
        if np.ptp(grid_valid) < 1e-10:
            return np.full(n_points, np.nan, dtype=float), np.full(n_points, np.nan, dtype=float)

        try:
            mean_interp = interp1d(
                grid_valid,
                mean_valid,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )
            std_interp = interp1d(
                grid_valid,
                std_valid,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )

            return mean_interp(log_E), std_interp(log_E)
        except Exception as e:
            logger.warning(f"Interpolation failed in predict_at_points: {e}")
            return np.full(n_points, np.nan, dtype=float), np.full(n_points, np.nan, dtype=float)
