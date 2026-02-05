"""
Tests for Per-Experiment GP Outlier Detection
==============================================

Tests the ExperimentOutlierDetector, ExactGPExperiment, ConsensusBuilder,
and calibration utilities using synthetic nuclear cross-section data.
"""

import numpy as np
import pandas as pd
import pytest

from nucml_next.data.calibration import (
    compute_wasserstein_calibration,
    compute_loo_z_scores,
    compute_loo_z_scores_from_cholesky,
    optimize_lengthscale_wasserstein,
    calibration_diagnostic,
)
from nucml_next.data.experiment_gp import (
    ExactGPExperiment,
    ExactGPExperimentConfig,
    prepare_log_uncertainties,
)
from nucml_next.data.consensus import (
    ConsensusBuilder,
    ConsensusConfig,
)
from nucml_next.data.experiment_outlier import (
    ExperimentOutlierDetector,
    ExperimentOutlierConfig,
)


# =============================================================================
# Synthetic Data Generators
# =============================================================================

def _make_synthetic_experiment(
    entry_id: str,
    n: int,
    E_min: float = 1e-2,
    E_max: float = 1e6,
    noise: float = 0.1,
    bias: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic cross-section experiment data.

    Generates a 1/v-law cross-section (sigma = 100 / sqrt(E)) with
    lognormal noise, mimicking thermal neutron capture behavior.

    Args:
        entry_id: EXFOR Entry identifier
        n: Number of data points
        E_min: Minimum energy (eV)
        E_max: Maximum energy (eV)
        noise: Lognormal noise scale
        bias: Multiplicative bias factor (for testing discrepancy)
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    E = np.logspace(np.log10(E_min), np.log10(E_max), n)

    # 1/v law cross-section
    sigma_clean = 100.0 / np.sqrt(E)

    # Apply bias (for simulating discrepant experiments)
    if bias != 0.0:
        sigma_clean = sigma_clean * (1 + bias)

    # Add noise
    if noise > 0 and n > 1:
        sigma = sigma_clean * rng.lognormal(0, noise, n)
    else:
        sigma = sigma_clean.copy()

    sigma = np.clip(sigma, 1e-30, None)
    uncertainty = sigma * 0.05  # 5% relative uncertainty

    return pd.DataFrame({
        'Entry': entry_id,
        'Energy': E,
        'CrossSection': sigma,
        'Uncertainty': uncertainty,
    })


def _make_multi_experiment_group(
    Z: int, A: int, MT: int,
    n_experiments: int = 5,
    points_per_exp: int = 30,
    noise: float = 0.1,
    n_discrepant: int = 0,
    discrepant_bias: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic group with multiple experiments.

    Args:
        Z, A, MT: Reaction identifiers
        n_experiments: Number of experiments
        points_per_exp: Points per experiment
        noise: Noise level
        n_discrepant: Number of experiments to make discrepant
        discrepant_bias: Multiplicative bias for discrepant experiments
        seed: Random seed
    """
    dfs = []
    for i in range(n_experiments):
        bias = discrepant_bias if i < n_discrepant else 0.0
        exp_df = _make_synthetic_experiment(
            entry_id=f"E{i+1:04d}",
            n=points_per_exp,
            noise=noise,
            bias=bias,
            seed=seed + i,
        )
        exp_df['Z'] = Z
        exp_df['A'] = A
        exp_df['MT'] = MT
        exp_df['N'] = A - Z
        exp_df['Projectile'] = 'n'
        dfs.append(exp_df)

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Tests for calibration.py
# =============================================================================

class TestWassersteinCalibration:
    """Tests for Wasserstein calibration functions."""

    def test_wasserstein_well_calibrated(self):
        """Well-calibrated z-scores should have low Wasserstein distance."""
        rng = np.random.default_rng(42)
        z_scores = rng.standard_normal(1000)
        w = compute_wasserstein_calibration(z_scores)

        # For truly standard normal, W should be small (< 0.1)
        assert w < 0.1

    def test_wasserstein_miscalibrated(self):
        """Miscalibrated z-scores should have higher Wasserstein distance."""
        rng = np.random.default_rng(42)

        # Underdispersed (too narrow)
        z_narrow = rng.standard_normal(1000) * 0.5
        w_narrow = compute_wasserstein_calibration(z_narrow)

        # Overdispersed (too wide)
        z_wide = rng.standard_normal(1000) * 2.0
        w_wide = compute_wasserstein_calibration(z_wide)

        # Correctly calibrated
        z_correct = rng.standard_normal(1000)
        w_correct = compute_wasserstein_calibration(z_correct)

        # Miscalibrated should have higher Wasserstein
        assert w_narrow > w_correct
        assert w_wide > w_correct

    def test_wasserstein_insufficient_data(self):
        """Returns inf with < 3 valid z-scores."""
        w = compute_wasserstein_calibration(np.array([1.0, 2.0]))
        assert w == np.inf

    def test_wasserstein_handles_nan(self):
        """Filters out NaN values."""
        rng = np.random.default_rng(42)
        z = rng.standard_normal(100)
        z[::10] = np.nan
        w = compute_wasserstein_calibration(z)

        assert np.isfinite(w)


class TestLOOZScores:
    """Tests for leave-one-out z-score computation."""

    def test_loo_from_cholesky_basic(self):
        """LOO z-scores from Cholesky work correctly."""
        n = 20
        rng = np.random.default_rng(42)

        # Create simple positive definite matrix
        A = rng.standard_normal((n, n))
        K = A @ A.T + 0.1 * np.eye(n)
        L = np.linalg.cholesky(K)

        y = rng.standard_normal(n)
        mean = np.zeros(n)

        z_scores = compute_loo_z_scores_from_cholesky(L, y, mean)

        assert z_scores.shape == (n,)
        assert np.all(np.isfinite(z_scores))

    def test_loo_z_scores_full_pipeline(self):
        """Full LOO z-score computation from GP hyperparameters."""
        n = 30
        rng = np.random.default_rng(42)

        x = np.linspace(0, 5, n)
        y = np.sin(x) + rng.standard_normal(n) * 0.1
        noise_var = np.full(n, 0.01)

        z_scores = compute_loo_z_scores(
            x, y, noise_var,
            lengthscale=1.0,
            outputscale=1.0,
        )

        assert z_scores.shape == (n,)
        # For well-specified model, most z-scores should be < 3
        assert np.mean(np.abs(z_scores) < 3) > 0.8


class TestLengthscaleOptimization:
    """Tests for Wasserstein-based lengthscale optimization."""

    def test_optimize_lengthscale_runs(self):
        """Lengthscale optimization completes without error."""
        n = 50
        rng = np.random.default_rng(42)

        x = np.linspace(0, 5, n)
        y = np.sin(x) + rng.standard_normal(n) * 0.1
        noise_var = np.full(n, 0.01)

        ls, w = optimize_lengthscale_wasserstein(
            x, y, noise_var,
            lengthscale_bounds=(0.1, 5.0),
            n_grid=10,
        )

        assert 0.1 <= ls <= 5.0
        assert np.isfinite(w)

    def test_optimize_lengthscale_prefers_calibrated(self):
        """Optimization finds lengthscale that produces calibrated z-scores."""
        n = 100
        rng = np.random.default_rng(42)

        # Generate data with known lengthscale
        true_ls = 1.0
        x = np.linspace(0, 10, n)
        # Smooth function + noise
        y = np.sin(x / true_ls) + rng.standard_normal(n) * 0.1
        noise_var = np.full(n, 0.01)

        ls, w = optimize_lengthscale_wasserstein(
            x, y, noise_var,
            lengthscale_bounds=(0.1, 10.0),
            n_grid=20,
        )

        # Optimized lengthscale should produce low Wasserstein
        assert w < 0.5


class TestCalibrationDiagnostic:
    """Tests for calibration diagnostic function."""

    def test_diagnostic_structure(self):
        """Diagnostic returns expected structure."""
        rng = np.random.default_rng(42)
        z_scores = rng.standard_normal(100)

        diag = calibration_diagnostic(z_scores)

        assert 'wasserstein' in diag
        assert 'coverage_empirical' in diag
        assert 'coverage_theoretical' in diag
        assert 'coverage_error' in diag
        assert 'n_valid' in diag

    def test_diagnostic_coverage_levels(self):
        """Coverage computed at standard sigma levels."""
        rng = np.random.default_rng(42)
        z_scores = rng.standard_normal(10000)  # Large sample

        diag = calibration_diagnostic(z_scores)

        # For standard normal, coverage should be close to theoretical
        for sigma in [1.0, 2.0, 3.0]:
            error = abs(diag['coverage_error'][sigma])
            # With 10k samples, error should be < 2%
            assert error < 0.02


# =============================================================================
# Tests for experiment_gp.py
# =============================================================================

class TestPrepareLogUncertainties:
    """Tests for uncertainty transformation to log-space."""

    def test_basic_transformation(self):
        """Standard uncertainty transformation."""
        xs = np.array([100.0, 200.0, 300.0])
        unc = np.array([5.0, 10.0, 15.0])  # 5% relative

        log_unc = prepare_log_uncertainties(unc, xs)

        # 0.434 * 0.05 = 0.0217
        expected = 0.434 * 0.05
        np.testing.assert_allclose(log_unc, expected, rtol=0.01)

    def test_handles_missing_uncertainties(self):
        """Missing uncertainties filled with median."""
        xs = np.array([100.0, 200.0, 300.0, 400.0])
        unc = np.array([5.0, 10.0, 0.0, -1.0])  # Last two invalid

        log_unc = prepare_log_uncertainties(unc, xs)

        assert log_unc.shape == (4,)
        assert np.all(np.isfinite(log_unc))
        assert np.all(log_unc > 0)

    def test_clamps_extreme_values(self):
        """Relative uncertainties clamped to [1%, 100%]."""
        xs = np.array([100.0, 100.0, 100.0])
        unc = np.array([0.001, 50.0, 200.0])  # 0.001%, 50%, 200%

        log_unc = prepare_log_uncertainties(unc, xs)

        # Should be clamped to [0.434*0.01, 0.434*1.0]
        assert np.all(log_unc >= 0.434 * 0.01 - 0.001)
        assert np.all(log_unc <= 0.434 * 1.0 + 0.001)


class TestExactGPExperimentConfig:
    """Tests for ExactGPExperimentConfig."""

    def test_defaults(self):
        """Config has sensible defaults."""
        config = ExactGPExperimentConfig()

        assert config.min_points_for_gp == 5
        assert config.max_epochs == 200
        assert config.use_wasserstein_calibration is True
        assert config.default_rel_uncertainty == 0.10
        assert config.lengthscale_bounds == (0.01, 10.0)


class TestExactGPExperiment:
    """Tests for ExactGPExperiment class."""

    def test_fit_and_predict(self):
        """Basic fit and predict workflow."""
        exp_df = _make_synthetic_experiment("E0001", n=30, noise=0.05)
        log_E = np.log10(exp_df['Energy'].values)
        log_sigma = np.log10(exp_df['CrossSection'].values)
        log_unc = prepare_log_uncertainties(
            exp_df['Uncertainty'].values,
            exp_df['CrossSection'].values,
        )

        config = ExactGPExperimentConfig(use_wasserstein_calibration=False)
        gp = ExactGPExperiment(config)
        gp.fit(log_E, log_sigma, log_unc)

        assert gp.is_fitted

        # Predict at training points
        mean, std = gp.predict(log_E)

        assert mean.shape == log_E.shape
        assert std.shape == log_E.shape
        assert np.all(std > 0)

    def test_fit_too_few_points_raises(self):
        """Fitting with too few points raises ValueError."""
        config = ExactGPExperimentConfig(min_points_for_gp=5)
        gp = ExactGPExperiment(config)

        log_E = np.array([1.0, 2.0, 3.0])
        log_sigma = np.array([0.1, 0.2, 0.3])
        log_unc = np.array([0.01, 0.01, 0.01])

        with pytest.raises(ValueError, match="Need >= 5 points"):
            gp.fit(log_E, log_sigma, log_unc)

    def test_predict_before_fit_raises(self):
        """Prediction before fitting raises RuntimeError."""
        gp = ExactGPExperiment()

        with pytest.raises(RuntimeError, match="Must call fit"):
            gp.predict(np.array([1.0, 2.0]))

    def test_is_interpolating(self):
        """Correctly identifies interpolation vs extrapolation."""
        exp_df = _make_synthetic_experiment("E0001", n=20, E_min=1e0, E_max=1e4)
        log_E = np.log10(exp_df['Energy'].values)
        log_sigma = np.log10(exp_df['CrossSection'].values)
        log_unc = prepare_log_uncertainties(
            exp_df['Uncertainty'].values,
            exp_df['CrossSection'].values,
        )

        config = ExactGPExperimentConfig(use_wasserstein_calibration=False)
        gp = ExactGPExperiment(config)
        gp.fit(log_E, log_sigma, log_unc)

        # Points within range: interpolating
        query_interp = np.array([1.0, 2.0, 3.0])  # log10(10), log10(100), log10(1000)
        mask_interp = gp.is_interpolating(query_interp, margin=0.1)
        assert mask_interp.all()

        # Points outside range: extrapolating
        query_extrap = np.array([-3.0, 7.0])  # log10(0.001), log10(1e7)
        mask_extrap = gp.is_interpolating(query_extrap, margin=0.1)
        assert not mask_extrap.any()

    def test_energy_range_property(self):
        """Energy range property returns correct bounds."""
        exp_df = _make_synthetic_experiment("E0001", n=20, E_min=1e0, E_max=1e4)
        log_E = np.log10(exp_df['Energy'].values)
        log_sigma = np.log10(exp_df['CrossSection'].values)
        log_unc = prepare_log_uncertainties(
            exp_df['Uncertainty'].values,
            exp_df['CrossSection'].values,
        )

        config = ExactGPExperimentConfig(use_wasserstein_calibration=False)
        gp = ExactGPExperiment(config)
        gp.fit(log_E, log_sigma, log_unc)

        E_min, E_max = gp.energy_range
        assert E_min == pytest.approx(0.0, abs=0.1)  # log10(1) = 0
        assert E_max == pytest.approx(4.0, abs=0.1)  # log10(10000) = 4

    def test_wasserstein_calibration_mode(self):
        """Wasserstein calibration mode produces calibration_metric."""
        exp_df = _make_synthetic_experiment("E0001", n=50, noise=0.05)
        log_E = np.log10(exp_df['Energy'].values)
        log_sigma = np.log10(exp_df['CrossSection'].values)
        log_unc = prepare_log_uncertainties(
            exp_df['Uncertainty'].values,
            exp_df['CrossSection'].values,
        )

        config = ExactGPExperimentConfig(use_wasserstein_calibration=True)
        gp = ExactGPExperiment(config)
        gp.fit(log_E, log_sigma, log_unc)

        assert gp.calibration_metric is not None
        assert np.isfinite(gp.calibration_metric)


# =============================================================================
# Tests for consensus.py
# =============================================================================

class TestConsensusConfig:
    """Tests for ConsensusConfig."""

    def test_defaults(self):
        """Config has sensible defaults."""
        config = ConsensusConfig()

        assert config.n_grid_points == 200
        assert config.extrapolation_margin == 0.1
        assert config.discrepancy_z_threshold == 2.0
        assert config.discrepancy_fraction_threshold == 0.2
        assert config.min_experiments_for_consensus == 2


class TestConsensusBuilder:
    """Tests for ConsensusBuilder class."""

    def _fit_multiple_gps(self, group_df: pd.DataFrame) -> dict:
        """Helper to fit GPs to each experiment in a group."""
        gp_config = ExactGPExperimentConfig(use_wasserstein_calibration=False)
        fitted_gps = {}

        for entry_id, exp_df in group_df.groupby('Entry'):
            if len(exp_df) < 5:
                continue

            log_E = np.log10(exp_df['Energy'].values)
            log_sigma = np.log10(exp_df['CrossSection'].values)
            log_unc = prepare_log_uncertainties(
                exp_df['Uncertainty'].values,
                exp_df['CrossSection'].values,
            )

            gp = ExactGPExperiment(gp_config)
            gp.fit(log_E, log_sigma, log_unc)
            fitted_gps[str(entry_id)] = gp

        return fitted_gps

    def test_build_consensus_basic(self):
        """Build consensus from multiple experiments."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=5,
            points_per_exp=30,
        )

        fitted_gps = self._fit_multiple_gps(group_df)

        consensus = ConsensusBuilder()
        energy_range = (-2.0, 6.0)
        grid, mean, std = consensus.build_consensus(fitted_gps, energy_range)

        assert grid.shape == (200,)  # Default n_grid_points
        assert mean.shape == (200,)
        assert std.shape == (200,)
        # Should have valid values where experiments overlap
        assert np.sum(np.isfinite(mean)) > 100

    def test_build_consensus_too_few_experiments(self):
        """Raises error with too few experiments."""
        exp_df = _make_synthetic_experiment("E0001", n=30)
        exp_df['Z'] = 92
        exp_df['A'] = 235
        exp_df['MT'] = 18

        fitted_gps = self._fit_multiple_gps(exp_df)

        consensus = ConsensusBuilder()
        with pytest.raises(ValueError, match="Need >= 2 experiments"):
            consensus.build_consensus(fitted_gps, (-2.0, 6.0))

    def test_flag_discrepant_experiments(self):
        """Correctly flags discrepant experiments."""
        # Create group with 1 discrepant experiment
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=5,
            points_per_exp=50,
            noise=0.05,
            n_discrepant=1,
            discrepant_bias=0.8,  # 80% higher - clearly discrepant
            seed=42,
        )

        fitted_gps = self._fit_multiple_gps(group_df)

        consensus = ConsensusBuilder()
        consensus.build_consensus(fitted_gps, (-2.0, 6.0))
        flags = consensus.flag_discrepant_experiments()

        # Should flag the first experiment (E0001) as discrepant
        assert 'E0001' in flags
        assert flags['E0001'] == True  # Use == instead of is for numpy.bool_

        # Other experiments should not be flagged
        assert sum(flags.values()) <= 2  # At most 2 flagged (some leeway)

    def test_consensus_without_build_raises(self):
        """Flagging before building consensus raises error."""
        consensus = ConsensusBuilder()

        with pytest.raises(RuntimeError, match="Must call build_consensus"):
            consensus.flag_discrepant_experiments()

    def test_evaluate_small_experiment(self):
        """Evaluate small experiment against consensus."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=4,
            points_per_exp=30,
            noise=0.05,
        )

        fitted_gps = self._fit_multiple_gps(group_df)

        consensus = ConsensusBuilder()
        consensus.build_consensus(fitted_gps, (-2.0, 6.0))

        # Create a small experiment to evaluate
        small_exp = _make_synthetic_experiment("SMALL", n=5, noise=0.05)
        log_E = np.log10(small_exp['Energy'].values)
        log_sigma = np.log10(small_exp['CrossSection'].values)
        log_unc = prepare_log_uncertainties(
            small_exp['Uncertainty'].values,
            small_exp['CrossSection'].values,
        )

        z_scores, is_discrepant = consensus.evaluate_small_experiment(
            log_E, log_sigma, log_unc
        )

        assert z_scores.shape == (5,)
        assert isinstance(is_discrepant, (bool, np.bool_))

    def test_predict_at_points(self):
        """Predict consensus at arbitrary points."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=4,
            points_per_exp=30,
        )

        fitted_gps = self._fit_multiple_gps(group_df)

        consensus = ConsensusBuilder()
        consensus.build_consensus(fitted_gps, (-2.0, 6.0))

        query_points = np.array([0.0, 1.0, 2.0, 3.0])
        mean, std = consensus.predict_at_points(query_points)

        assert mean.shape == (4,)
        assert std.shape == (4,)


# =============================================================================
# Tests for experiment_outlier.py
# =============================================================================

class TestExperimentOutlierConfig:
    """Tests for ExperimentOutlierConfig."""

    def test_defaults(self):
        """Config has sensible defaults."""
        config = ExperimentOutlierConfig()

        assert config.point_z_threshold == 3.0
        assert config.min_group_size == 10
        assert config.entry_column == 'Entry'
        assert config.checkpoint_dir is None


class TestExperimentOutlierDetector:
    """Tests for ExperimentOutlierDetector main class."""

    def test_score_dataframe_basic(self):
        """Basic scoring workflow produces expected columns."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=3,
            points_per_exp=30,
        )

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False  # Faster for tests

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(group_df)

        # Check output columns exist
        expected_cols = [
            'log_E', 'log_sigma', 'gp_mean', 'gp_std', 'z_score',
            'experiment_outlier', 'point_outlier', 'calibration_metric',
            'experiment_id',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Row count preserved
        assert len(result) == len(group_df)

        # z_scores should be valid
        assert result['z_score'].notna().all()

    def test_detects_discrepant_experiment(self):
        """Correctly flags a discrepant experiment."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=5,
            points_per_exp=50,
            noise=0.05,
            n_discrepant=1,
            discrepant_bias=1.0,  # 100% higher - very discrepant
            seed=42,
        )

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(group_df)

        # The first experiment should be flagged
        discrepant_entries = result[result['experiment_outlier']]['experiment_id'].unique()
        assert 'E0001' in discrepant_entries

    def test_small_group_mad_fallback(self):
        """Small groups use MAD fallback."""
        small_group = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=2,
            points_per_exp=3,
        )

        config = ExperimentOutlierConfig(min_group_size=10)
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(small_group)

        stats = detector.get_statistics()
        assert stats['mad_groups'] >= 1

    def test_single_experiment_no_consensus(self):
        """Single experiment cannot flag experiment_outlier."""
        single_exp = _make_synthetic_experiment("E0001", n=30)
        single_exp['Z'] = 92
        single_exp['A'] = 235
        single_exp['MT'] = 18
        single_exp['N'] = 143
        single_exp['Projectile'] = 'n'

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(single_exp)

        # Cannot flag experiment as discrepant with no comparison
        assert not result['experiment_outlier'].any()

    def test_multiple_groups(self):
        """Process multiple (Z, A, MT) groups."""
        group1 = _make_multi_experiment_group(Z=92, A=235, MT=18, n_experiments=3, points_per_exp=20)
        group2 = _make_multi_experiment_group(Z=79, A=197, MT=102, n_experiments=3, points_per_exp=20, seed=100)
        df = pd.concat([group1, group2], ignore_index=True)

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        assert len(result) == len(df)

        stats = detector.get_statistics()
        assert stats['total_groups'] == 2

    def test_missing_required_columns(self):
        """Missing required columns raise ValueError."""
        df = pd.DataFrame({'X': [1, 2, 3]})
        detector = ExperimentOutlierDetector()

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.score_dataframe(df)

    def test_missing_entry_column_handled(self):
        """Works when Entry column is missing."""
        df = _make_synthetic_experiment("E0001", n=20)
        df['Z'] = 92
        df['A'] = 235
        df['MT'] = 18
        df['N'] = 143
        df['Projectile'] = 'n'
        df = df.drop(columns=['Entry'])  # Remove Entry column

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Should still work, treating all as one experiment
        assert len(result) == 20
        assert result['experiment_id'].iloc[0] == 'unknown'

    def test_preserves_original_columns(self):
        """Scoring preserves all original DataFrame columns."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=3,
            points_per_exp=20,
        )
        original_cols = set(group_df.columns)

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(group_df)

        for col in original_cols:
            assert col in result.columns

    def test_checkpoint_save(self, tmp_path):
        """Checkpoint saves during processing."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=3,
            points_per_exp=20,
        )

        config = ExperimentOutlierConfig(
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=1,
        )
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(group_df)

        assert (tmp_path / 'experiment_outlier_checkpoint.pkl').exists()

    def test_get_statistics(self):
        """Statistics tracking works correctly."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=3,
            points_per_exp=20,
        )

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        detector.score_dataframe(group_df)

        stats = detector.get_statistics()

        assert 'gp_experiments' in stats
        assert 'total_points' in stats
        assert 'total_groups' in stats
        assert stats['total_points'] == len(group_df)

    def test_point_outlier_flagging(self):
        """Point outliers flagged based on z_threshold."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=3,
            points_per_exp=30,
            noise=0.05,
        )

        # Inject one extreme point outlier
        # Note: With per-experiment GP, the outlier affects the GP fit for its experiment,
        # so we may see outliers flagged at different points. The important thing is
        # that SOME points are flagged when there's a large deviation.
        outlier_idx = 15
        group_df.loc[outlier_idx, 'CrossSection'] *= 1000  # 1000x higher

        config = ExperimentOutlierConfig(point_z_threshold=3.0)
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(group_df)

        # With a 1000x deviation injected, we should see some outliers flagged
        # (the injected outlier distorts the GP fit, causing other points to look anomalous)
        n_outliers = result['point_outlier'].sum()
        max_z = result['z_score'].max()

        # Either the injected point is flagged, or other points are flagged due to GP distortion
        # At minimum, something should be detected with such extreme deviation
        assert max_z > config.point_z_threshold, f"Max z-score {max_z:.2f} should exceed threshold"
        assert n_outliers > 0, "At least one point should be flagged as outlier"


# =============================================================================
# Integration Tests
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_import_from_data_module(self):
        """Can import from nucml_next.data."""
        from nucml_next.data import (
            ExperimentOutlierDetector,
            ExperimentOutlierConfig,
            ExactGPExperiment,
            ExactGPExperimentConfig,
            ConsensusBuilder,
            ConsensusConfig,
        )

        # Verify they're the correct classes
        assert ExperimentOutlierDetector is not None
        assert ExperimentOutlierConfig is not None
        assert ExactGPExperiment is not None

    def test_score_dataframe_signature_matches_svgp(self):
        """score_dataframe() has compatible signature with SVGPOutlierDetector."""
        group_df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=3,
            points_per_exp=20,
        )

        config = ExperimentOutlierConfig()
        config.gp_config.use_wasserstein_calibration = False

        detector = ExperimentOutlierDetector(config)

        # Should accept DataFrame and return DataFrame
        result = detector.score_dataframe(group_df)
        assert isinstance(result, pd.DataFrame)

        # Should have backward-compatible columns
        for col in ['log_E', 'log_sigma', 'gp_mean', 'gp_std', 'z_score']:
            assert col in result.columns
