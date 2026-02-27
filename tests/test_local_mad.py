"""Tests for the local_mad scoring method.

Tests the compute_rolling_mad_interpolator() function in smooth_mean.py,
the _score_group_local_mad() method in experiment_outlier.py, and the full
pipeline integration for the local_mad scoring method.
"""

import numpy as np
import pandas as pd
import pytest

from nucml_next.data.smooth_mean import (
    SmoothMeanConfig,
    fit_smooth_mean,
    compute_rolling_mad_interpolator,
)
from nucml_next.data.experiment_outlier import (
    ExperimentOutlierConfig,
    ExperimentOutlierDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n=200, seed=42):
    """Create synthetic cross-section data with a smooth trend and noise."""
    rng = np.random.RandomState(seed)
    log_E = np.sort(rng.uniform(-2, 8, n))
    trend = 2.0 - 0.3 * log_E
    noise = rng.normal(0, 0.1, n)
    log_sigma = trend + noise

    config = SmoothMeanConfig(smooth_mean_type='spline')
    mean_fn = fit_smooth_mean(log_E, log_sigma, config)

    return log_E, log_sigma, mean_fn


def _make_volatile_region_data(n=300, seed=42):
    """Create data with one volatile region (resonance) and one smooth region.

    Smooth region: log_E in [-2, 3) with noise std ~ 0.05
    Volatile region: log_E in [3, 8] with noise std ~ 0.5
    """
    rng = np.random.RandomState(seed)

    # Smooth region
    n_smooth = n // 2
    log_E_smooth = np.sort(rng.uniform(-2, 3, n_smooth))
    trend_smooth = 2.0 - 0.3 * log_E_smooth
    noise_smooth = rng.normal(0, 0.05, n_smooth)

    # Volatile region (resonance-like)
    n_volatile = n - n_smooth
    log_E_volatile = np.sort(rng.uniform(3, 8, n_volatile))
    trend_volatile = 2.0 - 0.3 * log_E_volatile
    noise_volatile = rng.normal(0, 0.5, n_volatile)

    log_E = np.concatenate([log_E_smooth, log_E_volatile])
    log_sigma = np.concatenate([
        trend_smooth + noise_smooth,
        trend_volatile + noise_volatile,
    ])

    config = SmoothMeanConfig(smooth_mean_type='spline')
    mean_fn = fit_smooth_mean(log_E, log_sigma, config)

    return log_E, log_sigma, mean_fn


def _make_multi_experiment_df(n_experiments=3, n_per_exp=100, seed=42,
                              biased_exp=None, bias_offset=0.0):
    """Create a synthetic DataFrame with multiple experiments.

    Args:
        n_experiments: Number of experiments to create.
        n_per_exp: Points per experiment.
        seed: Random seed.
        biased_exp: Index of experiment to bias (0-based). None = no bias.
        bias_offset: Offset to add to the biased experiment.
    """
    rng = np.random.RandomState(seed)
    dfs = []

    for i in range(n_experiments):
        log_E = np.sort(rng.uniform(-2, 8, n_per_exp))
        trend = 2.0 - 0.3 * log_E
        noise = rng.normal(0, 0.1, n_per_exp)
        log_sigma = trend + noise

        if biased_exp is not None and i == biased_exp:
            log_sigma += bias_offset

        df = pd.DataFrame({
            'Z': 92,
            'A': 235,
            'MT': 18,
            'Energy': 10.0 ** log_E,
            'CrossSection': 10.0 ** log_sigma,
            'Entry': f'E{i:04d}',
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Test ExperimentOutlierConfig defaults and local_mad settings."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = ExperimentOutlierConfig()
        assert config.point_z_threshold == 3.0
        assert config.mad_use_measurement_uncertainty is True

    def test_local_mad_config_with_custom_thresholds(self):
        """Config should accept custom thresholds."""
        config = ExperimentOutlierConfig(
            exp_z_threshold=5.0,
            exp_fraction_threshold=0.20,
            mad_floor=0.05,
        )
        assert config.exp_z_threshold == 5.0
        assert config.exp_fraction_threshold == 0.20
        assert config.mad_floor == 0.05


# ---------------------------------------------------------------------------
# Rolling MAD interpolator tests
# ---------------------------------------------------------------------------

class TestRollingMadInterpolator:
    """Tests for compute_rolling_mad_interpolator()."""

    def test_mad_basic(self):
        """Synthetic 1/v data should return callable with MAD > 0 everywhere."""
        log_E, log_sigma, mean_fn = _make_synthetic_data()
        mad_fn = compute_rolling_mad_interpolator(log_E, log_sigma, mean_fn)

        assert mad_fn is not None
        assert callable(mad_fn)

        # Evaluate at original and new points
        mad_values = mad_fn(log_E)
        assert len(mad_values) == len(log_E)
        assert np.all(mad_values > 0)
        assert np.all(np.isfinite(mad_values))

    def test_mad_volatile_region(self):
        """Volatile region should have higher MAD than smooth region."""
        log_E, log_sigma, mean_fn = _make_volatile_region_data()
        mad_fn = compute_rolling_mad_interpolator(log_E, log_sigma, mean_fn)

        assert mad_fn is not None

        # Sample points in smooth vs volatile regions
        smooth_E = np.array([0.0, 1.0, 2.0])
        volatile_E = np.array([5.0, 6.0, 7.0])

        mad_smooth = mad_fn(smooth_E)
        mad_volatile = mad_fn(volatile_E)

        # Volatile region should have clearly higher MAD
        assert np.median(mad_volatile) > np.median(mad_smooth) * 2

    def test_mad_smooth_region(self):
        """Smooth sine wave should produce roughly uniform, small MAD."""
        rng = np.random.RandomState(42)
        n = 200
        log_E = np.sort(rng.uniform(-2, 8, n))
        trend = np.sin(log_E * 0.5) + 1.0  # smooth
        noise = rng.normal(0, 0.02, n)  # very small noise
        log_sigma = trend + noise

        config = SmoothMeanConfig(smooth_mean_type='spline')
        mean_fn = fit_smooth_mean(log_E, log_sigma, config)
        mad_fn = compute_rolling_mad_interpolator(log_E, log_sigma, mean_fn)

        assert mad_fn is not None
        mad_values = mad_fn(log_E)

        # MAD should be small (noise std is 0.02)
        assert np.median(mad_values) < 0.2

    def test_mad_floor(self):
        """All identical data should produce MAD = floor value."""
        n = 50
        log_E = np.linspace(-2, 8, n)
        log_sigma = np.ones(n) * 2.0  # constant data

        mean_fn = lambda x: np.full_like(x, 2.0)
        floor = 0.05
        mad_fn = compute_rolling_mad_interpolator(
            log_E, log_sigma, mean_fn, mad_floor=floor,
        )

        assert mad_fn is not None
        mad_values = mad_fn(log_E)

        # All values should be exactly the floor (MAD of constant data is 0)
        np.testing.assert_array_equal(mad_values, floor)

    def test_mad_few_points_returns_none(self):
        """Fewer than 10 points should return None."""
        log_E = np.array([1.0, 2.0, 3.0])
        log_sigma = np.array([1.0, 1.5, 1.2])
        mean_fn = lambda x: np.full_like(x, 1.0)

        result = compute_rolling_mad_interpolator(log_E, log_sigma, mean_fn)
        assert result is None


# ---------------------------------------------------------------------------
# Point scoring tests
# ---------------------------------------------------------------------------

class TestPointScoring:
    """Tests for z-score computation in the local_mad scoring method."""

    def test_clean_data_low_outlier_rate(self):
        """Synthetic smooth data should have very low z > 3 rate."""
        df = _make_multi_experiment_df(n_experiments=3, n_per_exp=150, seed=42)

        config = ExperimentOutlierConfig(point_z_threshold=3.0)
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        z_scores = result['z_score'].values
        finite_z = z_scores[np.isfinite(z_scores)]
        outlier_rate = np.mean(finite_z > 3.0)

        # Should be < 10% (typically ~1-5%) for clean Gaussian data
        assert outlier_rate < 0.10, (
            f"Outlier rate {outlier_rate:.1%} is too high for clean data"
        )

    def test_injected_outliers_detected(self):
        """Points offset by large amount should have high z-scores."""
        rng = np.random.RandomState(42)
        n = 200
        log_E = np.sort(rng.uniform(-2, 8, n))
        trend = 2.0 - 0.3 * log_E
        noise = rng.normal(0, 0.1, n)
        log_sigma = trend + noise

        # Inject 5 extreme outliers
        outlier_indices = [10, 50, 100, 150, 190]
        for idx in outlier_indices:
            log_sigma[idx] += 5.0  # 50x the noise level

        df = pd.DataFrame({
            'Z': 92, 'A': 235, 'MT': 18,
            'Energy': 10.0 ** log_E,
            'CrossSection': 10.0 ** log_sigma,
            'Entry': 'E0001',
        })

        config = ExperimentOutlierConfig(point_z_threshold=5.0)
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Check that the injected outliers have high z-scores
        for idx in outlier_indices:
            assert result.iloc[idx]['z_score'] > 5.0, (
                f"Outlier at index {idx} has z_score={result.iloc[idx]['z_score']:.2f}, expected > 5"
            )

    def test_resonance_region_tolerant(self):
        """Resonance-like scatter should NOT be systematically flagged."""
        log_E, log_sigma, mean_fn = _make_volatile_region_data(n=400, seed=123)

        df = pd.DataFrame({
            'Z': 92, 'A': 235, 'MT': 18,
            'Energy': 10.0 ** log_E,
            'CrossSection': 10.0 ** log_sigma,
            'Entry': 'E0001',
        })

        config = ExperimentOutlierConfig(point_z_threshold=3.0)
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Focus on the volatile region (log_E > 3)
        volatile_mask = result['log_E'] > 3.0
        volatile_z = result.loc[volatile_mask, 'z_score'].values
        volatile_outlier_rate = np.mean(volatile_z > 3.0)

        # Should be < 15% -- the local MAD accommodates volatile regions
        assert volatile_outlier_rate < 0.15, (
            f"Volatile region outlier rate {volatile_outlier_rate:.1%} is too high. "
            f"Local MAD should accommodate resonance scatter."
        )

    def test_z_score_energy_local(self):
        """Same residual should get different z-scores at different energies."""
        log_E, log_sigma, mean_fn = _make_volatile_region_data()
        mad_fn = compute_rolling_mad_interpolator(log_E, log_sigma, mean_fn)

        assert mad_fn is not None

        # At smooth region (low MAD): same residual gets higher z
        # At volatile region (high MAD): same residual gets lower z
        mad_smooth = mad_fn(np.array([0.0]))[0]
        mad_volatile = mad_fn(np.array([6.0]))[0]

        residual = 0.5  # fixed residual
        z_smooth = residual / mad_smooth
        z_volatile = residual / mad_volatile

        assert z_smooth > z_volatile, (
            f"z_smooth={z_smooth:.2f} should be > z_volatile={z_volatile:.2f}"
        )


# ---------------------------------------------------------------------------
# Experiment discrepancy tests
# ---------------------------------------------------------------------------

class TestExperimentDiscrepancy:
    """Tests for experiment-level flagging in local_mad method."""

    def test_good_experiment(self):
        """Experiment with clean data should NOT be flagged."""
        df = _make_multi_experiment_df(n_experiments=3, n_per_exp=100)

        config = ExperimentOutlierConfig(
            exp_z_threshold=3.0,
            exp_fraction_threshold=0.30,
        )
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # No experiments should be flagged with clean data
        assert result['experiment_outlier'].sum() == 0

    def test_bad_experiment(self):
        """Experiment with large bias should be flagged as discrepant."""
        # Bias experiment 1 by 2.0 (massive offset)
        df = _make_multi_experiment_df(
            n_experiments=3, n_per_exp=100,
            biased_exp=1, bias_offset=2.0,
        )

        config = ExperimentOutlierConfig(
            exp_z_threshold=3.0,
            exp_fraction_threshold=0.30,
        )
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # The biased experiment should be flagged
        biased_mask = result['experiment_id'] == 'E0001'
        assert result.loc[biased_mask, 'experiment_outlier'].all(), (
            "Biased experiment E0001 should be flagged as discrepant"
        )

    def test_threshold_sensitivity(self):
        """Experiment flagging should be sensitive to threshold settings."""
        # Moderate bias — borderline case
        df = _make_multi_experiment_df(
            n_experiments=3, n_per_exp=100,
            biased_exp=1, bias_offset=0.8,
        )

        # Strict threshold (60%) — harder to flag
        config_strict = ExperimentOutlierConfig(
            exp_z_threshold=3.0,
            exp_fraction_threshold=0.60,
        )
        detector_strict = ExperimentOutlierDetector(config_strict)
        result_strict = detector_strict.score_dataframe(df)

        # Loose threshold (10%) — easier to flag
        config_loose = ExperimentOutlierConfig(
            exp_z_threshold=3.0,
            exp_fraction_threshold=0.10,
        )
        detector_loose = ExperimentOutlierDetector(config_loose)
        result_loose = detector_loose.score_dataframe(df)

        # Loose threshold should flag at least as many experiments as strict
        n_flagged_strict = result_strict['experiment_outlier'].sum()
        n_flagged_loose = result_loose['experiment_outlier'].sum()
        assert n_flagged_loose >= n_flagged_strict


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests for local_mad scoring."""

    def test_full_pipeline_local_mad(self):
        """Full pipeline with local_mad should complete and produce expected columns."""
        df = _make_multi_experiment_df(n_experiments=3, n_per_exp=100)

        config = ExperimentOutlierConfig()
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Check all expected columns exist
        for col in ['z_score', 'point_outlier', 'experiment_outlier',
                     'gp_mean', 'gp_std', 'calibration_metric',
                     'outlier_probability', 'experiment_id', 'log_E', 'log_sigma']:
            assert col in result.columns, f"Missing column: {col}"

        # Check z_scores are finite and non-negative
        z_scores = result['z_score'].values
        assert np.all(np.isfinite(z_scores))
        assert np.all(z_scores >= 0)

        # Check gp_std (local MAD) is positive everywhere
        gp_std = result['gp_std'].values
        assert np.all(gp_std > 0)

        # Check boolean columns
        assert result['point_outlier'].dtype == bool
        assert result['experiment_outlier'].dtype == bool

        # Check calibration_metric is NaN (not used by local_mad)
        assert result['calibration_metric'].isna().all()

    def test_single_experiment_group(self):
        """Single-experiment group should score correctly, not flagged as discrepant."""
        df = _make_multi_experiment_df(n_experiments=1, n_per_exp=100)

        config = ExperimentOutlierConfig()
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Single experiment cannot be flagged as discrepant
        assert not result['experiment_outlier'].any(), (
            "Single experiment should never be flagged as discrepant"
        )

        # But z-scores should still be computed
        assert np.all(np.isfinite(result['z_score'].values))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for local_mad scoring."""

    def test_all_same_experiment(self):
        """Group with one large experiment should work correctly."""
        rng = np.random.RandomState(42)
        n = 1000
        log_E = np.sort(rng.uniform(-2, 8, n))
        trend = 2.0 - 0.3 * log_E
        noise = rng.normal(0, 0.1, n)
        log_sigma = trend + noise

        df = pd.DataFrame({
            'Z': 92, 'A': 235, 'MT': 18,
            'Energy': 10.0 ** log_E,
            'CrossSection': 10.0 ** log_sigma,
            'Entry': 'E0001',
        })

        config = ExperimentOutlierConfig()
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Should complete without error
        assert len(result) == n
        assert np.all(np.isfinite(result['z_score'].values))

    def test_tiny_group(self):
        """Group with fewer than min_group_size points falls back to MAD."""
        df = pd.DataFrame({
            'Z': 92, 'A': 235, 'MT': 18,
            'Energy': [1.0, 2.0, 3.0, 4.0, 5.0],
            'CrossSection': [10.0, 11.0, 9.0, 10.5, 50.0],  # last is outlier
            'Entry': 'E0001',
        })

        config = ExperimentOutlierConfig(min_group_size=10)  # > 5 points
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Should fall back to MAD but not crash
        assert len(result) == 5
        assert np.all(np.isfinite(result['z_score'].values))


# ---------------------------------------------------------------------------
# Measurement uncertainty integration tests
# ---------------------------------------------------------------------------

def _make_data_with_uncertainty(n=200, seed=42, rel_unc=0.03):
    """Create synthetic data with EXFOR-like reported uncertainties.

    Args:
        n: Number of data points.
        seed: Random seed.
        rel_unc: Relative uncertainty (d_sigma/sigma) for each point.

    Returns:
        DataFrame with Z, A, MT, Energy, CrossSection, Uncertainty, Entry columns.
    """
    rng = np.random.RandomState(seed)
    log_E = np.sort(rng.uniform(-2, 8, n))
    trend = 2.0 - 0.3 * log_E
    noise = rng.normal(0, 0.1, n)
    log_sigma = trend + noise

    energy = 10.0 ** log_E
    cross_section = 10.0 ** log_sigma
    uncertainty = cross_section * rel_unc  # absolute uncertainty

    return pd.DataFrame({
        'Z': 92,
        'A': 235,
        'MT': 18,
        'Energy': energy,
        'CrossSection': cross_section,
        'Uncertainty': uncertainty,
        'Entry': 'E0001',
    })


class TestMeasurementUncertainty:
    """Tests for measurement uncertainty integration in local_mad scoring."""

    def test_uncertainty_reduces_z_scores(self):
        """Including measurement uncertainty should reduce z-scores."""
        df = _make_data_with_uncertainty(n=200, rel_unc=0.05)

        # Without uncertainty
        config_no_unc = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=False,
        )
        result_no_unc = ExperimentOutlierDetector(config_no_unc).score_dataframe(df)

        # With uncertainty
        config_with_unc = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=True,
        )
        result_with_unc = ExperimentOutlierDetector(config_with_unc).score_dataframe(df)

        z_no_unc = result_no_unc['z_score'].values
        z_with_unc = result_with_unc['z_score'].values

        # z-scores with uncertainty should be <= those without (for every point)
        assert np.all(z_with_unc <= z_no_unc + 1e-10), (
            "Uncertainty should never increase z-scores"
        )

        # Mean z-score should be meaningfully lower
        assert np.mean(z_with_unc) < np.mean(z_no_unc), (
            "Mean z-score should decrease when measurement uncertainty is included"
        )

    def test_no_uncertainty_column_unchanged(self):
        """Data without Uncertainty column should behave like mad_use_measurement_uncertainty=False."""
        df = _make_multi_experiment_df(n_experiments=1, n_per_exp=200)
        # This df has no 'Uncertainty' column

        config_on = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=True,
        )
        config_off = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=False,
        )

        result_on = ExperimentOutlierDetector(config_on).score_dataframe(df)
        result_off = ExperimentOutlierDetector(config_off).score_dataframe(df)

        # Should be identical since there's no Uncertainty column
        np.testing.assert_array_almost_equal(
            result_on['z_score'].values,
            result_off['z_score'].values,
        )

    def test_large_uncertainty_suppresses_outliers(self):
        """Large reported uncertainties should suppress false positives."""
        # Create data with large noise but also large reported uncertainty
        rng = np.random.RandomState(42)
        n = 300
        log_E = np.sort(rng.uniform(-2, 8, n))
        trend = 2.0 - 0.3 * log_E
        noise = rng.normal(0, 0.3, n)  # Large noise
        log_sigma = trend + noise

        energy = 10.0 ** log_E
        cross_section = 10.0 ** log_sigma
        # Report 30% relative uncertainty (much larger than MAD floor)
        uncertainty = cross_section * 0.30

        df = pd.DataFrame({
            'Z': 92, 'A': 235, 'MT': 18,
            'Energy': energy,
            'CrossSection': cross_section,
            'Uncertainty': uncertainty,
            'Entry': 'E0001',
        })

        config = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=True,
            point_z_threshold=3.0,
        )
        result = ExperimentOutlierDetector(config).score_dataframe(df)

        outlier_rate = result['point_outlier'].mean()
        # With 30% reported uncertainty, the effective sigma is large,
        # so very few points should be flagged
        assert outlier_rate < 0.05, (
            f"Outlier rate {outlier_rate:.1%} should be < 5% with large "
            f"reported uncertainties"
        )

    def test_effective_sigma_stored_in_gp_std(self):
        """gp_std should contain effective_sigma (not raw local_mad) when uncertainty is used."""
        df = _make_data_with_uncertainty(n=200, rel_unc=0.10)

        config = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=True,
        )
        result = ExperimentOutlierDetector(config).score_dataframe(df)

        config_no_unc = ExperimentOutlierConfig(
            mad_use_measurement_uncertainty=False,
        )
        result_no_unc = ExperimentOutlierDetector(config_no_unc).score_dataframe(df)

        # gp_std with uncertainty should be >= gp_std without (quadrature sum)
        assert np.all(
            result['gp_std'].values >= result_no_unc['gp_std'].values - 1e-10
        ), "effective_sigma >= local_mad always (quadrature sum)"
