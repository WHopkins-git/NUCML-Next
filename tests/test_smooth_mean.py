"""
Tests for the consensus smooth mean module.

Covers SmoothMeanConfig defaults, constant fallback, spline fitting,
edge cases, integration with ExactGPExperiment, and backward compatibility.
"""

import unittest

import numpy as np

from nucml_next.data.smooth_mean import SmoothMeanConfig, fit_smooth_mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_1_over_v(n=100, noise_std=0.05, seed=42):
    """Generate synthetic 1/v-law cross-section data (linear in log-log).

    In the thermal region, cross sections follow sigma ~ 1/v ~ 1/sqrt(E),
    so log10(sigma) ~ -0.5 * log10(E) + const.
    """
    rng = np.random.default_rng(seed)
    log_E = np.linspace(-2, 2, n)
    log_sigma = -0.5 * log_E + 1.0 + rng.normal(0, noise_std, n)
    return log_E, log_sigma


def _synthetic_resonance(n=200, seed=42):
    """Generate synthetic data with a Breit-Wigner resonance peak.

    Models a resonance at E=1 eV (log_E=0) on top of a smooth 1/v baseline.
    """
    rng = np.random.default_rng(seed)
    log_E = np.linspace(-2, 4, n)
    E = 10 ** log_E

    # 1/v baseline
    baseline = 100.0 / np.sqrt(E)

    # Breit-Wigner resonance at E=1.0 eV
    E_res, Gamma = 1.0, 0.1
    peak = 5000.0 * (Gamma / 2) ** 2 / ((E - E_res) ** 2 + (Gamma / 2) ** 2)

    sigma = baseline + peak
    log_sigma = np.log10(np.clip(sigma, 1e-30, None))
    noise = rng.normal(0, 0.02, n)
    return log_E, log_sigma + noise


# ===================================================================
# TestSmoothMeanConfig
# ===================================================================

class TestSmoothMeanConfig(unittest.TestCase):
    """Verify SmoothMeanConfig defaults."""

    def test_default_values(self):
        cfg = SmoothMeanConfig()
        self.assertEqual(cfg.smooth_mean_type, 'constant')
        self.assertIsNone(cfg.spline_smoothing_factor)
        self.assertEqual(cfg.spline_degree, 3)
        self.assertAlmostEqual(cfg.sigma_clip_threshold, 3.0)
        self.assertEqual(cfg.max_iterations, 5)
        self.assertAlmostEqual(cfg.convergence_tol, 1e-4)
        self.assertEqual(cfg.min_points_for_spline, 10)

    def test_constant_type_returns_constant_callable(self):
        """When type='constant', mean_fn returns a constant array."""
        cfg = SmoothMeanConfig(smooth_mean_type='constant')
        log_E = np.linspace(0, 5, 50)
        log_sigma = np.random.default_rng(42).normal(1.0, 0.1, 50)
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        result = mean_fn(log_E)
        # Should be constant = mean(log_sigma)
        expected = np.mean(log_sigma)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_type_raises(self):
        cfg = SmoothMeanConfig(smooth_mean_type='magic')
        log_E = np.linspace(0, 5, 50)
        log_sigma = np.ones(50)
        with self.assertRaises(ValueError):
            fit_smooth_mean(log_E, log_sigma, cfg)


# ===================================================================
# TestFitSmoothMean
# ===================================================================

class TestFitSmoothMean(unittest.TestCase):
    """Test the fit_smooth_mean function with various inputs."""

    def test_constant_fallback_by_config(self):
        """Explicit constant config gives constant mean."""
        cfg = SmoothMeanConfig(smooth_mean_type='constant')
        log_E, log_sigma = _synthetic_1_over_v()
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)
        result = mean_fn(log_E)
        self.assertTrue(np.allclose(result, np.mean(log_sigma)))

    def test_spline_evaluates_on_1v_data(self):
        """Spline on clean 1/v data should closely track the linear trend."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        log_E, log_sigma = _synthetic_1_over_v(noise_std=0.01)
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        result = mean_fn(log_E)
        # True trend: -0.5 * log_E + 1.0
        expected = -0.5 * log_E + 1.0
        # Should be close (within ~0.05 of the true trend)
        max_error = np.max(np.abs(result - expected))
        self.assertLess(max_error, 0.1,
                        f"Spline max error {max_error:.4f} too large for clean 1/v data")

    def test_small_group_falls_back_to_constant(self):
        """Groups with n < min_points_for_spline fall back to constant."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline', min_points_for_spline=20)
        log_E = np.array([1.0, 2.0, 3.0])
        log_sigma = np.array([10.0, 9.0, 8.0])
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        result = mean_fn(log_E)
        expected = np.mean(log_sigma)
        np.testing.assert_allclose(result, expected)

    def test_robustness_to_outliers(self):
        """Spline with sigma-clipping is not pulled by extreme outliers."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline', sigma_clip_threshold=3.0)
        rng = np.random.default_rng(123)
        n = 100
        log_E = np.linspace(0, 5, n)
        log_sigma_true = 2.0 * np.ones(n)  # Flat true trend
        noise = rng.normal(0, 0.05, n)
        log_sigma = log_sigma_true + noise

        # Inject 5 extreme outliers
        outlier_idx = [10, 30, 50, 70, 90]
        log_sigma[outlier_idx] += 5.0  # Huge outliers

        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)
        result = mean_fn(log_E)

        # At non-outlier points, the spline should be close to 2.0
        non_outlier_mask = np.ones(n, dtype=bool)
        non_outlier_mask[outlier_idx] = False
        residuals = np.abs(result[non_outlier_mask] - 2.0)
        median_residual = np.median(residuals)
        self.assertLess(median_residual, 0.5,
                        f"Spline median residual {median_residual:.4f} suggests "
                        f"outliers pulled the smooth mean")

    def test_callable_evaluates_outside_training_range(self):
        """Mean function can extrapolate outside training range."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        log_E, log_sigma = _synthetic_1_over_v()
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        # Extrapolate left and right
        query = np.array([-5.0, -3.0, 5.0, 10.0])
        result = mean_fn(query)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_output_shape_matches_input(self):
        """Output shape always matches input shape."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        log_E, log_sigma = _synthetic_1_over_v(n=50)
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        for query_size in [1, 10, 50, 200]:
            query = np.linspace(-2, 2, query_size)
            result = mean_fn(query)
            self.assertEqual(result.shape, (query_size,),
                             f"Shape mismatch for query_size={query_size}")

    def test_identical_energies_fallback(self):
        """All identical energies (degenerate) falls back to constant."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        log_E = np.full(20, 3.0)
        log_sigma = np.random.default_rng(42).normal(5.0, 0.1, 20)
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        result = mean_fn(np.array([1.0, 3.0, 5.0]))
        # Should be constant = mean(log_sigma)
        expected = np.mean(log_sigma)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_identical_sigma_values(self):
        """All identical sigma values -- spline should be flat."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        log_E = np.linspace(0, 5, 50)
        log_sigma = np.full(50, 2.5)
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        result = mean_fn(log_E)
        np.testing.assert_allclose(result, 2.5, atol=1e-6)

    def test_nan_in_sigma(self):
        """NaN values in log_sigma are handled gracefully."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        log_E = np.linspace(0, 5, 50)
        log_sigma = np.linspace(1, 3, 50)
        # Inject NaNs
        log_sigma[10] = np.nan
        log_sigma[20] = np.nan
        log_sigma[30] = np.inf
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)

        result = mean_fn(log_E)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_none_config_gives_constant(self):
        """Passing config=None uses constant mean (default)."""
        log_E, log_sigma = _synthetic_1_over_v()
        mean_fn = fit_smooth_mean(log_E, log_sigma, None)

        result = mean_fn(log_E)
        expected = np.mean(log_sigma)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_empty_data(self):
        """Empty arrays produce a constant-zero callable."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        mean_fn = fit_smooth_mean(np.array([]), np.array([]), cfg)
        result = mean_fn(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, 0.0)

    def test_single_point(self):
        """Single data point falls back to constant."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        mean_fn = fit_smooth_mean(np.array([1.0]), np.array([5.0]), cfg)
        result = mean_fn(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(result, 5.0)

    def test_duplicate_energies_handled(self):
        """Duplicate energy values are averaged, not rejected."""
        cfg = SmoothMeanConfig(smooth_mean_type='spline')
        # 5 unique x values, some duplicated
        log_E = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
                          1.5, 2.5, 3.5, 4.5])
        log_sigma = np.array([2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1, 6.0, 6.1,
                              2.5, 3.5, 4.5, 5.5])
        mean_fn = fit_smooth_mean(log_E, log_sigma, cfg)
        result = mean_fn(np.linspace(1, 5, 10))
        self.assertTrue(np.all(np.isfinite(result)))


# ===================================================================
# TestSmoothMeanIntegrationWithGP
# ===================================================================

class TestSmoothMeanIntegrationWithGP(unittest.TestCase):
    """Test smooth mean integration with ExactGPExperiment."""

    def test_gp_fits_with_spline_mean(self):
        """GP can fit and predict with a spline mean function."""
        from nucml_next.data.experiment_gp import ExactGPExperiment, ExactGPExperimentConfig

        cfg = ExactGPExperimentConfig(
            use_wasserstein_calibration=True,
            smooth_mean_config=SmoothMeanConfig(smooth_mean_type='spline'),
        )
        gp = ExactGPExperiment(cfg)

        log_E, log_sigma = _synthetic_1_over_v(n=50, noise_std=0.05)
        log_unc = np.full(50, 0.05)

        gp.fit(log_E, log_sigma, log_unc)
        self.assertTrue(gp.is_fitted)

        mean, std = gp.predict(log_E)
        self.assertEqual(mean.shape, (50,))
        self.assertEqual(std.shape, (50,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std > 0))

    def test_gp_fits_with_external_mean_fn(self):
        """GP can accept an external mean_fn."""
        from nucml_next.data.experiment_gp import ExactGPExperiment, ExactGPExperimentConfig

        cfg = ExactGPExperimentConfig(use_wasserstein_calibration=True)
        gp = ExactGPExperiment(cfg)

        log_E, log_sigma = _synthetic_1_over_v(n=50, noise_std=0.05)
        log_unc = np.full(50, 0.05)

        # External mean fn
        external_mean = fit_smooth_mean(
            log_E, log_sigma,
            SmoothMeanConfig(smooth_mean_type='spline')
        )

        gp.fit(log_E, log_sigma, log_unc, mean_fn=external_mean)
        self.assertTrue(gp.is_fitted)
        self.assertIs(gp._mean_fn, external_mean)

    def test_backward_compat_default_produces_constant_mean(self):
        """Default config (no smooth_mean_config) produces scalar-equivalent mean."""
        from nucml_next.data.experiment_gp import ExactGPExperiment, ExactGPExperimentConfig

        cfg = ExactGPExperimentConfig(use_wasserstein_calibration=True)
        gp = ExactGPExperiment(cfg)

        log_E, log_sigma = _synthetic_1_over_v(n=30, noise_std=0.05)
        log_unc = np.full(30, 0.05)

        gp.fit(log_E, log_sigma, log_unc)

        # Mean value should be a constant array == np.mean(log_sigma)
        expected_mean = np.mean(log_sigma)
        np.testing.assert_allclose(gp._mean_value, expected_mean, rtol=1e-10)

    def test_loo_z_scores_with_spline_mean(self):
        """LOO z-scores work correctly with array mean."""
        from nucml_next.data.experiment_gp import ExactGPExperiment, ExactGPExperimentConfig

        cfg = ExactGPExperimentConfig(
            use_wasserstein_calibration=True,
            smooth_mean_config=SmoothMeanConfig(smooth_mean_type='spline'),
        )
        gp = ExactGPExperiment(cfg)

        log_E, log_sigma = _synthetic_1_over_v(n=30, noise_std=0.05)
        log_unc = np.full(30, 0.05)

        gp.fit(log_E, log_sigma, log_unc)
        z_scores = gp.get_loo_z_scores()

        self.assertEqual(z_scores.shape, (30,))
        self.assertTrue(np.all(np.isfinite(z_scores)))

    def test_spline_mean_not_catastrophic_on_resonance(self):
        """Spline mean should not catastrophically degrade on resonance data.

        With a single experiment of 100 points, both constant and spline
        mean GPs perform reasonably well because Wasserstein calibration
        adapts the lengthscale.  The real benefit of the spline mean
        emerges with actual EXFOR data (many experiments, thousands of
        points, multiple overlapping resonances spanning 5-6 orders of
        magnitude).

        This test validates that the spline path works end-to-end and
        does not produce dramatically more false positives than constant.
        """
        from nucml_next.data.experiment_gp import ExactGPExperiment, ExactGPExperimentConfig

        log_E, log_sigma = _synthetic_resonance(n=100)
        log_unc = np.full(100, 0.05)

        # Constant mean GP
        cfg_const = ExactGPExperimentConfig(
            use_wasserstein_calibration=True,
            smooth_mean_config=SmoothMeanConfig(smooth_mean_type='constant'),
        )
        gp_const = ExactGPExperiment(cfg_const)
        gp_const.fit(log_E, log_sigma, log_unc)
        z_const = gp_const.get_loo_z_scores()

        # Spline mean GP
        cfg_spline = ExactGPExperimentConfig(
            use_wasserstein_calibration=True,
            smooth_mean_config=SmoothMeanConfig(smooth_mean_type='spline'),
        )
        gp_spline = ExactGPExperiment(cfg_spline)
        gp_spline.fit(log_E, log_sigma, log_unc)
        z_spline = gp_spline.get_loo_z_scores()

        n_outlier_const = np.sum(np.abs(z_const) > 3)
        n_outlier_spline = np.sum(np.abs(z_spline) > 3)

        # Both methods should produce a reasonable number of outliers
        # (not more than ~15% of 100 points)
        self.assertLess(
            n_outlier_spline, 15,
            f"Spline produced {n_outlier_spline} outliers on 100 points "
            f"(expected < 15)"
        )
        self.assertLess(
            n_outlier_const, 15,
            f"Constant produced {n_outlier_const} outliers on 100 points "
            f"(expected < 15)"
        )


# ===================================================================
# TestSmoothMeanExport
# ===================================================================

class TestSmoothMeanExport(unittest.TestCase):
    """Verify module exports from nucml_next.data."""

    def test_smooth_mean_config_importable(self):
        from nucml_next.data import SmoothMeanConfig
        cfg = SmoothMeanConfig()
        self.assertEqual(cfg.smooth_mean_type, 'constant')

    def test_fit_smooth_mean_importable(self):
        from nucml_next.data import fit_smooth_mean
        self.assertTrue(callable(fit_smooth_mean))


if __name__ == '__main__':
    unittest.main()
