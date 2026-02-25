"""Tests for data-driven lengthscale estimation.

Tests the compute_lengthscale_from_residuals() function in smooth_mean.py,
its integration with the Gibbs kernel in kernels.py, and end-to-end wiring
through experiment_outlier.py.
"""

import numpy as np
import pytest

from nucml_next.data.smooth_mean import (
    SmoothMeanConfig,
    fit_smooth_mean,
    compute_lengthscale_from_residuals,
    _softplus_inverse,
)
from nucml_next.data.kernels import (
    KernelConfig,
    GibbsKernel,
    build_kernel,
    _softplus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n=200, seed=42):
    """Create synthetic cross-section data with a smooth trend and noise.

    Returns (log_E, log_sigma, mean_fn) where mean_fn is a simple callable.
    """
    rng = np.random.RandomState(seed)
    log_E = np.sort(rng.uniform(-2, 8, n))
    # Smooth 1/v-like trend
    trend = 2.0 - 0.3 * log_E
    noise = rng.normal(0, 0.1, n)
    log_sigma = trend + noise

    config = SmoothMeanConfig(smooth_mean_type='spline')
    mean_fn = fit_smooth_mean(log_E, log_sigma, config)

    return log_E, log_sigma, mean_fn


def _make_volatile_region_data(n=300, seed=42):
    """Create data with one volatile region (resonance) and one smooth region.

    Volatile region: log_E in [2, 4] with high noise (sigma=0.5).
    Smooth region: elsewhere with low noise (sigma=0.05).
    """
    rng = np.random.RandomState(seed)
    log_E = np.sort(rng.uniform(-1, 7, n))
    trend = 1.5 - 0.2 * log_E

    noise = np.where(
        (log_E >= 2) & (log_E <= 4),
        rng.normal(0, 0.5, n),    # volatile
        rng.normal(0, 0.05, n),   # smooth
    )
    log_sigma = trend + noise

    config = SmoothMeanConfig(smooth_mean_type='spline')
    mean_fn = fit_smooth_mean(log_E, log_sigma, config)

    return log_E, log_sigma, mean_fn


# ===========================================================================
# Tests for _softplus_inverse
# ===========================================================================

class TestSoftplusInverse:
    def test_roundtrip(self):
        """softplus(softplus_inverse(x)) == x for positive x."""
        x = np.array([0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        recovered = _softplus(_softplus_inverse(x))
        np.testing.assert_allclose(recovered, x, rtol=1e-6)

    def test_large_values(self):
        """For large x, softplus_inverse(x) ≈ x."""
        x = np.array([25.0, 50.0])
        np.testing.assert_allclose(_softplus_inverse(x), x, atol=1e-6)

    def test_small_positive(self):
        """Works for small positive values."""
        x = np.array([0.001, 0.01])
        result = _softplus_inverse(x)
        assert np.all(np.isfinite(result))
        # softplus_inverse of small values should be large negative
        assert np.all(result < 0)


# ===========================================================================
# Tests for compute_lengthscale_from_residuals
# ===========================================================================

class TestComputeLengthscaleFromResiduals:
    def test_basic_returns_callable(self):
        """Basic synthetic data returns a callable."""
        log_E, log_sigma, mean_fn = _make_synthetic_data()
        result = compute_lengthscale_from_residuals(log_E, log_sigma, mean_fn)
        assert callable(result)

        # Can evaluate at query points
        query = np.linspace(-2, 8, 50)
        signal = result(query)
        assert signal.shape == (50,)
        assert np.all(np.isfinite(signal))

    def test_high_variability_short_lengthscale(self):
        """Volatile region gets shorter lengthscale than smooth region."""
        log_E, log_sigma, mean_fn = _make_volatile_region_data()
        interp_fn = compute_lengthscale_from_residuals(
            log_E, log_sigma, mean_fn,
        )
        assert interp_fn is not None

        # Evaluate signal in volatile vs smooth regions
        volatile_x = np.array([2.5, 3.0, 3.5])
        smooth_x = np.array([0.0, 5.5, 6.0])

        signal_volatile = interp_fn(volatile_x)
        signal_smooth = interp_fn(smooth_x)

        # Convert signal to lengthscale via softplus
        ell_volatile = _softplus(signal_volatile)
        ell_smooth = _softplus(signal_smooth)

        # Volatile region should have shorter average lengthscale
        assert np.mean(ell_volatile) < np.mean(ell_smooth), (
            f"Volatile ℓ={np.mean(ell_volatile):.3f} should be < "
            f"smooth ℓ={np.mean(ell_smooth):.3f}"
        )

    def test_nan_inf_handling(self):
        """NaN and inf values in input are filtered gracefully."""
        rng = np.random.RandomState(99)
        n = 100
        log_E = np.sort(rng.uniform(0, 6, n))
        log_sigma = 1.0 + rng.normal(0, 0.1, n)

        # Inject some NaN and inf
        log_E_dirty = log_E.copy()
        log_sigma_dirty = log_sigma.copy()
        log_sigma_dirty[5] = np.nan
        log_sigma_dirty[10] = np.inf
        log_E_dirty[15] = np.nan

        config = SmoothMeanConfig(smooth_mean_type='spline')
        mean_fn = fit_smooth_mean(log_E, log_sigma, config)

        result = compute_lengthscale_from_residuals(
            log_E_dirty, log_sigma_dirty, mean_fn,
        )
        # Should still return a valid callable (after filtering)
        assert callable(result)
        query = np.array([3.0])
        signal = result(query)
        assert np.all(np.isfinite(signal))

    def test_output_range(self):
        """All softplus(signal) values within [min_l, max_l]."""
        log_E, log_sigma, mean_fn = _make_synthetic_data()
        min_l, max_l = 0.05, 1.5
        interp_fn = compute_lengthscale_from_residuals(
            log_E, log_sigma, mean_fn,
            min_lengthscale=min_l, max_lengthscale=max_l,
        )
        assert interp_fn is not None

        query = np.linspace(-2, 8, 100)
        signal = interp_fn(query)
        ell = _softplus(signal)

        # Allow small tolerance for softplus roundtrip
        assert np.all(ell >= min_l - 1e-6), f"min ℓ={ell.min():.6f} < {min_l}"
        assert np.all(ell <= max_l + 1e-6), f"max ℓ={ell.max():.6f} > {max_l}"

    def test_extrapolation(self):
        """Query outside training range gets edge values (not NaN)."""
        log_E, log_sigma, mean_fn = _make_synthetic_data()
        interp_fn = compute_lengthscale_from_residuals(log_E, log_sigma, mean_fn)
        assert interp_fn is not None

        # Far outside training range
        query = np.array([-10.0, 20.0])
        signal = interp_fn(query)
        assert np.all(np.isfinite(signal))

    def test_few_points_returns_none(self):
        """Fewer than 10 points returns None."""
        log_E = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        log_sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mean_fn = lambda x: np.full_like(x, 0.3)

        result = compute_lengthscale_from_residuals(log_E, log_sigma, mean_fn)
        assert result is None

    def test_constant_data(self):
        """All identical residuals → uniform ℓ ≈ max_lengthscale."""
        n = 100
        log_E = np.linspace(0, 6, n)
        log_sigma = np.ones(n) * 1.5
        mean_fn = lambda x: np.full_like(x, 1.5)

        max_l = 2.0
        interp_fn = compute_lengthscale_from_residuals(
            log_E, log_sigma, mean_fn, max_lengthscale=max_l,
        )
        assert interp_fn is not None

        query = np.linspace(0, 6, 20)
        signal = interp_fn(query)
        ell = _softplus(signal)

        # Should be uniform and close to max_lengthscale
        np.testing.assert_allclose(ell, max_l, atol=0.01)


# ===========================================================================
# Kernel integration tests
# ===========================================================================

class TestGibbsWithDataInterpolator:
    def test_gibbs_with_data_interpolator_psd(self):
        """GibbsKernel with data_lengthscale_interpolator produces valid PSD matrix."""
        log_E, log_sigma, mean_fn = _make_synthetic_data(n=50)
        interp_fn = compute_lengthscale_from_residuals(log_E, log_sigma, mean_fn)

        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=0.5,
            data_lengthscale_interpolator=interp_fn,
        )
        kernel = GibbsKernel(config)

        x = np.linspace(0, 6, 30)
        K = kernel.compute_matrix(x)

        # Check shape
        assert K.shape == (30, 30)

        # Check positive semi-definiteness: all eigenvalues >= -eps
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-8), f"Negative eigenvalue: {eigvals.min()}"

        # Check symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_data_interpolator_priority(self):
        """When both data and RIPL-3 set, data interpolator wins."""
        # Data interpolator returns constant signal=0.5
        data_interp = lambda x: np.full_like(x, 0.5)
        # RIPL-3 interpolator returns constant signal=5.0 (very different)
        ripl_interp = lambda x: np.full_like(x, 5.0)

        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            data_lengthscale_interpolator=data_interp,
            ripl_log_D_interpolator=ripl_interp,
        )
        kernel = GibbsKernel(config)

        x = np.array([3.0])
        ell = kernel._compute_lengthscales(x)

        # Should use data_interp (signal=0.5), not ripl_interp (signal=5.0)
        expected_ell = _softplus(np.array([0.5]))
        np.testing.assert_allclose(ell, expected_ell, rtol=1e-6)

    def test_build_kernel_data_only(self):
        """build_kernel with data interp, no RIPL-3 → returns GibbsKernel."""
        data_interp = lambda x: np.full_like(x, 0.0)

        config = KernelConfig(
            kernel_type='gibbs',
            data_lengthscale_interpolator=data_interp,
            ripl_log_D_interpolator=None,  # No RIPL-3
        )
        kernel = build_kernel(config)
        assert isinstance(kernel, GibbsKernel)


# ===========================================================================
# End-to-end integration tests
# ===========================================================================

class TestEndToEnd:
    def test_multi_experiment_pipeline(self):
        """ExperimentOutlierDetector with spline + gibbs: pipeline completes."""
        import pandas as pd
        from nucml_next.data.experiment_outlier import (
            ExperimentOutlierDetector,
            ExperimentOutlierConfig,
        )
        from nucml_next.data.experiment_gp import ExactGPExperimentConfig
        from nucml_next.data.likelihood import LikelihoodConfig

        rng = np.random.RandomState(42)

        # Create two experiments with shared trend but different noise levels
        n1, n2 = 60, 50
        log_E_1 = np.sort(rng.uniform(0, 6, n1))
        log_E_2 = np.sort(rng.uniform(0, 6, n2))
        trend = lambda x: 2.0 - 0.3 * x

        df = pd.DataFrame({
            'Z': [92] * (n1 + n2),
            'A': [235] * (n1 + n2),
            'MT': [18] * (n1 + n2),
            'Energy': np.concatenate([
                10 ** log_E_1, 10 ** log_E_2,
            ]),
            'CrossSection': np.concatenate([
                10 ** (trend(log_E_1) + rng.normal(0, 0.1, n1)),
                10 ** (trend(log_E_2) + rng.normal(0, 0.15, n2)),
            ]),
            'Uncertainty': np.concatenate([
                10 ** (trend(log_E_1)) * 0.05,
                10 ** (trend(log_E_2)) * 0.08,
            ]),
            'Entry': ['EXP001'] * n1 + ['EXP002'] * n2,
        })

        # Full stack: spline mean + Gibbs kernel (data-driven) + gaussian
        gp_config = ExactGPExperimentConfig(
            smooth_mean_config=SmoothMeanConfig(smooth_mean_type='spline'),
            kernel_config=KernelConfig(kernel_type='gibbs'),
            min_points_for_gp=15,
        )
        config = ExperimentOutlierConfig(
            gp_config=gp_config,
            gibbs_lengthscale_source='data',
            min_group_size=5,
        )

        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Pipeline should complete without error
        assert result is not None
        assert 'z_score' in result.columns
        assert 'experiment_outlier' in result.columns
        assert result['z_score'].notna().sum() > 0

    def test_lengthscale_varies(self):
        """Verify data-driven ℓ is NOT constant across energy."""
        log_E, log_sigma, mean_fn = _make_volatile_region_data(n=300)
        interp_fn = compute_lengthscale_from_residuals(log_E, log_sigma, mean_fn)
        assert interp_fn is not None

        query = np.linspace(0, 6, 50)
        signal = interp_fn(query)
        ell = _softplus(signal)

        # Standard deviation of lengthscale should be non-trivial
        # (if it were constant like RIPL-3, std would be ~0)
        assert np.std(ell) > 0.01, (
            f"Lengthscale std={np.std(ell):.4f} is too small; "
            f"should vary across energy"
        )

    def test_ripl_source_skips_data(self):
        """gibbs_lengthscale_source='ripl' skips data-driven computation."""
        from nucml_next.data.experiment_outlier import (
            ExperimentOutlierDetector,
            ExperimentOutlierConfig,
        )
        from nucml_next.data.experiment_gp import ExactGPExperimentConfig

        gp_config = ExactGPExperimentConfig(
            smooth_mean_config=SmoothMeanConfig(smooth_mean_type='spline'),
            kernel_config=KernelConfig(kernel_type='gibbs'),
        )
        config = ExperimentOutlierConfig(
            gp_config=gp_config,
            gibbs_lengthscale_source='ripl',
        )
        detector = ExperimentOutlierDetector(config)

        log_E = np.linspace(0, 6, 100)
        log_sigma = np.ones(100)
        mean_fn = lambda x: np.ones_like(x)

        result = detector._build_data_lengthscale(log_E, log_sigma, mean_fn)
        assert result is None, "Should return None when source='ripl'"
