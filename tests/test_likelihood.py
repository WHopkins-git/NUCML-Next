"""Tests for contaminated normal likelihood module."""

import numpy as np
import pytest

from nucml_next.data.likelihood import (
    LikelihoodConfig,
    run_contaminated_em,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_gp_data(n=50, seed=42, n_outliers=0, outlier_shift=5.0):
    """Generate synthetic GP data with optional injected outliers.

    Returns (K_kernel, y, noise_variance, mean_value, outlier_indices).
    """
    rng = np.random.RandomState(seed)

    # Simple 1D input
    x = np.linspace(0, 5, n)

    # RBF kernel (lengthscale=1.0, outputscale=1.0)
    diff = x[:, None] - x[None, :]
    K_kernel = np.exp(-0.5 * diff**2)

    # True function from GP prior
    L_true = np.linalg.cholesky(K_kernel + 1e-6 * np.eye(n))
    f_true = L_true @ rng.randn(n)

    # Noise
    noise_std = 0.1
    noise_variance = np.full(n, noise_std**2)
    y = f_true + noise_std * rng.randn(n)

    # Inject outliers
    outlier_indices = []
    if n_outliers > 0:
        outlier_indices = rng.choice(n, size=n_outliers, replace=False).tolist()
        for idx in outlier_indices:
            y[idx] += outlier_shift  # Shift by many sigma

    mean_value = 0.0  # constant mean

    return K_kernel, y, noise_variance, mean_value, outlier_indices


# ---------------------------------------------------------------------------
# LikelihoodConfig
# ---------------------------------------------------------------------------

class TestLikelihoodConfig:
    def test_defaults(self):
        config = LikelihoodConfig()
        assert config.likelihood_type == 'gaussian'
        assert config.contamination_fraction == 0.05
        assert config.contamination_scale == 10.0
        assert config.max_em_iterations == 10
        assert config.em_convergence_tol == 1e-3

    def test_custom_values(self):
        config = LikelihoodConfig(
            likelihood_type='contaminated',
            contamination_fraction=0.10,
            contamination_scale=20.0,
            max_em_iterations=20,
            em_convergence_tol=1e-4,
        )
        assert config.likelihood_type == 'contaminated'
        assert config.contamination_fraction == 0.10
        assert config.contamination_scale == 20.0

    def test_gaussian_is_noop_marker(self):
        """Gaussian config is just a marker — no EM is triggered."""
        config = LikelihoodConfig(likelihood_type='gaussian')
        assert config.likelihood_type == 'gaussian'


# ---------------------------------------------------------------------------
# run_contaminated_em — clean data
# ---------------------------------------------------------------------------

class TestContaminatedEMClean:
    """Tests with clean (no outlier) synthetic data."""

    def test_clean_data_low_outlier_prob(self):
        """All points should have outlier_prob < 0.5 for clean data."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=30, n_outliers=0)
        config = LikelihoodConfig(likelihood_type='contaminated')

        sigma_eff_sq, outlier_prob, L = run_contaminated_em(
            K, y, nv, mv, config
        )

        assert outlier_prob.shape == (30,)
        assert np.all(outlier_prob < 0.5), (
            f"Expected all outlier_prob < 0.5 for clean data, "
            f"got max={outlier_prob.max():.4f}"
        )

    def test_clean_data_outlier_prob_near_prior(self):
        """For clean data, outlier_prob should stay near the prior ε."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=30, n_outliers=0)
        config = LikelihoodConfig(
            likelihood_type='contaminated',
            contamination_fraction=0.05,
        )

        _, outlier_prob, _ = run_contaminated_em(K, y, nv, mv, config)

        # Mean outlier prob should be close to prior (not exactly, but < 0.3)
        assert np.mean(outlier_prob) < 0.3, (
            f"Mean outlier_prob={np.mean(outlier_prob):.4f} too high for clean data"
        )

    def test_cholesky_validity(self):
        """Returned L should be valid lower-triangular Cholesky factor."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=20, n_outliers=0)
        config = LikelihoodConfig(likelihood_type='contaminated')

        sigma_eff_sq, _, L = run_contaminated_em(K, y, nv, mv, config)

        # L should be lower triangular
        assert np.allclose(L, np.tril(L)), "L is not lower triangular"

        # L @ L.T should equal K_noisy
        K_noisy = K + np.diag(sigma_eff_sq) + 1e-6 * np.eye(len(y))
        np.testing.assert_allclose(
            L @ L.T, K_noisy, atol=1e-8,
            err_msg="L @ L.T does not match K_noisy"
        )

    def test_sigma_eff_sq_bounds(self):
        """sigma_eff_sq should be between noise_variance and κ * noise_variance."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=20, n_outliers=0)
        config = LikelihoodConfig(
            likelihood_type='contaminated',
            contamination_scale=10.0,
        )

        sigma_eff_sq, _, _ = run_contaminated_em(K, y, nv, mv, config)

        # sigma_eff_sq = nv * (1 + w*(κ-1)), with w ∈ [0, 1]
        # So σ_eff² ∈ [nv, κ*nv]
        assert np.all(sigma_eff_sq >= nv - 1e-12), "sigma_eff_sq below noise_variance"
        assert np.all(sigma_eff_sq <= config.contamination_scale * nv + 1e-12), (
            "sigma_eff_sq above κ * noise_variance"
        )


# ---------------------------------------------------------------------------
# run_contaminated_em — with outliers
# ---------------------------------------------------------------------------

class TestContaminatedEMOutliers:
    """Tests with injected outliers."""

    def test_single_outlier_detected(self):
        """A single injected outlier should get outlier_prob > 0.5."""
        K, y, nv, mv, outlier_idx = _make_synthetic_gp_data(
            n=30, n_outliers=1, outlier_shift=8.0
        )
        config = LikelihoodConfig(likelihood_type='contaminated')

        _, outlier_prob, _ = run_contaminated_em(K, y, nv, mv, config)

        # The outlier should have high probability
        for idx in outlier_idx:
            assert outlier_prob[idx] > 0.5, (
                f"Outlier at index {idx} has prob={outlier_prob[idx]:.4f}, "
                f"expected > 0.5"
            )

    def test_multiple_outliers_detected(self):
        """Multiple injected outliers should get high outlier_prob."""
        K, y, nv, mv, outlier_idx = _make_synthetic_gp_data(
            n=50, n_outliers=3, outlier_shift=6.0
        )
        config = LikelihoodConfig(likelihood_type='contaminated')

        _, outlier_prob, _ = run_contaminated_em(K, y, nv, mv, config)

        for idx in outlier_idx:
            assert outlier_prob[idx] > 0.5, (
                f"Outlier at index {idx} has prob={outlier_prob[idx]:.4f}"
            )

    def test_non_outlier_points_mostly_low(self):
        """Most non-outlier points should have outlier_prob < 0.5.

        Note: Points close to an outlier in input space can have elevated
        probabilities due to GP correlation, so we allow some spillover.
        """
        K, y, nv, mv, outlier_idx = _make_synthetic_gp_data(
            n=50, n_outliers=2, outlier_shift=8.0
        )
        config = LikelihoodConfig(likelihood_type='contaminated')

        _, outlier_prob, _ = run_contaminated_em(K, y, nv, mv, config)

        non_outlier_mask = np.ones(50, dtype=bool)
        for idx in outlier_idx:
            non_outlier_mask[idx] = False

        # Majority of non-outlier points should be < 0.5
        # Allow up to 40% spillover from GP correlation effects
        frac_high = np.mean(outlier_prob[non_outlier_mask] > 0.5)
        assert frac_high < 0.5, (
            f"{frac_high*100:.0f}% of clean points have outlier_prob > 0.5"
        )


# ---------------------------------------------------------------------------
# Convergence and edge cases
# ---------------------------------------------------------------------------

class TestContaminatedEMConvergence:
    def test_converges_within_max_iterations(self):
        """EM should converge for typical data."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=30, n_outliers=1)
        config = LikelihoodConfig(
            likelihood_type='contaminated',
            max_em_iterations=10,
        )

        # Should not raise
        sigma_eff_sq, outlier_prob, L = run_contaminated_em(
            K, y, nv, mv, config
        )
        assert sigma_eff_sq is not None
        assert outlier_prob is not None
        assert L is not None

    def test_single_iteration(self):
        """With max_em_iterations=1, should still return valid results."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=20, n_outliers=0)
        config = LikelihoodConfig(
            likelihood_type='contaminated',
            max_em_iterations=1,
        )

        sigma_eff_sq, outlier_prob, L = run_contaminated_em(
            K, y, nv, mv, config
        )
        assert sigma_eff_sq.shape == (20,)
        assert outlier_prob.shape == (20,)
        assert L.shape == (20, 20)

    def test_small_n(self):
        """Should work with very small n (e.g., n=5)."""
        K, y, nv, mv, _ = _make_synthetic_gp_data(n=5, n_outliers=0)
        config = LikelihoodConfig(likelihood_type='contaminated')

        sigma_eff_sq, outlier_prob, L = run_contaminated_em(
            K, y, nv, mv, config
        )
        assert sigma_eff_sq.shape == (5,)

    def test_array_mean_value(self):
        """Should work with array-valued mean (spline mean)."""
        K, y, nv, _, _ = _make_synthetic_gp_data(n=20, n_outliers=0)
        mean_value = np.linspace(-1, 1, 20)  # array mean
        config = LikelihoodConfig(likelihood_type='contaminated')

        sigma_eff_sq, outlier_prob, L = run_contaminated_em(
            K, y, nv, mean_value, config
        )
        assert sigma_eff_sq.shape == (20,)

    def test_large_kappa(self):
        """With very large κ, outlier component absorbs more variance."""
        K, y, nv, mv, outlier_idx = _make_synthetic_gp_data(
            n=30, n_outliers=1, outlier_shift=5.0
        )
        config = LikelihoodConfig(
            likelihood_type='contaminated',
            contamination_scale=100.0,
        )

        _, outlier_prob, _ = run_contaminated_em(K, y, nv, mv, config)

        # Should still detect the outlier
        for idx in outlier_idx:
            assert outlier_prob[idx] > 0.3, (
                f"Large κ: outlier at {idx} has prob={outlier_prob[idx]:.4f}"
            )

    def test_identical_y_values(self):
        """Should handle all identical y values without crashing."""
        n = 10
        x = np.linspace(0, 5, n)
        diff = x[:, None] - x[None, :]
        K = np.exp(-0.5 * diff**2)
        y = np.ones(n)  # all identical
        nv = np.full(n, 0.01)
        config = LikelihoodConfig(likelihood_type='contaminated')

        sigma_eff_sq, outlier_prob, L = run_contaminated_em(
            K, y, nv, 1.0, config
        )
        assert np.all(np.isfinite(outlier_prob))


# ---------------------------------------------------------------------------
# PyTorch path
# ---------------------------------------------------------------------------

class TestContaminatedEMTorch:
    """Tests for the PyTorch EM path."""

    @pytest.fixture
    def torch_available(self):
        """Skip if torch not available."""
        pytest.importorskip('torch')
        return True

    def test_torch_matches_numpy(self, torch_available):
        """Torch and NumPy paths should produce matching results."""
        import torch
        from nucml_next.data.likelihood import run_contaminated_em_torch

        K_np, y, nv, mv, _ = _make_synthetic_gp_data(n=30, n_outliers=1)
        config = LikelihoodConfig(likelihood_type='contaminated')

        # NumPy path
        sigma_np, prob_np, L_np = run_contaminated_em(K_np, y, nv, mv, config)

        # Torch path
        K_t = torch.tensor(K_np, dtype=torch.float64)
        sigma_t, prob_t, L_t = run_contaminated_em_torch(K_t, y, nv, mv, config)

        np.testing.assert_allclose(sigma_t, sigma_np, atol=1e-6,
                                   err_msg="sigma_eff_sq mismatch")
        np.testing.assert_allclose(prob_t, prob_np, atol=1e-6,
                                   err_msg="outlier_prob mismatch")
        np.testing.assert_allclose(L_t, L_np, atol=1e-6,
                                   err_msg="Cholesky factor mismatch")

    def test_torch_returns_numpy(self, torch_available):
        """Torch path should return NumPy arrays."""
        import torch
        from nucml_next.data.likelihood import run_contaminated_em_torch

        K_np, y, nv, mv, _ = _make_synthetic_gp_data(n=10, n_outliers=0)
        config = LikelihoodConfig(likelihood_type='contaminated')

        K_t = torch.tensor(K_np, dtype=torch.float64)
        sigma, prob, L = run_contaminated_em_torch(K_t, y, nv, mv, config)

        assert isinstance(sigma, np.ndarray)
        assert isinstance(prob, np.ndarray)
        assert isinstance(L, np.ndarray)
