"""Tests for kernel abstraction layer."""

import numpy as np
import pytest

from nucml_next.data.kernels import (
    KernelConfig,
    RBFKernel,
    GibbsKernel,
    build_kernel,
    _softplus,
)


# ---------------------------------------------------------------------------
# Softplus helper
# ---------------------------------------------------------------------------

class TestSoftplus:
    def test_positive_output(self):
        x = np.array([-10, -1, 0, 1, 10])
        assert np.all(_softplus(x) > 0)

    def test_large_input(self):
        """For large x, softplus(x) ≈ x."""
        x = np.array([50.0, 100.0])
        np.testing.assert_allclose(_softplus(x), x, atol=1e-8)

    def test_small_input(self):
        """For very negative x, softplus(x) ≈ 0."""
        x = np.array([-50.0])
        assert _softplus(x)[0] < 1e-10


# ---------------------------------------------------------------------------
# KernelConfig
# ---------------------------------------------------------------------------

class TestKernelConfig:
    def test_defaults(self):
        config = KernelConfig()
        assert config.kernel_type == 'rbf'
        assert config.outputscale == 1.0
        assert config.lengthscale == 1.0
        assert config.ripl_log_D_interpolator is None

    def test_custom_values(self):
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=2.5,
            gibbs_correction_a0=0.1,
            gibbs_correction_a1=-0.05,
        )
        assert config.kernel_type == 'gibbs'
        assert config.outputscale == 2.5
        assert config.gibbs_correction_a0 == 0.1


# ---------------------------------------------------------------------------
# RBFKernel
# ---------------------------------------------------------------------------

class TestRBFKernel:
    def test_symmetric(self):
        """K(x1, x2) should be symmetric."""
        config = KernelConfig(outputscale=1.0, lengthscale=0.5)
        kernel = RBFKernel(config)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        K = kernel.compute_matrix(x)
        np.testing.assert_allclose(K, K.T, atol=1e-15)

    def test_diagonal_equals_outputscale(self):
        """K(x, x) diagonal should equal outputscale."""
        config = KernelConfig(outputscale=3.7, lengthscale=1.0)
        kernel = RBFKernel(config)
        x = np.array([0.0, 1.0, 2.0])
        K = kernel.compute_matrix(x)
        np.testing.assert_allclose(np.diag(K), 3.7, atol=1e-15)

    def test_prior_variance(self):
        config = KernelConfig(outputscale=2.5)
        kernel = RBFKernel(config)
        assert kernel.prior_variance() == 2.5

    def test_positive_definite(self):
        """Kernel matrix should be positive definite."""
        config = KernelConfig(outputscale=1.0, lengthscale=1.0)
        kernel = RBFKernel(config)
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        K = kernel.compute_matrix(x)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-10)

    def test_cross_covariance(self):
        """K(x1, x2) with different x1, x2."""
        config = KernelConfig(outputscale=1.0, lengthscale=1.0)
        kernel = RBFKernel(config)
        x1 = np.array([0.0, 1.0])
        x2 = np.array([0.5, 1.5, 2.5])
        K = kernel.compute_matrix(x1, x2)
        assert K.shape == (2, 3)

    def test_matches_hardcoded_formula(self):
        """Should exactly match the old hardcoded RBF formula."""
        config = KernelConfig(outputscale=2.0, lengthscale=0.7)
        kernel = RBFKernel(config)
        x1 = np.array([0.0, 1.0, 2.5])
        x2 = np.array([0.5, 1.5])

        K_new = kernel.compute_matrix(x1, x2)

        # Hardcoded formula from experiment_gp.py line 325-326
        diff = x1[:, None] - x2[None, :]
        K_old = 2.0 * np.exp(-0.5 * diff ** 2 / 0.7 ** 2)

        np.testing.assert_allclose(K_new, K_old, atol=1e-15)

    def test_n_optimizable_params(self):
        kernel = RBFKernel(KernelConfig())
        assert kernel.n_optimizable_params() == 1

    def test_get_set_optimizable_params(self):
        config = KernelConfig(lengthscale=1.5)
        kernel = RBFKernel(config)
        params = kernel.get_optimizable_params()
        np.testing.assert_array_equal(params, [1.5])

        kernel.set_optimizable_params(np.array([2.3]))
        assert kernel.config.lengthscale == 2.3

    def test_get_all_params(self):
        config = KernelConfig(outputscale=2.0, lengthscale=0.5)
        kernel = RBFKernel(config)
        params = kernel.get_all_params()
        assert params['outputscale'] == 2.0
        assert params['lengthscale'] == 0.5
        assert params['kernel_type'] == 'rbf'

    def test_torch_matches_numpy(self):
        """PyTorch and NumPy paths should produce identical results."""
        pytest.importorskip('torch')
        import torch

        config = KernelConfig(outputscale=1.5, lengthscale=0.8)
        kernel = RBFKernel(config)

        x = np.array([0.0, 1.0, 2.0, 3.0])
        K_np = kernel.compute_matrix(x)

        x_t = torch.tensor(x, dtype=torch.float64)
        K_torch = kernel.compute_matrix_torch(x_t).numpy()

        np.testing.assert_allclose(K_torch, K_np, atol=1e-12)

    def test_torch_cross_covariance(self):
        """PyTorch cross-covariance K(x1, x2)."""
        pytest.importorskip('torch')
        import torch

        config = KernelConfig(outputscale=1.0, lengthscale=1.0)
        kernel = RBFKernel(config)

        x1 = np.array([0.0, 1.0])
        x2 = np.array([0.5, 1.5, 2.5])

        K_np = kernel.compute_matrix(x1, x2)

        x1_t = torch.tensor(x1, dtype=torch.float64)
        x2_t = torch.tensor(x2, dtype=torch.float64)
        K_torch = kernel.compute_matrix_torch(x1_t, x2_t).numpy()

        np.testing.assert_allclose(K_torch, K_np, atol=1e-12)

    def test_lengthscale_effect(self):
        """Larger lengthscale → slower correlation decay."""
        x = np.array([0.0, 1.0])

        k_short = RBFKernel(KernelConfig(outputscale=1.0, lengthscale=0.1))
        k_long = RBFKernel(KernelConfig(outputscale=1.0, lengthscale=10.0))

        K_short = k_short.compute_matrix(x)
        K_long = k_long.compute_matrix(x)

        # Off-diagonal: shorter lengthscale → smaller correlation
        assert K_short[0, 1] < K_long[0, 1]


# ---------------------------------------------------------------------------
# GibbsKernel
# ---------------------------------------------------------------------------

class TestGibbsKernel:
    """Test the nonstationary Gibbs kernel."""

    @staticmethod
    def _constant_log_D(log_E):
        """Mock interpolator: constant D = 1 eV → log₁₀(D) = 0."""
        return np.zeros_like(log_E)

    @staticmethod
    def _linear_log_D(log_E):
        """Mock interpolator: D decreases with energy."""
        return -0.5 * log_E  # log D decreases with log E

    def test_symmetric(self):
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel = GibbsKernel(config)
        x = np.array([0.0, 1.0, 2.0, 3.0])
        K = kernel.compute_matrix(x)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_diagonal_equals_outputscale(self):
        """K(x, x) diagonal = outputscale (norm factor = 1 on diagonal)."""
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=2.5,
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel = GibbsKernel(config)
        x = np.array([0.0, 1.0, 2.0])
        K = kernel.compute_matrix(x)
        np.testing.assert_allclose(np.diag(K), 2.5, atol=1e-12)

    def test_positive_semi_definite(self):
        """Gibbs kernel matrix should be PSD."""
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            ripl_log_D_interpolator=self._linear_log_D,
        )
        kernel = GibbsKernel(config)
        x = np.linspace(0, 5, 20)
        K = kernel.compute_matrix(x)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-8), f"Min eigenvalue: {eigenvalues.min()}"

    def test_reduces_to_rbf_with_constant_lengthscale(self):
        """When ℓ(x) = constant, Gibbs should produce same matrix as RBF."""
        # With a₀=0, a₁=0, constant log_D → ℓ = softplus(0) = ln(2) ≈ 0.693
        config_gibbs = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            gibbs_correction_a0=0.0,
            gibbs_correction_a1=0.0,
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel_gibbs = GibbsKernel(config_gibbs)

        ell = np.log(2)  # softplus(0) = ln(2)
        config_rbf = KernelConfig(outputscale=1.0, lengthscale=ell)
        kernel_rbf = RBFKernel(config_rbf)

        x = np.array([0.0, 1.0, 2.0, 3.0])
        K_gibbs = kernel_gibbs.compute_matrix(x)
        K_rbf = kernel_rbf.compute_matrix(x)

        np.testing.assert_allclose(K_gibbs, K_rbf, atol=1e-10)

    def test_n_optimizable_params(self):
        config = KernelConfig(
            kernel_type='gibbs',
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel = GibbsKernel(config)
        assert kernel.n_optimizable_params() == 2

    def test_get_set_optimizable_params(self):
        config = KernelConfig(
            kernel_type='gibbs',
            gibbs_correction_a0=0.1,
            gibbs_correction_a1=-0.05,
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel = GibbsKernel(config)
        params = kernel.get_optimizable_params()
        np.testing.assert_allclose(params, [0.1, -0.05])

        kernel.set_optimizable_params(np.array([0.3, 0.2]))
        assert kernel.config.gibbs_correction_a0 == 0.3
        assert kernel.config.gibbs_correction_a1 == 0.2

    def test_nonstationary_effect(self):
        """With varying ℓ(x), off-diagonal structure should differ from RBF.

        In a region where ℓ is small (dense resonances), points should be
        less correlated at the same distance than in a region where ℓ is large.
        """
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            ripl_log_D_interpolator=self._linear_log_D,
        )
        kernel = GibbsKernel(config)

        # Two pairs of points at the same spacing (Δx = 0.5):
        # Pair A: low energy (large ℓ → high correlation)
        # Pair B: high energy (small ℓ → low correlation)
        x_low = np.array([0.0, 0.5])
        x_high = np.array([4.0, 4.5])

        K_low = kernel.compute_matrix(x_low)
        K_high = kernel.compute_matrix(x_high)

        # Normalised correlation
        corr_low = K_low[0, 1] / np.sqrt(K_low[0, 0] * K_low[1, 1])
        corr_high = K_high[0, 1] / np.sqrt(K_high[0, 0] * K_high[1, 1])

        # At low energy, ℓ is larger → higher correlation
        assert corr_low > corr_high, (
            f"Expected higher correlation at low energy: "
            f"corr_low={corr_low:.4f}, corr_high={corr_high:.4f}"
        )

    def test_correction_params_affect_lengthscale(self):
        """Changing a₀ shifts all lengthscales."""
        config_base = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            gibbs_correction_a0=0.0,
            ripl_log_D_interpolator=self._constant_log_D,
        )
        config_shifted = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            gibbs_correction_a0=2.0,  # larger a₀ → larger ℓ
            ripl_log_D_interpolator=self._constant_log_D,
        )

        k_base = GibbsKernel(config_base)
        k_shifted = GibbsKernel(config_shifted)

        x = np.array([0.0, 1.0])
        K_base = k_base.compute_matrix(x)
        K_shifted = k_shifted.compute_matrix(x)

        # Larger a₀ → larger ℓ → higher off-diagonal correlation
        assert K_shifted[0, 1] > K_base[0, 1]

    def test_cross_covariance(self):
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel = GibbsKernel(config)
        x1 = np.array([0.0, 1.0])
        x2 = np.array([0.5, 1.5, 2.5])
        K = kernel.compute_matrix(x1, x2)
        assert K.shape == (2, 3)

    def test_torch_matches_numpy(self):
        """PyTorch and NumPy paths should agree."""
        pytest.importorskip('torch')
        import torch

        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.5,
            gibbs_correction_a0=0.3,
            gibbs_correction_a1=-0.1,
            ripl_log_D_interpolator=self._linear_log_D,
        )
        kernel = GibbsKernel(config)

        x = np.array([0.0, 1.0, 2.0, 3.0])
        K_np = kernel.compute_matrix(x)

        x_t = torch.tensor(x, dtype=torch.float64)
        K_torch = kernel.compute_matrix_torch(x_t).numpy()

        np.testing.assert_allclose(K_torch, K_np, atol=1e-10)

    def test_lengthscale_bounds_enforced(self):
        """Lengthscale should be clipped to bounds."""
        config = KernelConfig(
            kernel_type='gibbs',
            outputscale=1.0,
            gibbs_correction_a0=-100,  # Force very small raw value
            gibbs_lengthscale_bounds=(0.05, 5.0),
            ripl_log_D_interpolator=self._constant_log_D,
        )
        kernel = GibbsKernel(config)
        x = np.array([1.0])
        ell = kernel._compute_lengthscales(x)
        assert ell[0] >= 0.05


# ---------------------------------------------------------------------------
# build_kernel factory
# ---------------------------------------------------------------------------

class TestBuildKernel:
    def test_none_returns_rbf(self):
        kernel = build_kernel(None)
        assert isinstance(kernel, RBFKernel)

    def test_rbf_config(self):
        config = KernelConfig(kernel_type='rbf', lengthscale=2.0)
        kernel = build_kernel(config)
        assert isinstance(kernel, RBFKernel)
        assert kernel.config.lengthscale == 2.0

    def test_gibbs_with_interpolator(self):
        config = KernelConfig(
            kernel_type='gibbs',
            ripl_log_D_interpolator=lambda x: np.zeros_like(x),
        )
        kernel = build_kernel(config)
        assert isinstance(kernel, GibbsKernel)

    def test_gibbs_without_interpolator_falls_back(self):
        """Gibbs without RIPL-3 data falls back to RBF with warning."""
        config = KernelConfig(kernel_type='gibbs')
        kernel = build_kernel(config)
        assert isinstance(kernel, RBFKernel)

    def test_unknown_type_raises(self):
        config = KernelConfig(kernel_type='matern52')
        with pytest.raises(ValueError, match="Unknown kernel_type"):
            build_kernel(config)
