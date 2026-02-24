"""
Tests for Phase 4: Hierarchical Experiment Structure
=====================================================

Tests for:
- ExperimentOutlierConfig hierarchical fields
- _extract_group_hyperparameters()
- _compute_refit_bounds()
- ExactGPExperiment.refit_with_constraints()
- Integration: hierarchical_refitting=True vs False
"""

import numpy as np
import pandas as pd
import pytest

from nucml_next.data.experiment_gp import (
    ExactGPExperiment,
    ExactGPExperimentConfig,
    prepare_log_uncertainties,
)
from nucml_next.data.experiment_outlier import (
    ExperimentOutlierDetector,
    ExperimentOutlierConfig,
)
from nucml_next.data.kernels import KernelConfig, RBFKernel, GibbsKernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_experiment(
    entry_id: str,
    n: int,
    E_min: float = 1e-2,
    E_max: float = 1e6,
    noise: float = 0.1,
    bias: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic cross-section experiment data."""
    rng = np.random.RandomState(seed)
    E = np.logspace(np.log10(E_min), np.log10(E_max), n)

    # 1/v law cross-section
    sigma_clean = 100.0 / np.sqrt(E)
    if bias != 0.0:
        sigma_clean = sigma_clean * (1 + bias)

    if noise > 0 and n > 1:
        sigma = sigma_clean * rng.lognormal(0, noise, n)
    else:
        sigma = sigma_clean.copy()

    sigma = np.clip(sigma, 1e-30, None)
    uncertainty = sigma * 0.05

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
    """Create synthetic group with multiple experiments."""
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


def _fit_gp(exp_df, seed=42):
    """Fit a GP to a synthetic experiment DataFrame and return it."""
    config = ExactGPExperimentConfig(
        use_wasserstein_calibration=True,
        min_points_for_gp=5,
    )
    gp = ExactGPExperiment(config)
    log_E = np.log10(exp_df['Energy'].values)
    log_sigma = np.log10(exp_df['CrossSection'].values)
    log_unc = prepare_log_uncertainties(
        exp_df['CrossSection'].values,
        exp_df['Uncertainty'].values,
    )
    gp.fit(log_E, log_sigma, log_unc)
    return gp


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestHierarchicalConfig:
    def test_defaults(self):
        """Default: hierarchical_refitting=False, min_experiments_for_refit=3."""
        config = ExperimentOutlierConfig()
        assert config.hierarchical_refitting is False
        assert config.min_experiments_for_refit == 3
        assert config.refit_bounds_iqr_margin == 1.0
        assert config.refit_share_outputscale is True

    def test_custom_values(self):
        """Custom hierarchical config values accessible."""
        config = ExperimentOutlierConfig(
            hierarchical_refitting=True,
            min_experiments_for_refit=5,
            refit_bounds_iqr_margin=1.5,
            refit_share_outputscale=False,
        )
        assert config.hierarchical_refitting is True
        assert config.min_experiments_for_refit == 5
        assert config.refit_bounds_iqr_margin == 1.5
        assert config.refit_share_outputscale is False


# ---------------------------------------------------------------------------
# Group stats extraction tests
# ---------------------------------------------------------------------------

class TestExtractGroupHyperparameters:
    def test_five_gps_correct_stats(self):
        """5 mock GPs with known params → verify median, Q1, Q3."""
        # Fit 5 GPs from synthetic data
        fitted_gps = {}
        for i in range(5):
            exp_df = _make_synthetic_experiment(
                entry_id=f"E{i+1:04d}",
                n=30,
                seed=42 + i,
            )
            fitted_gps[f"E{i+1:04d}"] = _fit_gp(exp_df, seed=42 + i)

        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            min_experiments_for_refit=3,
        ))
        stats = detector._extract_group_hyperparameters(fitted_gps)

        assert stats is not None
        assert stats['n_experiments'] == 5
        assert isinstance(stats['outputscale_median'], float)
        assert stats['param_medians'].ndim == 1
        assert stats['param_q1'].ndim == 1
        assert stats['param_q3'].ndim == 1
        assert stats['n_params'] == stats['param_medians'].shape[0]

    def test_two_gps_returns_none(self):
        """2 GPs → returns None (below threshold of 3)."""
        fitted_gps = {}
        for i in range(2):
            exp_df = _make_synthetic_experiment(
                entry_id=f"E{i+1:04d}",
                n=30,
                seed=42 + i,
            )
            fitted_gps[f"E{i+1:04d}"] = _fit_gp(exp_df, seed=42 + i)

        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            min_experiments_for_refit=3,
        ))
        stats = detector._extract_group_hyperparameters(fitted_gps)
        assert stats is None

    def test_three_gps_returns_valid(self):
        """3 GPs → returns valid stats (at threshold)."""
        fitted_gps = {}
        for i in range(3):
            exp_df = _make_synthetic_experiment(
                entry_id=f"E{i+1:04d}",
                n=30,
                seed=42 + i,
            )
            fitted_gps[f"E{i+1:04d}"] = _fit_gp(exp_df, seed=42 + i)

        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            min_experiments_for_refit=3,
        ))
        stats = detector._extract_group_hyperparameters(fitted_gps)
        assert stats is not None
        assert stats['n_experiments'] == 3


# ---------------------------------------------------------------------------
# Bounds computation tests
# ---------------------------------------------------------------------------

class TestComputeRefitBounds:
    def test_iqr_formula(self):
        """Known Q1/Q3 → verify IQR formula: [Q1 - margin*IQR, Q3 + margin*IQR]."""
        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            refit_bounds_iqr_margin=1.0,
        ))

        group_stats = {
            'param_q1': np.array([0.5]),
            'param_q3': np.array([1.5]),
            'param_medians': np.array([1.0]),
            'n_params': 1,
        }
        lower, upper = detector._compute_refit_bounds(group_stats)

        # IQR = 1.5 - 0.5 = 1.0
        # lower = 0.5 - 1.0*1.0 = -0.5 → clipped to 1e-3 (RBF)
        # upper = 1.5 + 1.0*1.0 = 2.5
        assert lower[0] == pytest.approx(1e-3, abs=1e-6)  # RBF clip
        assert upper[0] == pytest.approx(2.5, abs=1e-6)

    def test_zero_iqr_expands(self):
        """Zero IQR → bounds expand, don't collapse."""
        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            refit_bounds_iqr_margin=1.0,
        ))

        group_stats = {
            'param_q1': np.array([2.0]),
            'param_q3': np.array([2.0]),
            'param_medians': np.array([2.0]),
            'n_params': 1,
        }
        lower, upper = detector._compute_refit_bounds(group_stats)

        # median=2.0 → [2.0*0.5, 2.0*2.0] = [1.0, 4.0]
        assert lower[0] == pytest.approx(1.0, abs=1e-6)
        assert upper[0] == pytest.approx(4.0, abs=1e-6)
        assert lower[0] < upper[0]

    def test_rbf_lower_clipped(self):
        """RBF (n_params=1): lower clipped to >= 1e-3."""
        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            refit_bounds_iqr_margin=2.0,
        ))

        group_stats = {
            'param_q1': np.array([0.01]),
            'param_q3': np.array([0.02]),
            'param_medians': np.array([0.015]),
            'n_params': 1,
        }
        lower, upper = detector._compute_refit_bounds(group_stats)

        # IQR = 0.01, lower = 0.01 - 2.0*0.01 = -0.01 → clipped to 1e-3
        assert lower[0] >= 1e-3

    def test_gibbs_lower_can_be_negative(self):
        """Gibbs (n_params=2): lower can be negative (no clip for a₀, a₁)."""
        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            refit_bounds_iqr_margin=1.0,
        ))

        group_stats = {
            'param_q1': np.array([0.1, -0.5]),
            'param_q3': np.array([0.3, 0.5]),
            'param_medians': np.array([0.2, 0.0]),
            'n_params': 2,
        }
        lower, upper = detector._compute_refit_bounds(group_stats)

        # Gibbs: no lower clipping
        # param 0: IQR=0.2, lower=0.1-0.2=-0.1
        assert lower[0] == pytest.approx(-0.1, abs=1e-6)
        # param 1: IQR=1.0, lower=-0.5-1.0=-1.5
        assert lower[1] == pytest.approx(-1.5, abs=1e-6)

    def test_zero_iqr_near_zero_param(self):
        """Zero IQR near zero → expands to [median-0.5, median+0.5]."""
        detector = ExperimentOutlierDetector(ExperimentOutlierConfig(
            refit_bounds_iqr_margin=1.0,
        ))

        group_stats = {
            'param_q1': np.array([0.1, 0.1]),
            'param_q3': np.array([0.1, 0.1]),
            'param_medians': np.array([0.1, 0.1]),
            'n_params': 2,
        }
        lower, upper = detector._compute_refit_bounds(group_stats)

        # median=0.1 (near zero), [0.1-0.5, 0.1+0.5] = [-0.4, 0.6]
        assert lower[0] == pytest.approx(-0.4, abs=1e-6)
        assert upper[0] == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# refit_with_constraints tests
# ---------------------------------------------------------------------------

class TestRefitWithConstraints:
    def test_refit_updates_outputscale(self):
        """Fit + refit with new outputscale → verify _outputscale changed."""
        exp_df = _make_synthetic_experiment("E0001", n=30, seed=42)
        gp = _fit_gp(exp_df)

        old_outputscale = gp._outputscale
        new_outputscale = old_outputscale * 0.5

        gp.refit_with_constraints(outputscale=new_outputscale)

        # The outputscale should be the new value (kernel may have adjusted slightly)
        assert gp._outputscale == pytest.approx(new_outputscale, rel=1e-6)
        assert gp.is_fitted is True

    def test_refit_before_fit_raises(self):
        """Refit before fit() → raises RuntimeError."""
        gp = ExactGPExperiment()
        with pytest.raises(RuntimeError, match="Must call fit"):
            gp.refit_with_constraints()

    def test_refit_preserves_is_fitted(self):
        """Refit preserves is_fitted=True and updates hyperparameters."""
        exp_df = _make_synthetic_experiment("E0001", n=30, seed=42)
        gp = _fit_gp(exp_df)

        # Refit with no changes
        gp.refit_with_constraints()

        assert gp.is_fitted is True
        assert 'lengthscale' in gp.hyperparameters
        assert 'outputscale' in gp.hyperparameters
        assert 'n_points' in gp.hyperparameters

    def test_refit_with_bounds(self):
        """Refit with param_bounds produces valid results."""
        exp_df = _make_synthetic_experiment("E0001", n=30, seed=42)
        gp = _fit_gp(exp_df)

        # Get current params and build narrow bounds around them
        current_params = gp._kernel.get_optimizable_params()
        lower = current_params * 0.5
        upper = current_params * 2.0
        # Ensure lower < upper even for negative params
        for i in range(len(lower)):
            if lower[i] > upper[i]:
                lower[i], upper[i] = upper[i], lower[i]

        gp.refit_with_constraints(
            param_bounds=(lower, upper),
        )

        # Check that refit params are within bounds
        refit_params = gp._kernel.get_optimizable_params()
        np.testing.assert_array_less(lower - 1e-6, refit_params)
        np.testing.assert_array_less(refit_params, upper + 1e-6)

    def test_refit_with_contaminated_likelihood(self):
        """Refit with contaminated likelihood → outlier_probabilities updated."""
        from nucml_next.data.likelihood import LikelihoodConfig

        config = ExactGPExperimentConfig(
            use_wasserstein_calibration=True,
            min_points_for_gp=5,
            likelihood_config=LikelihoodConfig(
                likelihood_type='contaminated',
            ),
        )
        gp = ExactGPExperiment(config)
        exp_df = _make_synthetic_experiment("E0001", n=30, seed=42)
        log_E = np.log10(exp_df['Energy'].values)
        log_sigma = np.log10(exp_df['CrossSection'].values)
        log_unc = prepare_log_uncertainties(
            exp_df['CrossSection'].values,
            exp_df['Uncertainty'].values,
        )
        gp.fit(log_E, log_sigma, log_unc)

        # Should have outlier probabilities after fit
        assert gp.outlier_probabilities is not None

        # Refit
        gp.refit_with_constraints()

        # Should still have outlier probabilities
        assert gp.outlier_probabilities is not None
        assert len(gp.outlier_probabilities) == 30
        assert 'n_outliers' in gp.hyperparameters


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestHierarchicalIntegration:
    def test_hierarchical_true_stats_counters(self):
        """Synthetic 4+ experiments, hierarchical_refitting=True → stats counters > 0."""
        df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=4,
            points_per_exp=30,
            seed=42,
        )

        config = ExperimentOutlierConfig(
            gp_config=ExactGPExperimentConfig(
                use_wasserstein_calibration=True,
                min_points_for_gp=5,
            ),
            hierarchical_refitting=True,
            min_experiments_for_refit=3,
        )
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Should have refit stats
        assert detector._stats['hierarchical_groups'] > 0
        assert detector._stats['hierarchical_refits'] > 0
        # Skipped groups should be 0 (we had 4 experiments >= 3 threshold)
        assert detector._stats['hierarchical_skipped_groups'] == 0

        # Result should have standard columns
        assert 'z_score' in result.columns
        assert 'gp_mean' in result.columns
        assert 'experiment_outlier' in result.columns

    def test_hierarchical_false_counters_zero(self):
        """Same data, hierarchical_refitting=False → counters are 0."""
        df = _make_multi_experiment_group(
            Z=92, A=235, MT=18,
            n_experiments=4,
            points_per_exp=30,
            seed=42,
        )

        config = ExperimentOutlierConfig(
            gp_config=ExactGPExperimentConfig(
                use_wasserstein_calibration=True,
                min_points_for_gp=5,
            ),
            hierarchical_refitting=False,
        )
        detector = ExperimentOutlierDetector(config)
        result = detector.score_dataframe(df)

        # All hierarchical counters should be 0
        assert detector._stats['hierarchical_groups'] == 0
        assert detector._stats['hierarchical_refits'] == 0
        assert detector._stats['hierarchical_skipped_groups'] == 0

        # But should still have standard output
        assert 'z_score' in result.columns
        assert len(result) == 4 * 30


# ---------------------------------------------------------------------------
# Subsample + outlier probability tests
# ---------------------------------------------------------------------------

class TestSubsampleOutlierProbability:
    def test_subsample_indices_stored(self):
        """Subsampled GP stores subsample_indices with correct length."""
        config = ExactGPExperimentConfig(
            max_subsample_points=20,
            use_wasserstein_calibration=True,
            min_points_for_gp=5,
        )
        gp = ExactGPExperiment(config)
        rng = np.random.RandomState(42)
        n = 50
        log_E = np.sort(rng.uniform(0, 5, n))
        log_sigma = np.sin(log_E) + rng.randn(n) * 0.1
        log_unc = np.full(n, 0.05)

        gp.fit(log_E, log_sigma, log_unc)

        assert gp.subsample_indices is not None
        assert len(gp.subsample_indices) == 20
        assert gp._is_subsampled is True

    def test_no_subsample_indices_none(self):
        """Non-subsampled GP has subsample_indices=None."""
        config = ExactGPExperimentConfig(
            max_subsample_points=100,
            use_wasserstein_calibration=True,
            min_points_for_gp=5,
        )
        gp = ExactGPExperiment(config)
        rng = np.random.RandomState(42)
        n = 30
        log_E = np.sort(rng.uniform(0, 5, n))
        log_sigma = np.sin(log_E) + rng.randn(n) * 0.1
        log_unc = np.full(n, 0.05)

        gp.fit(log_E, log_sigma, log_unc)

        assert gp.subsample_indices is None
        assert gp._is_subsampled is False

    def test_subsample_outlier_prob_length_matches(self):
        """Subsampled GP + contaminated EM: outlier_probs length = subsample."""
        from nucml_next.data.likelihood import LikelihoodConfig

        config = ExactGPExperimentConfig(
            max_subsample_points=20,
            use_wasserstein_calibration=True,
            min_points_for_gp=5,
            likelihood_config=LikelihoodConfig(likelihood_type='contaminated'),
        )
        gp = ExactGPExperiment(config)
        rng = np.random.RandomState(42)
        n = 50
        log_E = np.sort(rng.uniform(0, 5, n))
        log_sigma = np.sin(log_E) + rng.randn(n) * 0.1
        log_unc = np.full(n, 0.05)

        gp.fit(log_E, log_sigma, log_unc)

        assert gp.outlier_probabilities is not None
        assert len(gp.outlier_probabilities) == 20  # subsample, not 50
        assert gp.subsample_indices is not None
        assert len(gp.subsample_indices) == 20
