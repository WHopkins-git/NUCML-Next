"""
Tests for SVGP Outlier Detection
=================================

Tests the SVGPOutlierDetector using synthetic nuclear cross-section data.
No real EXFOR database is required.
"""

import numpy as np
import pandas as pd
import pytest

from nucml_next.data.outlier_detection import SVGPOutlierDetector, SVGPConfig


def _make_synthetic_group(Z, A, MT, n, noise=0.1, seed=42):
    """Create synthetic cross-section data with known smooth curve + noise.

    Generates a 1/v-law cross-section (sigma = 100 / sqrt(E)) with
    lognormal noise, mimicking thermal neutron capture behavior.

    Args:
        Z: Atomic number
        A: Mass number
        MT: Reaction type code
        n: Number of data points
        noise: Lognormal noise scale (0 = no noise)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    E = np.logspace(-2, 7, n)
    # Simple smooth cross-section: sigma = 100 / sqrt(E)  (1/v law)
    sigma_clean = 100.0 / np.sqrt(E)
    if noise > 0 and n > 1:
        sigma = sigma_clean * rng.lognormal(0, noise, n)
    else:
        sigma = sigma_clean.copy()
    # Ensure positive
    sigma = np.clip(sigma, 1e-30, None)

    return pd.DataFrame({
        'Z': Z, 'A': A, 'MT': MT, 'N': A - Z,
        'Entry': 'SYNTH', 'Projectile': 'n',
        'Energy': E, 'CrossSection': sigma,
        'Uncertainty': sigma * 0.05,
        'Energy_Uncertainty': E * 0.01,
    })


class TestSVGPConfig:
    """Tests for SVGPConfig dataclass."""

    def test_defaults(self):
        """SVGPConfig has sensible defaults."""
        config = SVGPConfig()
        assert config.n_inducing == 50
        assert config.max_epochs == 300
        assert config.lr == 0.05
        assert config.convergence_tol == 1e-3
        assert config.patience == 10
        assert config.min_group_size_svgp == 10
        assert config.device == 'cpu'
        assert config.checkpoint_dir is None
        assert config.checkpoint_interval == 1000

    def test_custom_config(self):
        """SVGPConfig accepts custom values."""
        config = SVGPConfig(
            n_inducing=30,
            max_epochs=100,
            device='cuda',
            checkpoint_dir='/tmp/checkpoints',
        )
        assert config.n_inducing == 30
        assert config.max_epochs == 100
        assert config.device == 'cuda'
        assert config.checkpoint_dir == '/tmp/checkpoints'


class TestSVGPOutlierDetector:
    """Tests for SVGPOutlierDetector."""

    def test_basic_scoring(self):
        """Score a synthetic group with SVGP (>= min_group_size)."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=50, noise=0.1)
        detector = SVGPOutlierDetector(SVGPConfig(max_epochs=50, min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        # Check output columns exist
        assert 'z_score' in result.columns
        assert 'gp_mean' in result.columns
        assert 'gp_std' in result.columns
        assert 'log_E' in result.columns
        assert 'log_sigma' in result.columns

        # Check values are valid
        assert result['z_score'].notna().all()
        assert (result['gp_std'] > 0).all()
        assert (result['z_score'] >= 0).all()

        # Row count should be preserved
        assert len(result) == 50

    def test_mad_fallback_small_group(self):
        """Groups with < min_group_size use MAD fallback."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=5, noise=0.1)
        detector = SVGPOutlierDetector(SVGPConfig(min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        assert 'z_score' in result.columns
        assert result['z_score'].notna().all()
        assert (result['gp_std'] > 0).all()
        assert len(result) == 5

        # MAD group should have constant gp_mean (median) and gp_std
        # (since it uses median/MAD, not GP regression)
        assert result['gp_mean'].nunique() == 1  # All same (median)
        assert result['gp_std'].nunique() == 1   # All same (MAD * 1.4826)

    def test_single_point_group(self):
        """Single-point group gets z_score = 0."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=1)
        detector = SVGPOutlierDetector()
        result = detector.score_dataframe(df)

        assert len(result) == 1
        assert result['z_score'].iloc[0] == 0.0
        assert result['gp_std'].iloc[0] == 1.0

    def test_two_point_group(self):
        """Two-point group uses MAD fallback (not SVGP)."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=2, noise=0.1)
        detector = SVGPOutlierDetector(SVGPConfig(min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        assert len(result) == 2
        assert result['z_score'].notna().all()

    def test_known_outlier_detection(self):
        """Inject one extreme outlier into a clean group and verify detection."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=100, noise=0.02, seed=42)

        # Inject outlier: cross-section 1e6x too high (extreme, unambiguous)
        outlier_idx = df.index[50]
        df.loc[outlier_idx, 'CrossSection'] = df['CrossSection'].median() * 1e6

        detector = SVGPOutlierDetector(SVGPConfig(max_epochs=200, min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        # The injected outlier should have the highest z_score
        max_z_idx = result['z_score'].idxmax()
        assert max_z_idx == outlier_idx

        # With 1e6x deviation and low noise, this should be well above z=3
        assert result.loc[outlier_idx, 'z_score'] > 3.0

    def test_multiple_groups(self):
        """Process multiple (Z,A,MT) groups in one dataframe."""
        df1 = _make_synthetic_group(Z=1, A=2, MT=102, n=30, seed=42)
        df2 = _make_synthetic_group(Z=92, A=235, MT=18, n=40, seed=43)
        df = pd.concat([df1, df2], ignore_index=True)

        detector = SVGPOutlierDetector(SVGPConfig(max_epochs=50, min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        assert len(result) == 70
        assert result['z_score'].notna().all()

        # Each group should be scored independently
        group1 = result[result['Z'] == 1]
        group2 = result[result['Z'] == 92]
        assert len(group1) == 30
        assert len(group2) == 40

    def test_mixed_group_sizes(self):
        """DataFrame with large, small, and single-point groups."""
        df_large = _make_synthetic_group(Z=92, A=235, MT=18, n=50, seed=42)
        df_small = _make_synthetic_group(Z=17, A=35, MT=103, n=5, seed=43)
        df_single = _make_synthetic_group(Z=1, A=1, MT=2, n=1, seed=44)
        df = pd.concat([df_large, df_small, df_single], ignore_index=True)

        detector = SVGPOutlierDetector(SVGPConfig(max_epochs=50, min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        assert len(result) == 56
        assert result['z_score'].notna().all()

        # Single-point group should have z_score = 0
        single = result[(result['Z'] == 1) & (result['A'] == 1)]
        assert single['z_score'].iloc[0] == 0.0

    def test_checkpoint_save(self, tmp_path):
        """Checkpoint saves during processing."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=30)
        config = SVGPConfig(
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=1,
            max_epochs=20,
            min_group_size_svgp=10,
        )
        detector = SVGPOutlierDetector(config)
        result = detector.score_dataframe(df)

        # Checkpoint file should exist
        assert (tmp_path / 'svgp_checkpoint.pt').exists()
        assert len(result) == 30

    def test_preserves_original_columns(self):
        """Scoring preserves all original DataFrame columns."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=20)
        original_cols = set(df.columns)

        detector = SVGPOutlierDetector(SVGPConfig(max_epochs=20, min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        # All original columns should still be present
        for col in original_cols:
            assert col in result.columns, f"Original column '{col}' missing from result"

        # Plus new columns
        new_cols = {'log_E', 'log_sigma', 'gp_mean', 'gp_std', 'z_score'}
        for col in new_cols:
            assert col in result.columns, f"Expected new column '{col}' not in result"

    def test_missing_columns_raises(self):
        """Missing required columns raise ValueError."""
        df = pd.DataFrame({'X': [1, 2, 3]})
        detector = SVGPOutlierDetector()

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.score_dataframe(df)

    def test_z_scores_nonnegative(self):
        """Z-scores should always be non-negative (absolute deviation)."""
        df = _make_synthetic_group(Z=1, A=2, MT=102, n=50, noise=0.3)
        detector = SVGPOutlierDetector(SVGPConfig(max_epochs=50, min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        assert (result['z_score'] >= 0).all()


class TestMADFallback:
    """Tests specifically for the MAD fallback behavior."""

    def test_constant_values(self):
        """Group with identical cross-sections (MAD=0) gets z_score ~ 0."""
        df = pd.DataFrame({
            'Z': 1, 'A': 2, 'MT': 102, 'N': 1,
            'Entry': 'SYNTH', 'Projectile': 'n',
            'Energy': [1e0, 1e2, 1e4, 1e6],
            'CrossSection': [1.0, 1.0, 1.0, 1.0],
            'Uncertainty': [0.05] * 4,
            'Energy_Uncertainty': [0.01] * 4,
        })

        detector = SVGPOutlierDetector(SVGPConfig(min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        # With constant values, MAD=0, scale=1e-6, so z_scores should be ~0
        assert all(result['z_score'] < 1.0)

    def test_mad_with_outlier(self):
        """MAD fallback should detect an extreme outlier in small group."""
        df = pd.DataFrame({
            'Z': 1, 'A': 2, 'MT': 102, 'N': 1,
            'Entry': 'SYNTH', 'Projectile': 'n',
            'Energy': [1e0, 1e2, 1e4, 1e6, 1e8],
            'CrossSection': [10.0, 10.1, 9.9, 10.2, 10000.0],  # Last is outlier
            'Uncertainty': [0.5] * 5,
            'Energy_Uncertainty': [0.01] * 5,
        })

        detector = SVGPOutlierDetector(SVGPConfig(min_group_size_svgp=10))
        result = detector.score_dataframe(df)

        # The last point (10000) should have the highest z_score
        assert result['z_score'].idxmax() == 4
        assert result['z_score'].iloc[4] > 3.0


class TestDataSelectionIntegration:
    """Tests for DataSelection z_threshold integration."""

    def test_z_threshold_validation(self):
        """z_threshold must be positive if set."""
        from nucml_next.data.selection import DataSelection

        # Valid thresholds
        sel = DataSelection(z_threshold=3.0)
        assert sel.z_threshold == 3.0

        sel = DataSelection(z_threshold=None)
        assert sel.z_threshold is None

        # Invalid threshold
        with pytest.raises(ValueError, match="z_threshold must be positive"):
            DataSelection(z_threshold=-1.0)

        with pytest.raises(ValueError, match="z_threshold must be positive"):
            DataSelection(z_threshold=0.0)

    def test_include_outliers_default(self):
        """include_outliers defaults to True (keep all data)."""
        from nucml_next.data.selection import DataSelection

        sel = DataSelection()
        assert sel.include_outliers is True

    def test_repr_shows_outlier_settings(self):
        """DataSelection repr includes outlier settings when configured."""
        from nucml_next.data.selection import DataSelection

        sel = DataSelection(z_threshold=3.0, include_outliers=False)
        repr_str = repr(sel)
        assert 'z_threshold' in repr_str
        assert '3.0' in repr_str
