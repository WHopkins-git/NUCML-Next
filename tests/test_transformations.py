"""
Unit Tests for Transformation Pipeline
=======================================

Tests for reversible transformations in nuclear cross-section ML:
- Log-scaling for cross-sections and energies
- StandardScaler for features
- Inverse transformations
- Save/load functionality
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.data.transformations import TransformationPipeline
from nucml_next.data.selection import TransformationConfig


class TestTransformationPipeline(unittest.TestCase):
    """Test suite for TransformationPipeline."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)

        # Create sample nuclear data
        n_samples = 100
        self.X = pd.DataFrame({
            'Z': np.random.randint(1, 100, n_samples),
            'A': np.random.randint(10, 250, n_samples),
            'N': np.random.randint(5, 150, n_samples),
            'R_fm': np.random.uniform(2.0, 8.0, n_samples),
            'kR': np.random.uniform(0.1, 10.0, n_samples),
            'Mass_Excess_MeV': np.random.uniform(-100, 100, n_samples),
        })

        self.y = pd.Series(
            np.random.uniform(0.01, 1000, n_samples),
            name='CrossSection'
        )

        self.energy = pd.Series(
            np.random.uniform(1e3, 1e7, n_samples),
            name='Energy'
        )

    def test_fit_transform(self):
        """Test pipeline fitting and transformation (default: minmax scaler)."""
        pipeline = TransformationPipeline()

        # Fit and transform
        X_transformed, y_transformed = pipeline.fit_transform(
            self.X, self.y, self.energy,
            feature_columns=['Z', 'A', 'N', 'R_fm', 'kR', 'Mass_Excess_MeV']
        )

        # Check pipeline is fitted
        self.assertTrue(pipeline.is_fitted_)
        # Default is now minmax, so check min/max params
        self.assertIsNotNone(pipeline.feature_min_)
        self.assertIsNotNone(pipeline.feature_max_)

        # Check transformed data shape
        self.assertEqual(len(X_transformed), len(self.X))
        self.assertEqual(len(y_transformed), len(self.y))

        # Check minmax scaling (values in [0, 1])
        for col in ['Z', 'A', 'N']:
            self.assertAlmostEqual(X_transformed[col].min(), 0.0, places=10)
            self.assertAlmostEqual(X_transformed[col].max(), 1.0, places=10)

    def test_log_transform_target(self):
        """Test log-transformation of cross-sections."""
        pipeline = TransformationPipeline()
        y_log = pipeline.transform_target(self.y)

        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(y_log)))

        # Check inverse reconstruction
        y_reconstructed = pipeline.inverse_transform_target(y_log)

        # Should match original values (within numerical precision)
        np.testing.assert_allclose(
            y_reconstructed.values,
            self.y.values,
            rtol=1e-6,
            atol=1e-10
        )

    def test_log_transform_energy(self):
        """Test log-transformation of energies."""
        pipeline = TransformationPipeline()

        # Add Energy column to X for this test
        X_with_energy = self.X.copy()
        X_with_energy['Energy'] = self.energy

        pipeline.fit(X_with_energy, feature_columns=['Z', 'A', 'N'])

        X_transformed = pipeline.transform(X_with_energy, self.energy)

        # Check energy is log-transformed
        energy_log = X_transformed['Energy']
        self.assertTrue(np.all(np.isfinite(energy_log)))

        # Verify log relationship
        np.testing.assert_allclose(
            10 ** energy_log.values,
            self.energy.values,
            rtol=1e-6
        )

    def test_inverse_transform(self):
        """Test inverse transformation recovers original data."""
        pipeline = TransformationPipeline()

        # Fit and transform
        X_transformed, y_transformed = pipeline.fit_transform(
            self.X, self.y, self.energy,
            feature_columns=['Z', 'A', 'N', 'R_fm']
        )

        # Inverse transform
        X_reconstructed = pipeline.inverse_transform(X_transformed, self.energy)
        y_reconstructed = pipeline.inverse_transform_target(y_transformed)

        # Check feature reconstruction
        for col in ['Z', 'A', 'N', 'R_fm']:
            np.testing.assert_allclose(
                X_reconstructed[col].values,
                self.X[col].values,
                rtol=1e-5,
                err_msg=f"Reconstruction failed for {col}"
            )

        # Check target reconstruction
        np.testing.assert_allclose(
            y_reconstructed.values,
            self.y.values,
            rtol=1e-5
        )

    def test_transform_without_fit_raises_error(self):
        """Test that transform() raises error if not fitted."""
        pipeline = TransformationPipeline()

        with self.assertRaises(RuntimeError):
            pipeline.transform(self.X)

        with self.assertRaises(RuntimeError):
            pipeline.inverse_transform(self.X)

        with self.assertRaises(RuntimeError):
            pipeline.get_params()

    def test_save_load(self):
        """Test pipeline save and load functionality (default: minmax)."""
        pipeline = TransformationPipeline()
        pipeline.fit(self.X, self.y, self.energy,
                    feature_columns=['Z', 'A', 'N', 'R_fm'])

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            pipeline.save(temp_path)

            # Load pipeline
            loaded_pipeline = TransformationPipeline.load(temp_path)

            # Check parameters match (default is now minmax)
            self.assertTrue(loaded_pipeline.is_fitted_)
            np.testing.assert_array_equal(
                loaded_pipeline.feature_min_,
                pipeline.feature_min_
            )
            np.testing.assert_array_equal(
                loaded_pipeline.feature_max_,
                pipeline.feature_max_
            )
            self.assertEqual(
                loaded_pipeline.feature_columns_,
                pipeline.feature_columns_
            )

            # Check loaded pipeline produces same transforms
            X_orig = pipeline.transform(self.X, self.energy)
            X_loaded = loaded_pipeline.transform(self.X, self.energy)

            for col in pipeline.feature_columns_:
                np.testing.assert_allclose(
                    X_orig[col].values,
                    X_loaded[col].values,
                    rtol=1e-10
                )

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    def test_numerical_stability(self):
        """Test pipeline handles edge cases (zero, negative, very small values)."""
        # Cross-sections near zero
        y_small = pd.Series([1e-10, 1e-8, 1e-6, 1e-4, 0.01, 1.0, 100.0])

        pipeline = TransformationPipeline()
        y_log = pipeline.transform_target(y_small)

        # Should not have inf or nan
        self.assertTrue(np.all(np.isfinite(y_log)))

        # Inverse should recover values
        y_recovered = pipeline.inverse_transform_target(y_log)
        np.testing.assert_allclose(y_recovered.values, y_small.values, rtol=1e-6)

    def test_zero_std_features(self):
        """Test pipeline handles constant features (std=0)."""
        X_constant = self.X.copy()
        X_constant['Constant'] = 42.0  # Constant feature

        pipeline = TransformationPipeline()
        pipeline.fit(X_constant, feature_columns=['Z', 'A', 'Constant'])

        # Should not raise error
        X_transformed = pipeline.transform(X_constant)

        # Constant feature should remain constant (not NaN)
        self.assertTrue(np.all(np.isfinite(X_transformed['Constant'])))

    def test_get_params(self):
        """Test parameter retrieval (default: minmax scaler)."""
        pipeline = TransformationPipeline()
        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])

        params = pipeline.get_params()

        # Default scaler is now minmax, so check for min/max params
        self.assertIn('feature_min', params)
        self.assertIn('feature_max', params)
        self.assertIn('feature_columns', params)
        self.assertIn('n_features', params)
        self.assertEqual(params['n_features'], 3)

    def test_promiscuity_factor_scaling(self):
        """Test that Promiscuity Factor is properly scaled (default: minmax)."""
        # Create sample data with P_Factor
        X_with_p = self.X.copy()
        X_with_p['Valence_N'] = np.random.randint(0, 20, len(self.X))
        X_with_p['Valence_P'] = np.random.randint(0, 20, len(self.X))

        # Compute Promiscuity Factor
        X_with_p['P_Factor'] = X_with_p.apply(
            lambda row: (row['Valence_N'] * row['Valence_P']) /
                       (row['Valence_N'] + row['Valence_P'])
                       if (row['Valence_N'] + row['Valence_P']) > 0 else 0.0,
            axis=1
        )

        # Default scaler is now minmax
        pipeline = TransformationPipeline()
        pipeline.fit(X_with_p, feature_columns=['Z', 'A', 'P_Factor'])

        X_transformed = pipeline.transform(X_with_p)

        # Check P_Factor is minmax-scaled to [0, 1]
        self.assertAlmostEqual(X_transformed['P_Factor'].min(), 0.0, places=10)
        self.assertAlmostEqual(X_transformed['P_Factor'].max(), 1.0, places=10)

    def test_promiscuity_factor_standardization(self):
        """Test that Promiscuity Factor is properly standardized with standard scaler."""
        # Create sample data with P_Factor
        X_with_p = self.X.copy()
        X_with_p['Valence_N'] = np.random.randint(0, 20, len(self.X))
        X_with_p['Valence_P'] = np.random.randint(0, 20, len(self.X))

        # Compute Promiscuity Factor
        X_with_p['P_Factor'] = X_with_p.apply(
            lambda row: (row['Valence_N'] * row['Valence_P']) /
                       (row['Valence_N'] + row['Valence_P'])
                       if (row['Valence_N'] + row['Valence_P']) > 0 else 0.0,
            axis=1
        )

        config = TransformationConfig(scaler_type='standard')
        pipeline = TransformationPipeline(config=config)
        pipeline.fit(X_with_p, feature_columns=['Z', 'A', 'P_Factor'])

        X_transformed = pipeline.transform(X_with_p)

        # Check P_Factor is standardized (mean ≈ 0, std ≈ 1)
        p_mean = X_transformed['P_Factor'].mean()
        p_std = X_transformed['P_Factor'].std()

        self.assertAlmostEqual(p_mean, 0.0, places=10)
        self.assertAlmostEqual(p_std, 1.0, places=1)


class TestToggleableTransformations(unittest.TestCase):
    """Test suite for toggleable transformations (scaler types, log toggles, etc.)."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)

        # Create sample nuclear data
        n_samples = 100
        self.X = pd.DataFrame({
            'Z': np.random.randint(1, 100, n_samples),
            'A': np.random.randint(10, 250, n_samples),
            'N': np.random.randint(5, 150, n_samples),
            'R_fm': np.random.uniform(2.0, 8.0, n_samples),
        })

        self.y = pd.Series(
            np.random.uniform(0.01, 1000, n_samples),
            name='CrossSection'
        )

        self.energy = pd.Series(
            np.random.uniform(1e3, 1e7, n_samples),
            name='Energy'
        )

    def test_scaler_standard(self):
        """Test standard (Z-score) scaler."""
        config = TransformationConfig(scaler_type='standard')
        pipeline = TransformationPipeline(config=config)

        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])
        X_transformed = pipeline.transform(self.X)

        # Check standardization (mean ≈ 0, std ≈ 1)
        for col in ['Z', 'A', 'N']:
            self.assertAlmostEqual(X_transformed[col].mean(), 0.0, places=10)
            self.assertAlmostEqual(X_transformed[col].std(), 1.0, places=1)

        # Check inverse reconstruction
        X_reconstructed = pipeline.inverse_transform(X_transformed)
        for col in ['Z', 'A', 'N']:
            np.testing.assert_allclose(
                X_reconstructed[col].values,
                self.X[col].values,
                rtol=1e-10
            )

    def test_scaler_minmax(self):
        """Test min-max scaler (scales to [0, 1])."""
        config = TransformationConfig(scaler_type='minmax')
        pipeline = TransformationPipeline(config=config)

        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])
        X_transformed = pipeline.transform(self.X)

        # Check min-max scaling (values in [0, 1])
        for col in ['Z', 'A', 'N']:
            self.assertGreaterEqual(X_transformed[col].min(), 0.0)
            self.assertLessEqual(X_transformed[col].max(), 1.0)
            # Check that it actually spans the range
            self.assertAlmostEqual(X_transformed[col].min(), 0.0, places=10)
            self.assertAlmostEqual(X_transformed[col].max(), 1.0, places=10)

        # Check inverse reconstruction
        X_reconstructed = pipeline.inverse_transform(X_transformed)
        for col in ['Z', 'A', 'N']:
            np.testing.assert_allclose(
                X_reconstructed[col].values,
                self.X[col].values,
                rtol=1e-10
            )

    def test_scaler_robust(self):
        """Test robust scaler (uses median and IQR)."""
        config = TransformationConfig(scaler_type='robust')
        pipeline = TransformationPipeline(config=config)

        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])
        X_transformed = pipeline.transform(self.X)

        # Robust scaler doesn't guarantee specific mean/std,
        # but should be centered around 0
        for col in ['Z', 'A', 'N']:
            # Median should be close to 0 after transformation
            self.assertAlmostEqual(X_transformed[col].median(), 0.0, places=1)

        # Check inverse reconstruction
        X_reconstructed = pipeline.inverse_transform(X_transformed)
        for col in ['Z', 'A', 'N']:
            np.testing.assert_allclose(
                X_reconstructed[col].values,
                self.X[col].values,
                rtol=1e-10
            )

    def test_scaler_none(self):
        """Test no scaling (features unchanged)."""
        config = TransformationConfig(scaler_type='none')
        pipeline = TransformationPipeline(config=config)

        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])
        X_transformed = pipeline.transform(self.X)

        # Features should be unchanged
        for col in ['Z', 'A', 'N']:
            np.testing.assert_allclose(
                X_transformed[col].values,
                self.X[col].values,
                rtol=1e-15
            )

    def test_log_target_toggle(self):
        """Test toggling log transformation for target."""
        # With log transform (default)
        config_log = TransformationConfig(log_target=True)
        pipeline_log = TransformationPipeline(config=config_log)
        y_log = pipeline_log.transform_target(self.y)

        # Values should be different from original
        self.assertFalse(np.allclose(y_log.values, self.y.values))

        # Without log transform
        config_no_log = TransformationConfig(log_target=False)
        pipeline_no_log = TransformationPipeline(config=config_no_log)
        y_no_log = pipeline_no_log.transform_target(self.y)

        # Values should be identical to original
        np.testing.assert_allclose(y_no_log.values, self.y.values, rtol=1e-15)

        # Inverse should also work
        y_reconstructed = pipeline_no_log.inverse_transform_target(y_no_log)
        np.testing.assert_allclose(y_reconstructed.values, self.y.values, rtol=1e-15)

    def test_log_energy_toggle(self):
        """Test toggling log transformation for energy."""
        # Test with log transform enabled
        config_log = TransformationConfig(log_energy=True)
        pipeline_log = TransformationPipeline(config=config_log)
        pipeline_log.fit(self.X, feature_columns=['Z', 'A'])

        X_log = pipeline_log.transform(self.X, energy=self.energy)

        # Energy should be log-transformed and added as Energy_log (since Energy not in X)
        self.assertIn('Energy_log', X_log.columns)
        # Log-transformed values should be much smaller (log10 of 1e3 to 1e7 is 3 to 7)
        self.assertLess(X_log['Energy_log'].mean(), 10)
        self.assertGreater(X_log['Energy_log'].mean(), 0)

        # Without log transform
        config_no_log = TransformationConfig(log_energy=False)
        pipeline_no_log = TransformationPipeline(config=config_no_log)
        pipeline_no_log.fit(self.X, feature_columns=['Z', 'A'])

        X_no_log = pipeline_no_log.transform(self.X, energy=self.energy)

        # Energy should not be added when log_energy=False
        self.assertNotIn('Energy_log', X_no_log.columns)
        self.assertNotIn('Energy', X_no_log.columns)

    def test_custom_epsilon(self):
        """Test custom epsilon value for log transform."""
        # Use very small cross-sections where epsilon matters
        y_small = pd.Series([1e-12, 1e-10, 1e-8, 1e-6], name='CrossSection')

        epsilon_1 = 1e-10
        epsilon_2 = 1e-8

        config_1 = TransformationConfig(target_epsilon=epsilon_1)
        config_2 = TransformationConfig(target_epsilon=epsilon_2)

        pipeline_1 = TransformationPipeline(config=config_1)
        pipeline_2 = TransformationPipeline(config=config_2)

        y_log_1 = pipeline_1.transform_target(y_small)
        y_log_2 = pipeline_2.transform_target(y_small)

        # Different epsilon should give different results for small values
        self.assertFalse(np.allclose(y_log_1.values, y_log_2.values))

        # But both should be invertible
        y_reconstructed_1 = pipeline_1.inverse_transform_target(y_log_1)
        y_reconstructed_2 = pipeline_2.inverse_transform_target(y_log_2)

        np.testing.assert_allclose(y_reconstructed_1.values, y_small.values, rtol=1e-5)
        np.testing.assert_allclose(y_reconstructed_2.values, y_small.values, rtol=1e-5)

    def test_log_base_10(self):
        """Test log base 10 (default)."""
        config = TransformationConfig(log_base=10)
        pipeline = TransformationPipeline(config=config)

        y_log = pipeline.transform_target(self.y)
        y_reconstructed = pipeline.inverse_transform_target(y_log)

        np.testing.assert_allclose(y_reconstructed.values, self.y.values, rtol=1e-6)

    def test_log_base_e(self):
        """Test natural logarithm (base e)."""
        config = TransformationConfig(log_base='e')
        pipeline = TransformationPipeline(config=config)

        y_log = pipeline.transform_target(self.y)
        y_reconstructed = pipeline.inverse_transform_target(y_log)

        np.testing.assert_allclose(y_reconstructed.values, self.y.values, rtol=1e-6)

    def test_log_base_2(self):
        """Test log base 2."""
        config = TransformationConfig(log_base=2)
        pipeline = TransformationPipeline(config=config)

        y_log = pipeline.transform_target(self.y)
        y_reconstructed = pipeline.inverse_transform_target(y_log)

        np.testing.assert_allclose(y_reconstructed.values, self.y.values, rtol=1e-6)

    def test_save_load_with_config(self):
        """Test save/load preserves transformation config."""
        config = TransformationConfig(
            scaler_type='minmax',
            log_target=True,
            log_energy=False,
            target_epsilon=1e-8,
            log_base='e'
        )
        pipeline = TransformationPipeline(config=config)
        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            pipeline.save(temp_path)

            # Load and check config is preserved
            pipeline_loaded = TransformationPipeline.load(temp_path)

            self.assertEqual(pipeline_loaded.config.scaler_type, 'minmax')
            self.assertEqual(pipeline_loaded.config.log_target, True)
            self.assertEqual(pipeline_loaded.config.log_energy, False)
            self.assertEqual(pipeline_loaded.config.target_epsilon, 1e-8)
            self.assertEqual(pipeline_loaded.config.log_base, 'e')

            # Check transformation produces same results
            X_orig = pipeline.transform(self.X)
            X_loaded = pipeline_loaded.transform(self.X)

            for col in ['Z', 'A', 'N']:
                np.testing.assert_allclose(
                    X_orig[col].values,
                    X_loaded[col].values,
                    rtol=1e-10
                )

        finally:
            Path(temp_path).unlink(missing_ok=True)


    def test_scale_features_all(self):
        """Test scale_features='all' scales every numeric column."""
        config = TransformationConfig(scaler_type='minmax', scale_features='all')
        pipeline = TransformationPipeline(config=config)

        pipeline.fit(self.X)  # No feature_columns arg -> uses config
        X_transformed = pipeline.transform(self.X)

        # All numeric columns should be scaled to [0, 1]
        for col in self.X.columns:
            self.assertAlmostEqual(X_transformed[col].min(), 0.0, places=10,
                                   msg=f"{col} min != 0")
            self.assertAlmostEqual(X_transformed[col].max(), 1.0, places=10,
                                   msg=f"{col} max != 1")

    def test_scale_features_explicit_list(self):
        """Test scale_features with an explicit column list."""
        config = TransformationConfig(
            scaler_type='minmax',
            scale_features=['Z', 'A']  # Only scale Z and A
        )
        pipeline = TransformationPipeline(config=config)

        pipeline.fit(self.X)
        X_transformed = pipeline.transform(self.X)

        # Z and A should be scaled
        for col in ['Z', 'A']:
            self.assertAlmostEqual(X_transformed[col].min(), 0.0, places=10)
            self.assertAlmostEqual(X_transformed[col].max(), 1.0, places=10)

        # N and R_fm should NOT be scaled (remain original values)
        for col in ['N', 'R_fm']:
            np.testing.assert_allclose(
                X_transformed[col].values,
                self.X[col].values,
                rtol=1e-15,
                err_msg=f"{col} should not be scaled"
            )

    def test_log_energy_before_scaling(self):
        """Test that log-energy is applied BEFORE scaling (new order)."""
        # Create X with Energy column
        X_with_energy = self.X.copy()
        X_with_energy['Energy'] = self.energy

        config = TransformationConfig(
            scaler_type='minmax',
            scale_features='all',
            log_energy=True,
        )
        pipeline = TransformationPipeline(config=config)
        pipeline.fit(X_with_energy, energy=self.energy)
        X_transformed = pipeline.transform(X_with_energy, self.energy)

        # Energy should be log-transformed AND THEN scaled to [0, 1]
        # Not just log-transformed
        self.assertAlmostEqual(X_transformed['Energy'].min(), 0.0, places=5)
        self.assertAlmostEqual(X_transformed['Energy'].max(), 1.0, places=5)

        # Verify inverse recovers original
        X_reconstructed = pipeline.inverse_transform(X_transformed, self.energy)
        for col in ['Z', 'A', 'N', 'R_fm']:
            np.testing.assert_allclose(
                X_reconstructed[col].values,
                X_with_energy[col].values,
                rtol=1e-5,
                err_msg=f"Reconstruction failed for {col}"
            )

    def test_scale_features_none_same_as_all(self):
        """Test that scale_features=None behaves same as 'all'."""
        config_all = TransformationConfig(scaler_type='minmax', scale_features='all')
        config_none = TransformationConfig(scaler_type='minmax', scale_features=None)

        pipeline_all = TransformationPipeline(config=config_all)
        pipeline_none = TransformationPipeline(config=config_none)

        pipeline_all.fit(self.X)
        pipeline_none.fit(self.X)

        X_all = pipeline_all.transform(self.X)
        X_none = pipeline_none.transform(self.X)

        for col in self.X.columns:
            np.testing.assert_allclose(
                X_all[col].values,
                X_none[col].values,
                rtol=1e-15,
                err_msg=f"Mismatch for {col}"
            )


class TestPromiscuityFactor(unittest.TestCase):
    """Test Promiscuity Factor calculation."""

    def test_promiscuity_formula(self):
        """Test P = N_p * N_n / (N_p + N_n) formula."""
        # Test cases: (N_p, N_n, expected_P)
        test_cases = [
            (10, 10, 5.0),   # Equal valence: P = 10*10/(10+10) = 5
            (8, 2, 1.6),     # Unequal: P = 8*2/(8+2) = 1.6
            (0, 10, 0.0),    # Magic proton shell: P = 0
            (10, 0, 0.0),    # Magic neutron shell: P = 0
            (0, 0, 0.0),     # Doubly magic: P = 0
        ]

        for valence_n, valence_p, expected_p in test_cases:
            if valence_n + valence_p == 0:
                p = 0.0
            else:
                p = (valence_n * valence_p) / (valence_n + valence_p)

            self.assertAlmostEqual(p, expected_p, places=5,
                                 msg=f"Failed for N_p={valence_p}, N_n={valence_n}")


if __name__ == '__main__':
    unittest.main()
