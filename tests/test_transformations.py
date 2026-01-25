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
        """Test pipeline fitting and transformation."""
        pipeline = TransformationPipeline()

        # Fit and transform
        X_transformed, y_transformed = pipeline.fit_transform(
            self.X, self.y, self.energy,
            feature_columns=['Z', 'A', 'N', 'R_fm', 'kR', 'Mass_Excess_MeV']
        )

        # Check pipeline is fitted
        self.assertTrue(pipeline.is_fitted_)
        self.assertIsNotNone(pipeline.feature_mean_)
        self.assertIsNotNone(pipeline.feature_std_)

        # Check transformed data shape
        self.assertEqual(len(X_transformed), len(self.X))
        self.assertEqual(len(y_transformed), len(self.y))

        # Check standardization (mean ≈ 0, std ≈ 1)
        for col in ['Z', 'A', 'N']:
            col_mean = X_transformed[col].mean()
            col_std = X_transformed[col].std()
            self.assertAlmostEqual(col_mean, 0.0, places=10)
            self.assertAlmostEqual(col_std, 1.0, places=1)

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
        """Test pipeline save and load functionality."""
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

            # Check parameters match
            self.assertTrue(loaded_pipeline.is_fitted_)
            np.testing.assert_array_equal(
                loaded_pipeline.feature_mean_,
                pipeline.feature_mean_
            )
            np.testing.assert_array_equal(
                loaded_pipeline.feature_std_,
                pipeline.feature_std_
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
        """Test parameter retrieval."""
        pipeline = TransformationPipeline()
        pipeline.fit(self.X, feature_columns=['Z', 'A', 'N'])

        params = pipeline.get_params()

        self.assertIn('feature_mean', params)
        self.assertIn('feature_std', params)
        self.assertIn('feature_columns', params)
        self.assertIn('n_features', params)
        self.assertEqual(params['n_features'], 3)

    def test_promiscuity_factor_standardization(self):
        """Test that Promiscuity Factor is properly standardized."""
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

        pipeline = TransformationPipeline()
        pipeline.fit(X_with_p, feature_columns=['Z', 'A', 'P_Factor'])

        X_transformed = pipeline.transform(X_with_p)

        # Check P_Factor is standardized
        p_mean = X_transformed['P_Factor'].mean()
        p_std = X_transformed['P_Factor'].std()

        self.assertAlmostEqual(p_mean, 0.0, places=10)
        self.assertAlmostEqual(p_std, 1.0, places=1)


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
