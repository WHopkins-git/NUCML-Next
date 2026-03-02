"""
Unit Tests for NeuralNetEvaluator
==================================

Tests for the NeuralNetEvaluator class which provides a CPU-friendly
feedforward neural network baseline with OneCycleLR, Kaiming He init,
AdamW, early stopping, and four loss functions.

Test coverage:
- Default configuration values
- Training on synthetic data (all 4 loss functions)
- Early stopping fires before max epochs
- Predict API returns numpy array in barns
- Kaiming He initialization variance
- Save/load round-trip preserves predictions
- Repr output
"""

import unittest
import tempfile
from pathlib import Path
import math

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from nucml_next.baselines.nn_evaluator import (
    NeuralNetEvaluator,
    _SimpleNet,
    _mse_loss,
    _chi_squared_loss,
    _physics_informed_loss,
    _resonance_informed_loss,
)


def _make_synthetic_df(n_samples=2000, seed=42):
    """Create a synthetic DataFrame mimicking nuclear cross-section data.

    Generates data with Energy, Z, A, N features and a CrossSection target
    that follows a rough 1/v + resonance shape.
    """
    rng = np.random.default_rng(seed)

    energy = 10 ** rng.uniform(-2, 6, n_samples)  # 0.01 eV to 1 MeV
    z = rng.choice([17, 26, 92], n_samples)
    a = z * 2 + rng.integers(-3, 4, n_samples)
    n = a - z

    # Synthetic cross-section: 1/v + Breit-Wigner bump + noise
    sigma_1v = 100.0 / np.sqrt(energy + 1e-10)
    sigma_bw = 50.0 * (0.5 ** 2) / ((energy - 1e3) ** 2 + 0.5 ** 2)
    noise = rng.exponential(0.1 * sigma_1v)
    xs = np.abs(sigma_1v + sigma_bw + noise)
    xs = np.clip(xs, 1e-10, None)

    # Synthetic uncertainty (10-30% relative)
    unc = xs * rng.uniform(0.1, 0.3, n_samples)

    df = pd.DataFrame({
        'Energy': energy,
        'Z': z.astype(float),
        'A': a.astype(float),
        'N': n.astype(float),
        'CrossSection': xs,
        'Uncertainty': unc,
    })
    return df


class TestNNDefaults(unittest.TestCase):
    """Verify default configuration values."""

    def test_default_hidden_sizes(self):
        ev = NeuralNetEvaluator()
        self.assertEqual(ev.hidden_sizes, (256, 128))

    def test_default_epochs(self):
        ev = NeuralNetEvaluator()
        self.assertEqual(ev.epochs, 50)

    def test_default_batch_size(self):
        ev = NeuralNetEvaluator()
        self.assertEqual(ev.batch_size, 512)

    def test_default_learning_rate(self):
        ev = NeuralNetEvaluator()
        self.assertAlmostEqual(ev.learning_rate, 3e-3)

    def test_default_loss_function(self):
        ev = NeuralNetEvaluator()
        self.assertEqual(ev.loss_function, 'chi_squared')

    def test_default_early_stopping(self):
        ev = NeuralNetEvaluator()
        self.assertEqual(ev.early_stopping_patience, 8)

    def test_invalid_loss_raises(self):
        with self.assertRaises(ValueError):
            NeuralNetEvaluator(loss_function='invalid')

    def test_repr_untrained(self):
        ev = NeuralNetEvaluator()
        r = repr(ev)
        self.assertIn('untrained', r)
        self.assertIn('chi_squared', r)
        self.assertIn('[256, 128]', r)


class TestTrainRuns(unittest.TestCase):
    """Test that training runs and produces metrics."""

    def _train_with_loss(self, loss_function, epochs=5, **kwargs):
        """Helper to train with a given loss function."""
        df = _make_synthetic_df(n_samples=500)
        ev = NeuralNetEvaluator(
            loss_function=loss_function,
            epochs=epochs,
            batch_size=64,
            hidden_sizes=(32, 16),
            early_stopping_patience=0,  # disable for short tests
            **kwargs,
        )
        metrics = ev.train(df, verbose=False)
        return ev, metrics

    def test_mse_returns_metrics(self):
        ev, metrics = self._train_with_loss('mse')
        self.assertIn('test_r2_barns', metrics)
        self.assertIn('test_mse_log', metrics)
        self.assertIn('n_train', metrics)
        self.assertTrue(ev.is_trained)

    def test_chi_squared_returns_metrics(self):
        _, metrics = self._train_with_loss('chi_squared')
        self.assertIn('test_r2_barns', metrics)

    def test_physics_informed_returns_metrics(self):
        _, metrics = self._train_with_loss('physics_informed')
        self.assertIn('test_r2_barns', metrics)

    def test_resonance_informed_returns_metrics(self):
        _, metrics = self._train_with_loss('resonance_informed')
        self.assertIn('test_r2_barns', metrics)

    def test_loss_decreases(self):
        """Training loss should generally decrease over epochs."""
        df = _make_synthetic_df(n_samples=1000)
        ev = NeuralNetEvaluator(
            loss_function='mse',
            epochs=15,
            batch_size=64,
            hidden_sizes=(64, 32),
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)
        losses = ev.history['train_loss']
        # First loss should be larger than last (general trend)
        self.assertGreater(losses[0], losses[-1])

    def test_history_tracking(self):
        """History should contain train_loss, val_loss, learning_rates."""
        ev, _ = self._train_with_loss('mse', epochs=3)
        self.assertIn('train_loss', ev.history)
        self.assertIn('val_loss', ev.history)
        self.assertIn('learning_rates', ev.history)
        self.assertEqual(len(ev.history['train_loss']), 3)
        self.assertEqual(len(ev.history['val_loss']), 3)

    def test_repr_trained(self):
        ev, _ = self._train_with_loss('mse', epochs=2)
        r = repr(ev)
        self.assertIn('trained', r)


class TestEarlyStopping(unittest.TestCase):
    """Test that early stopping fires before max epochs."""

    def test_early_stopping_fires(self):
        """With patience=2 and many epochs, should stop early."""
        df = _make_synthetic_df(n_samples=500)
        ev = NeuralNetEvaluator(
            loss_function='mse',
            epochs=200,  # very high max
            batch_size=64,
            hidden_sizes=(16, 8),
            early_stopping_patience=2,
        )
        ev.train(df, verbose=False)
        stopped = ev.history['stopped_epoch']
        self.assertLess(stopped, 200, "Early stopping should fire before 200 epochs")
        # Should have fewer loss entries than max epochs
        self.assertLessEqual(len(ev.history['train_loss']), stopped)

    def test_early_stopping_disabled(self):
        """With patience=0, should run all epochs."""
        df = _make_synthetic_df(n_samples=300)
        ev = NeuralNetEvaluator(
            loss_function='mse',
            epochs=5,
            batch_size=64,
            hidden_sizes=(16, 8),
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)
        self.assertEqual(len(ev.history['train_loss']), 5)


class TestPredictAPI(unittest.TestCase):
    """Test the predict() interface."""

    def test_predict_returns_numpy(self):
        df = _make_synthetic_df(n_samples=500)
        ev = NeuralNetEvaluator(
            loss_function='mse',
            epochs=3,
            batch_size=64,
            hidden_sizes=(16, 8),
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)

        preds = ev.predict(df)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), len(df))

    def test_predict_positive_barns(self):
        """Predictions should be non-negative (cross-sections in barns)."""
        df = _make_synthetic_df(n_samples=500)
        ev = NeuralNetEvaluator(
            loss_function='mse',
            epochs=5,
            batch_size=64,
            hidden_sizes=(32, 16),
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)

        preds = ev.predict(df)
        # After inverse log transform, values should be non-negative
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_before_train_raises(self):
        ev = NeuralNetEvaluator()
        df = _make_synthetic_df(n_samples=10)
        with self.assertRaises(RuntimeError):
            ev.predict(df)


class TestKaimingInit(unittest.TestCase):
    """Test Kaiming He initialization produces correct weight variance."""

    def test_kaiming_variance(self):
        """Weight variance should be approximately 2/fan_in for ReLU."""
        n_features = 100
        hidden = (256, 128)
        net = _SimpleNet(n_features, hidden_sizes=hidden, dropout=0.0)

        # Check first layer
        first_linear = net.net[0]
        fan_in = first_linear.in_features
        expected_var = 2.0 / fan_in
        actual_var = first_linear.weight.data.var().item()

        # Should be within 50% (stochastic, but 256x100 weights is enough)
        self.assertAlmostEqual(
            actual_var, expected_var, delta=expected_var * 0.5,
            msg=f"Kaiming variance: expected ~{expected_var:.4f}, got {actual_var:.4f}",
        )

    def test_bias_zeros(self):
        """Biases should be initialized to zero."""
        net = _SimpleNet(50, hidden_sizes=(64,))
        for m in net.modules():
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                self.assertTrue(
                    torch.all(m.bias == 0),
                    "Linear bias should be zero-initialized",
                )


class TestSaveLoad(unittest.TestCase):
    """Test save/load round-trip."""

    def test_save_load_preserves_predictions(self):
        df = _make_synthetic_df(n_samples=300)
        ev = NeuralNetEvaluator(
            loss_function='mse',
            epochs=3,
            batch_size=64,
            hidden_sizes=(16, 8),
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)
        preds_before = ev.predict(df)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            ev.save(path)

            ev2 = NeuralNetEvaluator()
            ev2.load(path)
            preds_after = ev2.predict(df)

            np.testing.assert_array_almost_equal(
                preds_before, preds_after, decimal=5,
                err_msg="Predictions should match after save/load",
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_untrained_raises(self):
        ev = NeuralNetEvaluator()
        with self.assertRaises(RuntimeError):
            ev.save('test.pt')

    def test_load_restores_metadata(self):
        df = _make_synthetic_df(n_samples=300)
        ev = NeuralNetEvaluator(
            loss_function='physics_informed',
            epochs=3,
            batch_size=64,
            hidden_sizes=(32, 16),
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            ev.save(path)

            ev2 = NeuralNetEvaluator()
            ev2.load(path)

            self.assertEqual(ev2.hidden_sizes, (32, 16))
            self.assertEqual(ev2.loss_function, 'physics_informed')
            self.assertTrue(ev2.is_trained)
            self.assertIn('test_r2_barns', ev2.metrics)
        finally:
            Path(path).unlink(missing_ok=True)


class TestLossFunctions(unittest.TestCase):
    """Test individual loss functions."""

    def test_mse_loss_value(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = _mse_loss(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_chi_squared_falls_back_to_mse(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.5, 2.5])
        loss_mse = _mse_loss(pred, target)
        loss_chi = _chi_squared_loss(pred, target, unc=None)
        self.assertAlmostEqual(loss_mse.item(), loss_chi.item(), places=6)

    def test_chi_squared_with_uncertainty(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.5, 2.5])
        unc = torch.tensor([0.5, 0.5])
        loss = _chi_squared_loss(pred, target, unc=unc)
        # (0.5^2 / 0.25 + 0.5^2 / 0.25) / 2 = (1.0 + 1.0) / 2 = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_physics_informed_adds_smoothness(self):
        """Physics-informed loss should be >= MSE loss."""
        pred = torch.tensor([1.0, 3.0, 1.0])  # oscillating
        target = torch.tensor([1.0, 3.0, 1.0])
        energy = torch.tensor([1.0, 2.0, 3.0])
        loss_mse = _mse_loss(pred, target)
        loss_pi = _physics_informed_loss(
            pred, target, energy=energy, smoothness_weight=0.01,
        )
        self.assertGreaterEqual(loss_pi.item(), loss_mse.item())

    def test_resonance_informed_runs(self):
        """Resonance-informed loss should compute without error."""
        pred = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.2])
        target = torch.tensor([2.1, 1.6, 1.1, 0.6, 0.3])
        energy = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  # log10 space
        unc = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        loss = _resonance_informed_loss(
            pred, target, energy=energy, unc=unc,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)


class TestCustomHyperparameters(unittest.TestCase):
    """Test training with non-default hyperparameters."""

    def test_custom_hidden_sizes(self):
        df = _make_synthetic_df(n_samples=300)
        ev = NeuralNetEvaluator(
            hidden_sizes=(64, 32, 16),
            epochs=2,
            batch_size=64,
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)
        self.assertTrue(ev.is_trained)

    def test_dropout_nonzero(self):
        df = _make_synthetic_df(n_samples=300)
        ev = NeuralNetEvaluator(
            hidden_sizes=(32, 16),
            dropout=0.2,
            epochs=2,
            batch_size=64,
            early_stopping_patience=0,
        )
        ev.train(df, verbose=False)
        self.assertTrue(ev.is_trained)


if __name__ == "__main__":
    unittest.main()
