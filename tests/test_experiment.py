"""
Tests for ExperimentManager, HoldoutConfig, and compute_holdout_metrics.

Uses synthetic data — no real Parquet files required.
"""

import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from nucml_next.experiment import (
    ExperimentManager,
    HoldoutConfig,
    compute_holdout_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Minimal DataFrame resembling to_tabular() output."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'Z': rng.choice([92, 17, 26], size=n),
        'A': rng.choice([233, 235, 35, 56], size=n),
        'N': rng.integers(20, 150, size=n),
        'Energy': rng.uniform(1e-3, 1e7, size=n),
        'MT': rng.choice([1, 2, 18, 102, 103], size=n),
        'CrossSection': rng.uniform(0.01, 1000, size=n),
        'Uncertainty': rng.uniform(0.001, 10, size=n),
        'Entry': rng.choice(['E0001', 'E0002', 'E0003', 'E9999'], size=n),
        'out_n': rng.integers(0, 3, size=n).astype(float),
        'out_p': rng.integers(0, 2, size=n).astype(float),
        'out_a': rng.integers(0, 2, size=n).astype(float),
    })
    return df


def _make_trained_evaluator(df: pd.DataFrame):
    """
    Train a minimal DecisionTreeEvaluator on *df* so we can test
    save / load round-trips.
    """
    from nucml_next.baselines import DecisionTreeEvaluator
    from nucml_next.data.selection import TransformationConfig

    config = TransformationConfig(
        log_target=True, log_energy=True, scaler_type='minmax',
    )
    ev = DecisionTreeEvaluator(max_depth=3, random_state=42)
    ev.train(df, transformation_config=config)
    return ev


# ===================================================================
# TestHoldoutConfig
# ===================================================================

class TestHoldoutConfig(unittest.TestCase):
    """HoldoutConfig rule engine."""

    def setUp(self):
        self.df = _make_synthetic_df(500)

    # ---- single-key rules ------------------------------------------------

    def test_isotope_rule(self):
        cfg = HoldoutConfig(rules=[{'Z': 92, 'A': 233}])
        mask = cfg.build_mask(self.df)
        expected = (self.df['Z'] == 92) & (self.df['A'] == 233)
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    def test_mt_rule(self):
        cfg = HoldoutConfig(rules=[{'MT': 18}])
        mask = cfg.build_mask(self.df)
        expected = self.df['MT'] == 18
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    def test_mt_list_rule(self):
        cfg = HoldoutConfig(rules=[{'MT': [18, 102]}])
        mask = cfg.build_mask(self.df)
        expected = self.df['MT'].isin([18, 102])
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    def test_energy_range_rule(self):
        cfg = HoldoutConfig(rules=[{'energy_range': (1.0, 100.0)}])
        mask = cfg.build_mask(self.df)
        expected = (self.df['Energy'] >= 1.0) & (self.df['Energy'] <= 100.0)
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    def test_xs_range_rule(self):
        cfg = HoldoutConfig(rules=[{'xs_range': (10.0, 500.0)}])
        mask = cfg.build_mask(self.df)
        expected = (self.df['CrossSection'] >= 10.0) & (self.df['CrossSection'] <= 500.0)
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    def test_entry_rule(self):
        cfg = HoldoutConfig(rules=[{'Entry': 'E9999'}])
        mask = cfg.build_mask(self.df)
        expected = self.df['Entry'] == 'E9999'
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    def test_entry_list_rule(self):
        cfg = HoldoutConfig(rules=[{'Entry': ['E9999', 'E0001']}])
        mask = cfg.build_mask(self.df)
        expected = self.df['Entry'].isin(['E9999', 'E0001'])
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    # ---- intersection within a rule --------------------------------------

    def test_intersection(self):
        cfg = HoldoutConfig(rules=[{
            'Z': 92, 'A': 235, 'MT': 102, 'energy_range': (1e-3, 1.0),
        }])
        mask = cfg.build_mask(self.df)
        expected = (
            (self.df['Z'] == 92)
            & (self.df['A'] == 235)
            & (self.df['MT'] == 102)
            & (self.df['Energy'] >= 1e-3)
            & (self.df['Energy'] <= 1.0)
        )
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    # ---- union across rules ----------------------------------------------

    def test_union_across_rules(self):
        cfg = HoldoutConfig(rules=[
            {'Z': 92, 'A': 233},
            {'MT': 18},
        ])
        mask = cfg.build_mask(self.df)
        r1 = (self.df['Z'] == 92) & (self.df['A'] == 233)
        r2 = self.df['MT'] == 18
        expected = r1 | r2
        pd.testing.assert_series_equal(mask, expected, check_names=False)

    # ---- edge cases -------------------------------------------------------

    def test_empty_rules(self):
        cfg = HoldoutConfig(rules=[])
        mask = cfg.build_mask(self.df)
        self.assertFalse(mask.any())

    def test_bool_truthy(self):
        self.assertTrue(bool(HoldoutConfig(rules=[{'MT': 18}])))
        self.assertFalse(bool(HoldoutConfig(rules=[])))

    # ---- split ------------------------------------------------------------

    def test_split(self):
        cfg = HoldoutConfig(rules=[{'Z': 92, 'A': 233}])
        df_train, df_hold = cfg.split(self.df)
        expected_hold = ((self.df['Z'] == 92) & (self.df['A'] == 233)).sum()
        self.assertEqual(len(df_hold), expected_hold)
        self.assertEqual(len(df_train) + len(df_hold), len(self.df))

    # ---- serialisation ----------------------------------------------------

    def test_to_dict_roundtrip(self):
        cfg = HoldoutConfig(rules=[
            {'Z': 92, 'A': 235, 'MT': 102, 'energy_range': (1e-3, 1.0)},
            {'Entry': ['E9999', 'E0001']},
            {'xs_range': (0.1, 100.0)},
        ])
        d = cfg.to_dict()
        cfg2 = HoldoutConfig.from_dict(d)
        self.assertEqual(len(cfg2.rules), 3)
        # Verify energy_range survives as tuple
        self.assertAlmostEqual(cfg2.rules[0]['energy_range'][0], 1e-3)
        # Verify xs_range survives as tuple
        self.assertAlmostEqual(cfg2.rules[2]['xs_range'][0], 0.1)
        self.assertAlmostEqual(cfg2.rules[2]['xs_range'][1], 100.0)

    def test_from_legacy(self):
        cfg = HoldoutConfig.from_legacy([(92, 235), (17, 35)])
        self.assertEqual(len(cfg.rules), 2)
        self.assertEqual(cfg.rules[0], {'Z': 92, 'A': 235})
        self.assertEqual(cfg.rules[1], {'Z': 17, 'A': 35})

    # ---- repr -------------------------------------------------------------

    def test_repr(self):
        cfg = HoldoutConfig(rules=[{'Z': 92, 'A': 235, 'MT': 102}])
        r = repr(cfg)
        self.assertIn('Rule 1', r)
        self.assertIn('Z=92', r)


# ===================================================================
# TestExperimentManager
# ===================================================================

class TestExperimentManager(unittest.TestCase):
    """ExperimentManager save / load round-trip."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='nucml_exp_test_')
        cls.df = _make_synthetic_df(300, seed=7)
        cls.model = _make_trained_evaluator(cls.df)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_save_creates_directory(self):
        mgr = ExperimentManager(base_dir=self.tmpdir)
        exp_dir = mgr.save_experiment(
            self.model, 'decision_tree',
        )
        self.assertTrue(exp_dir.exists())
        self.assertTrue((exp_dir / 'model.joblib').exists())
        self.assertTrue((exp_dir / 'scaler_state.pkl').exists())
        self.assertTrue((exp_dir / 'properties.yaml').exists())
        self.assertTrue((exp_dir / 'figures').is_dir())

    def test_properties_yaml_content(self):
        import yaml
        mgr = ExperimentManager(base_dir=self.tmpdir)
        holdout = HoldoutConfig(rules=[{'Z': 92, 'A': 233}])
        exp_dir = mgr.save_experiment(
            self.model, 'decision_tree',
            holdout_config=holdout,
        )
        with open(exp_dir / 'properties.yaml') as f:
            props = yaml.safe_load(f)
        self.assertEqual(props['model_type'], 'decision_tree')
        self.assertIn('model_params', props)
        self.assertIn('transformation', props)
        self.assertIn('holdout', props)
        self.assertIn('training_metrics', props)
        self.assertIn('feature_columns', props)

    def test_save_load_roundtrip(self):
        mgr = ExperimentManager(base_dir=self.tmpdir)
        exp_dir = mgr.save_experiment(self.model, 'decision_tree')

        env = ExperimentManager.load_experiment(exp_dir)
        self.assertTrue(env['model'].is_trained)
        self.assertIsNotNone(env['pipeline'])
        self.assertIsNotNone(env['properties'])

    def test_load_predict(self):
        mgr = ExperimentManager(base_dir=self.tmpdir)
        exp_dir = mgr.save_experiment(self.model, 'decision_tree')

        env = ExperimentManager.load_experiment(exp_dir)
        loaded_model = env['model']
        preds = loaded_model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_holdout_config_persisted(self):
        mgr = ExperimentManager(base_dir=self.tmpdir)
        cfg = HoldoutConfig(rules=[
            {'Z': 92, 'A': 235, 'MT': 102, 'energy_range': (1e-3, 1.0)},
        ])
        exp_dir = mgr.save_experiment(
            self.model, 'decision_tree', holdout_config=cfg,
        )
        env = ExperimentManager.load_experiment(exp_dir)
        self.assertIsNotNone(env['holdout_config'])
        self.assertEqual(len(env['holdout_config'].rules), 1)
        self.assertAlmostEqual(
            env['holdout_config'].rules[0]['energy_range'][0], 1e-3,
        )


# ===================================================================
# TestComputeHoldoutMetrics
# ===================================================================

class TestComputeHoldoutMetrics(unittest.TestCase):
    """compute_holdout_metrics returns correct structure."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_synthetic_df(300, seed=99)
        cls.model = _make_trained_evaluator(cls.df)

    def test_metrics_keys(self):
        metrics = compute_holdout_metrics(self.model, self.df)
        expected_keys = {
            'holdout_n',
            'holdout_mse_log', 'holdout_mae_log', 'holdout_r2_log',
            'holdout_mse_barns', 'holdout_mae_barns', 'holdout_r2_barns',
            'holdout_medae_barns',
        }
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_barns_metrics_finite(self):
        metrics = compute_holdout_metrics(self.model, self.df)
        self.assertTrue(np.isfinite(metrics['holdout_mse_barns']))
        self.assertTrue(np.isfinite(metrics['holdout_mae_barns']))
        self.assertGreater(metrics['holdout_n'], 0)

    def test_holdout_n_matches(self):
        metrics = compute_holdout_metrics(self.model, self.df)
        # All rows should be valid (synthetic data has no NaN)
        self.assertEqual(metrics['holdout_n'], len(self.df))


# ===================================================================
# TestDataSelectionLegacyBridge
# ===================================================================

class TestDataSelectionLegacyBridge(unittest.TestCase):
    """Verify holdout_isotopes→holdout_config auto-conversion."""

    def test_legacy_converts(self):
        from nucml_next.data.selection import DataSelection
        sel = DataSelection(holdout_isotopes=[(92, 235), (17, 35)])
        self.assertIsNotNone(sel.holdout_config)
        self.assertEqual(len(sel.holdout_config.rules), 2)

    def test_mutual_exclusion(self):
        from nucml_next.data.selection import DataSelection
        cfg = HoldoutConfig(rules=[{'Z': 92, 'A': 235}])
        with self.assertRaises(ValueError):
            DataSelection(
                holdout_isotopes=[(92, 235)],
                holdout_config=cfg,
            )

    def test_holdout_config_direct(self):
        from nucml_next.data.selection import DataSelection
        cfg = HoldoutConfig(rules=[{'MT': 18}])
        sel = DataSelection(holdout_config=cfg)
        self.assertIs(sel.holdout_config, cfg)
        self.assertIsNone(sel.holdout_isotopes)


if __name__ == '__main__':
    unittest.main()
