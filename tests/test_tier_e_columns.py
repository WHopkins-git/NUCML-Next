"""
Regression test: Tier E Q-value columns must survive the column filter.

Prior bug: _get_tier_columns() listed 'Q_alpha' etc., but _add_tier_e_features()
renamed them to 'Q_alpha_MeV' etc.  The column filter then silently dropped all 8.
"""

import unittest

import numpy as np
import pandas as pd

from nucml_next.data.features import FeatureGenerator


TIER_E_EXPECTED = [
    'Q_alpha_MeV', 'Q_2beta_minus_MeV', 'Q_ep_MeV', 'Q_beta_n_MeV',
    'Q_4beta_minus_MeV', 'Q_d_alpha_MeV', 'Q_p_alpha_MeV', 'Q_n_alpha_MeV',
]


def _make_enriched_df(n=5):
    """Build a minimal DataFrame that looks like pre-enriched Parquet output."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'Z': [92] * n,
        'A': [235] * n,
        'N': [143] * n,
        'Energy': np.logspace(-2, 6, n),
        'CrossSection': rng.uniform(0.1, 100, n),
        'Uncertainty': rng.uniform(0.01, 1, n),
        'MT': [18] * n,
        # Pre-enriched Q-value columns (keV, as stored in Parquet)
        'Q_alpha': rng.uniform(4000, 5000, n),
        'Q_2beta_minus': rng.uniform(1000, 2000, n),
        'Q_ep': rng.uniform(500, 1500, n),
        'Q_beta_n': rng.uniform(200, 800, n),
        'Q_4beta_minus': rng.uniform(3000, 4000, n),
        'Q_d_alpha': rng.uniform(10000, 15000, n),
        'Q_p_alpha': rng.uniform(8000, 12000, n),
        'Q_n_alpha': rng.uniform(5000, 9000, n),
    })
    return df


class TestTierEColumns(unittest.TestCase):
    """Tier E Q-value columns must appear in generate_features() output."""

    def test_tier_e_columns_present(self):
        df = _make_enriched_df()
        gen = FeatureGenerator()
        result = gen.generate_features(df, tiers=['A', 'E'])

        for col in TIER_E_EXPECTED:
            self.assertIn(col, result.columns, f"Missing Tier E column: {col}")

    def test_tier_e_values_converted_kev_to_mev(self):
        df = _make_enriched_df()
        gen = FeatureGenerator()
        result = gen.generate_features(df, tiers=['A', 'E'])

        # Q_alpha was ~4000-5000 keV, so MeV should be ~4-5
        self.assertTrue((result['Q_alpha_MeV'] < 10).all())
        self.assertTrue((result['Q_alpha_MeV'] > 0).all())

    def test_original_kev_columns_removed(self):
        df = _make_enriched_df()
        gen = FeatureGenerator()
        result = gen.generate_features(df, tiers=['A', 'E'])

        for col in ['Q_alpha', 'Q_2beta_minus', 'Q_ep', 'Q_beta_n',
                     'Q_4beta_minus', 'Q_d_alpha', 'Q_p_alpha', 'Q_n_alpha']:
            self.assertNotIn(col, result.columns,
                             f"Original keV column should be removed: {col}")

    def test_all_tiers_includes_tier_e(self):
        df = _make_enriched_df()
        gen = FeatureGenerator()
        result = gen.generate_features(df, tiers=['A', 'B', 'C', 'D', 'E'])

        for col in TIER_E_EXPECTED:
            self.assertIn(col, result.columns, f"Missing with all tiers: {col}")


if __name__ == '__main__':
    unittest.main()
