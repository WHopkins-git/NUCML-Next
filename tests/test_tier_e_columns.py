"""
Regression test: Tier E Q-value columns must survive the column filter.

Prior bugs:
1. _get_tier_columns() listed 'Q_alpha' but output is 'Q_alpha_MeV'
2. _add_tier_e_features() checked for 'Q_alpha' but enrichment stores 'Q_alpha_keV'

The synthetic data uses _keV suffixed columns to match what AME2020DataEnricher
actually produces (see enrichment.py lines 384-488).
"""

import unittest

import numpy as np
import pandas as pd

from nucml_next.data.features import FeatureGenerator


TIER_E_EXPECTED = [
    'Q_alpha_MeV', 'Q_2beta_minus_MeV', 'Q_ep_MeV', 'Q_beta_n_MeV',
    'Q_4beta_minus_MeV', 'Q_d_alpha_MeV', 'Q_p_alpha_MeV', 'Q_n_alpha_MeV',
]

# Source columns as produced by AME2020DataEnricher (keV units)
TIER_E_SOURCE_COLS = [
    'Q_alpha_keV', 'Q_2beta_minus_keV', 'Q_ep_keV', 'Q_beta_n_keV',
    'Q_4beta_minus_keV', 'Q_d_alpha_keV', 'Q_p_alpha_keV', 'Q_n_alpha_keV',
]


def _make_enriched_df(n=5):
    """Build a minimal DataFrame that looks like on-demand AME enrichment output.

    Column names use _keV suffix to match what AME2020DataEnricher.load_all()
    produces (enrichment.py lines 384-488).
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'Z': [92] * n,
        'A': [235] * n,
        'N': [143] * n,
        'Energy': np.logspace(-2, 6, n),
        'CrossSection': rng.uniform(0.1, 100, n),
        'Uncertainty': rng.uniform(0.01, 1, n),
        'MT': [18] * n,
        # Q-value columns in keV (as stored by AME enrichment)
        'Q_alpha_keV': rng.uniform(4000, 5000, n),
        'Q_2beta_minus_keV': rng.uniform(1000, 2000, n),
        'Q_ep_keV': rng.uniform(500, 1500, n),
        'Q_beta_n_keV': rng.uniform(200, 800, n),
        'Q_4beta_minus_keV': rng.uniform(3000, 4000, n),
        'Q_d_alpha_keV': rng.uniform(10000, 15000, n),
        'Q_p_alpha_keV': rng.uniform(8000, 12000, n),
        'Q_n_alpha_keV': rng.uniform(5000, 9000, n),
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

        for col in TIER_E_SOURCE_COLS:
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
