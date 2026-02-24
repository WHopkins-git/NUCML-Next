"""
Tests for spectrum-averaged data filter and DataSelection field.

Uses synthetic DataFrames — no real Parquet files required.
"""

import unittest

import numpy as np
import pandas as pd

from nucml_next.data.selection import SPECTRUM_AVERAGED_SF8, DataSelection


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _apply_spectrum_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the spectrum-averaged filter to a DataFrame.

    Mirrors the exact logic in dataset.py so tests validate the algorithm
    without needing to go through Parquet I/O.
    """
    if 'sf8' not in df.columns:
        return df
    sf8_tokens = df['sf8'].fillna('').str.split('/')
    is_avg = sf8_tokens.explode().isin(SPECTRUM_AVERAGED_SF8).groupby(level=0).any()
    return df[~is_avg]


def _make_df(sf8_values):
    """Build a minimal DataFrame with an sf8 column."""
    n = len(sf8_values)
    return pd.DataFrame({
        'Z': [92] * n,
        'A': [235] * n,
        'Energy': np.linspace(1, 100, n),
        'CrossSection': np.ones(n),
        'sf8': sf8_values,
    })


# ===================================================================
# TestSpectrumAveragedConstant
# ===================================================================

class TestSpectrumAveragedConstant(unittest.TestCase):
    """Verify the SPECTRUM_AVERAGED_SF8 set contents."""

    def test_expected_codes(self):
        expected = {'MXW', 'SPA', 'SDT', 'FST', 'FIS', 'TTA', 'BRA', 'BRS', 'AV'}
        self.assertEqual(SPECTRUM_AVERAGED_SF8, expected)

    def test_is_a_set(self):
        self.assertIsInstance(SPECTRUM_AVERAGED_SF8, set)


# ===================================================================
# TestSpectrumAveragedFilter
# ===================================================================

class TestSpectrumAveragedFilter(unittest.TestCase):
    """Test the spectrum-averaged filter logic (same algorithm as dataset.py)."""

    def test_excludes_exact_sf8_codes(self):
        """Each code in SPECTRUM_AVERAGED_SF8 is individually removed."""
        for code in SPECTRUM_AVERAGED_SF8:
            df = _make_df([code, None, ''])
            result = _apply_spectrum_filter(df)
            # The code row should be removed; NaN and empty kept
            self.assertEqual(len(result), 2, f"Code '{code}' should be removed")

    def test_excludes_compound_sf8(self):
        """Compound codes like BRA/REL, SDT/AV are removed (token splitting)."""
        df = _make_df(['BRA/REL', 'SDT/AV', 'BRS/REL', None])
        result = _apply_spectrum_filter(df)
        # All compound codes should be removed; only NaN row survives
        self.assertEqual(len(result), 1)

    def test_keeps_nan_and_empty_sf8(self):
        """NaN and empty sf8 values are preserved (monoenergetic data)."""
        df = _make_df([None, np.nan, '', None])
        result = _apply_spectrum_filter(df)
        self.assertEqual(len(result), 4)

    def test_keeps_safe_sf8_codes(self):
        """RAW, S0, 2G, RM, REL are NOT removed by this filter."""
        safe_codes = ['RAW', 'S0', '2G', 'RM', 'REL', 'FCT', 'RAT', 'RTE',
                      'ETA', 'ALF', 'RES', 'G']
        # None of these are in SPECTRUM_AVERAGED_SF8
        for code in safe_codes:
            if code in SPECTRUM_AVERAGED_SF8:
                continue  # skip if somehow in the set
            df = _make_df([code])
            result = _apply_spectrum_filter(df)
            self.assertEqual(len(result), 1, f"Code '{code}' should be kept")

    def test_no_sf8_column_skips_silently(self):
        """When sf8 column is absent, filter returns DataFrame unchanged."""
        df = pd.DataFrame({
            'Z': [92, 92],
            'A': [235, 235],
            'Energy': [1.0, 2.0],
            'CrossSection': [10.0, 20.0],
        })
        result = _apply_spectrum_filter(df)
        self.assertEqual(len(result), 2)

    def test_mixed_kept_and_removed(self):
        """Mix of spectrum-averaged and non-averaged codes."""
        df = _make_df([None, 'MXW', '', 'SPA', 'RAW', 'FIS', 'AV'])
        result = _apply_spectrum_filter(df)
        # Kept: None, '', RAW = 3 rows
        # Removed: MXW, SPA, FIS, AV = 4 rows
        self.assertEqual(len(result), 3)

    def test_compound_with_safe_token_still_removed(self):
        """A compound like MXW/REL is removed because MXW is in the set."""
        df = _make_df(['MXW/REL'])
        result = _apply_spectrum_filter(df)
        self.assertEqual(len(result), 0)

    def test_safe_compound_kept(self):
        """Compound of safe codes like RAW/REL is kept."""
        df = _make_df(['RAW/REL'])
        result = _apply_spectrum_filter(df)
        self.assertEqual(len(result), 1)


# ===================================================================
# TestDataSelectionField
# ===================================================================

class TestDataSelectionField(unittest.TestCase):
    """Verify exclude_spectrum_averaged field on DataSelection."""

    def test_default_is_true(self):
        """DataSelection() defaults to exclude_spectrum_averaged=True."""
        sel = DataSelection()
        self.assertTrue(sel.exclude_spectrum_averaged)

    def test_can_set_false(self):
        """exclude_spectrum_averaged=False is accepted."""
        sel = DataSelection(exclude_spectrum_averaged=False)
        self.assertFalse(sel.exclude_spectrum_averaged)

    def test_repr_includes_spectrum_averaged(self):
        """__repr__ includes the exclude_spectrum_averaged setting."""
        sel = DataSelection()
        r = repr(sel)
        self.assertIn('Exclude spectrum-averaged', r)

    def test_repr_shows_false_when_disabled(self):
        sel = DataSelection(exclude_spectrum_averaged=False)
        r = repr(sel)
        self.assertIn('Exclude spectrum-averaged: False', r)


if __name__ == '__main__':
    unittest.main()
