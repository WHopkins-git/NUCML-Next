"""
Unit Tests for ENDF Auto-Fallback Logic
========================================

Tests for the get_cross_section_best() auto-decision logic that determines
whether to use raw MF=3 data or fetch resonance-reconstructed pointwise
data from NNDC Sigma.

Test isotopes:
- Cl-35 (n,p) MT=103: Sparse MF=3 data (79 points) -> auto triggers NNDC
- U-235 fission MT=18: Dense MF=3 data (839 points) -> auto stays on MF=3
"""

import unittest
from unittest.mock import patch, MagicMock
import warnings
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.visualization.endf_reader import ENDFReader, NNDCSigmaFetcher

# Paths to local ENDF test files
DATA_DIR = Path(__file__).parent.parent / "data" / "ENDF-B" / "neutrons"
CL35_FILE = DATA_DIR / "n-017_Cl_035.endf"
U235_FILE = DATA_DIR / "n-092_U_235.endf"


class TestGetDataInfo(unittest.TestCase):
    """Verify get_data_info() correctly classifies sparse vs dense reactions."""

    @unittest.skipUnless(CL35_FILE.exists(), "Cl-35 ENDF file not available")
    def test_cl35_np_is_sparse(self):
        """Cl-35 (n,p) MT=103 must be classified as sparse."""
        reader = ENDFReader(CL35_FILE)
        info = reader.get_data_info(103)

        self.assertTrue(info["is_sparse"], "Cl-35 MT=103 should be sparse (<200 pts)")
        self.assertLess(info["n_points"], 200)
        self.assertTrue(info["has_placeholders"], "Cl-35 MT=103 should have placeholders")

    @unittest.skipUnless(U235_FILE.exists(), "U-235 ENDF file not available")
    def test_u235_fission_is_dense(self):
        """U-235 fission MT=18 must be classified as dense."""
        reader = ENDFReader(U235_FILE)
        info = reader.get_data_info(18)

        self.assertFalse(info["is_sparse"], "U-235 MT=18 should not be sparse")
        self.assertGreaterEqual(info["n_points"], 200)


class TestGetCrossSectionBest(unittest.TestCase):
    """Test the auto-decision logic in get_cross_section_best()."""

    @unittest.skipUnless(CL35_FILE.exists(), "Cl-35 ENDF file not available")
    def test_auto_sparse_attempts_nndc(self):
        """Auto mode on sparse Cl-35 MT=103 should attempt NNDC fetch."""
        reader = ENDFReader(CL35_FILE)

        # Mock the NNDC fetcher to avoid real network calls
        fake_energies = np.logspace(-5, 7, 5000)
        fake_xs = np.random.exponential(0.1, 5000)

        with patch.object(ENDFReader, '_fetch_nndc',
                          return_value=(fake_energies, fake_xs, "nndc_sigma")) as mock_fetch:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                energies, xs, source = reader.get_cross_section_best(mt=103)

            # Should have called NNDC
            mock_fetch.assert_called_once()
            self.assertEqual(source, "nndc_sigma")

            # Should have emitted a sparsity warning
            sparse_warnings = [x for x in w if "sparse" in str(x.message).lower()]
            self.assertGreater(len(sparse_warnings), 0,
                               "Expected a warning about sparse MF=3 data")

    @unittest.skipUnless(U235_FILE.exists(), "U-235 ENDF file not available")
    def test_auto_dense_stays_endf(self):
        """Auto mode on dense U-235 MT=18 should NOT attempt NNDC."""
        reader = ENDFReader(U235_FILE)

        with patch.object(ENDFReader, '_fetch_nndc') as mock_fetch:
            energies, xs, source = reader.get_cross_section_best(mt=18)

        # Should NOT have called NNDC
        mock_fetch.assert_not_called()
        self.assertEqual(source, "endf_mf3")
        self.assertGreater(len(energies), 200)

    @unittest.skipUnless(CL35_FILE.exists(), "Cl-35 ENDF file not available")
    def test_prefer_endf_skips_nndc(self):
        """prefer='endf' should always use raw MF=3, even for sparse data."""
        reader = ENDFReader(CL35_FILE)

        with patch.object(ENDFReader, '_fetch_nndc') as mock_fetch:
            energies, xs, source = reader.get_cross_section_best(mt=103, prefer="endf")

        mock_fetch.assert_not_called()
        self.assertEqual(source, "endf_mf3")

    @unittest.skipUnless(CL35_FILE.exists(), "Cl-35 ENDF file not available")
    def test_prefer_nndc_always_fetches(self):
        """prefer='nndc' should always fetch from NNDC, even for dense data."""
        reader = ENDFReader(CL35_FILE)

        fake_energies = np.logspace(-5, 7, 5000)
        fake_xs = np.random.exponential(0.1, 5000)

        with patch.object(ENDFReader, '_fetch_nndc',
                          return_value=(fake_energies, fake_xs, "nndc_sigma")) as mock_fetch:
            energies, xs, source = reader.get_cross_section_best(mt=103, prefer="nndc")

        mock_fetch.assert_called_once()
        self.assertEqual(source, "nndc_sigma")

    @unittest.skipUnless(CL35_FILE.exists(), "Cl-35 ENDF file not available")
    def test_auto_fallback_on_nndc_failure(self):
        """When NNDC fetch fails in auto mode, should fall back to MF=3."""
        reader = ENDFReader(CL35_FILE)

        with patch.object(ENDFReader, '_fetch_nndc',
                          side_effect=RuntimeError("NNDC timeout")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                energies, xs, source = reader.get_cross_section_best(mt=103)

        # Should have fallen back to MF=3
        self.assertEqual(source, "endf_mf3")
        self.assertGreater(len(energies), 0)

        # Should have two warnings: sparsity + fallback
        fallback_warnings = [x for x in w if "falling back" in str(x.message).lower()]
        self.assertGreater(len(fallback_warnings), 0,
                           "Expected a fallback warning when NNDC fails")

    def test_invalid_prefer_raises(self):
        """Invalid prefer value should raise ValueError."""
        reader = ENDFReader(CL35_FILE)
        with self.assertRaises(ValueError):
            reader.get_cross_section_best(mt=103, prefer="invalid")


class TestAutoLabelGeneration(unittest.TestCase):
    """Verify that CrossSectionFigure generates correct labels per source."""

    @unittest.skipUnless(U235_FILE.exists(), "U-235 ENDF file not available")
    def test_dense_label_says_raw(self):
        """Dense data label should include 'MF=3' and 'raw'."""
        from nucml_next.visualization.cross_section_figure import CrossSectionFigure

        fig = CrossSectionFigure('U-235', mt=18)
        fig.add_endf(U235_FILE, prefer='endf')

        # Check the generated label
        label = fig._legend_labels[-1]
        self.assertIn("MF=3", label)
        self.assertIn("raw", label.lower())
        fig.close()

    @unittest.skipUnless(CL35_FILE.exists(), "Cl-35 ENDF file not available")
    def test_nndc_label_says_reconstructed(self):
        """NNDC source label should include 'NNDC Sigma' and 'reconstructed'."""
        from nucml_next.visualization.cross_section_figure import CrossSectionFigure

        fake_energies = np.logspace(-5, 7, 5000)
        fake_xs = np.random.exponential(0.1, 5000)

        fig = CrossSectionFigure('Cl-35', mt=103)
        with patch.object(ENDFReader, '_fetch_nndc',
                          return_value=(fake_energies, fake_xs, "nndc_sigma")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.add_endf(CL35_FILE, prefer='nndc')

        label = fig._legend_labels[-1]
        self.assertIn("NNDC Sigma", label)
        self.assertIn("reconstructed", label.lower())
        fig.close()


class TestNNDCSigmaFetcher(unittest.TestCase):
    """Unit tests for NNDCSigmaFetcher cache and configuration."""

    def test_cache_path_format(self):
        """Cache path should follow expected naming convention."""
        fetcher = NNDCSigmaFetcher(library="endfb8.0")
        path = fetcher._get_cache_path(17, 35, 103)
        self.assertEqual(path.name, "Cl-35_MT103_endfb8.0.npz")

    def test_repr(self):
        """repr should include library name."""
        fetcher = NNDCSigmaFetcher(library="endfb7.1")
        r = repr(fetcher)
        self.assertIn("endfb7.1", r)

    def test_custom_cache_dir(self):
        """Custom cache_dir should be used."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)
            self.assertEqual(fetcher.cache_dir, Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
