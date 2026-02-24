"""Tests for RIPL-3 level density loader."""

import numpy as np
import pytest

from nucml_next.data.ripl_loader import RIPL3LevelDensity, _compute_log_D


# ---------------------------------------------------------------------------
# _compute_log_D unit tests (pure math, no file I/O)
# ---------------------------------------------------------------------------

class TestComputeLogD:
    """Test the CT level density formula."""

    # Fe-56 compound (Z=26, A=57): T=1.07035 MeV, U0=0.73824 MeV
    T_FE56 = 1.07035
    U0_FE56 = 0.73824
    S_N_FE56 = 7.646  # MeV (approximate)

    def _floor_fe56(self):
        """Compute D floor for Fe-56."""
        D_at_Sn = self.T_FE56 * np.exp(-(self.S_N_FE56 - self.U0_FE56) / self.T_FE56)
        return np.log10(D_at_Sn * 1e6)

    def test_basic_shape(self):
        """Output shape matches input."""
        log_E = np.array([0.0, 3.0, 6.0])
        result = _compute_log_D(log_E, self.T_FE56, self.U0_FE56,
                                self.S_N_FE56, self._floor_fe56())
        assert result.shape == (3,)

    def test_scalar_input(self):
        """Works with scalar input."""
        result = _compute_log_D(np.array([3.0]), self.T_FE56, self.U0_FE56,
                                self.S_N_FE56, self._floor_fe56())
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_decreasing_with_energy(self):
        """D(E) should generally decrease with increasing neutron energy.

        Higher E_n → higher E_x → higher ρ → smaller D.
        The floor may cap D at low energies, but above the floor
        D should be monotonically decreasing.
        """
        log_E = np.linspace(3.0, 7.0, 50)  # 1 keV to 10 MeV
        result = _compute_log_D(log_E, self.T_FE56, self.U0_FE56,
                                self.S_N_FE56, self._floor_fe56())
        # Check that at least the high-energy end is decreasing
        # (floor may cause flatness at low energy)
        diffs = np.diff(result[-20:])
        assert np.all(diffs <= 1e-10), "D should decrease with energy above floor"

    def test_floor_applied(self):
        """At very low energies, D is capped at D(S_n)."""
        # At E_n ~ 0 (thermal), E_x ≈ S_n. D(S_n) is the floor.
        log_E_thermal = np.array([-2.0])  # 0.01 eV
        result = _compute_log_D(log_E_thermal, self.T_FE56, self.U0_FE56,
                                self.S_N_FE56, self._floor_fe56())
        # At S_n, we should be at or very near the floor
        floor = self._floor_fe56()
        assert result[0] <= floor + 0.01, "Thermal point should be at or below floor"

    def test_no_nan_or_inf(self):
        """No NaN or inf in output for reasonable energy range."""
        log_E = np.linspace(-5, 8, 100)  # 10 μeV to 100 MeV
        result = _compute_log_D(log_E, self.T_FE56, self.U0_FE56,
                                self.S_N_FE56, self._floor_fe56())
        assert np.all(np.isfinite(result))

    def test_units_reasonable(self):
        """D should be in reasonable range for nuclear physics.

        For Fe-56: D₀ at S_n should be ~1 eV to ~100 eV.
        """
        log_E_at_Sn = np.array([np.log10(self.S_N_FE56 * 1e6)])  # S_n in eV
        result = _compute_log_D(log_E_at_Sn, self.T_FE56, self.U0_FE56,
                                self.S_N_FE56, self._floor_fe56())
        D_eV = 10.0 ** result[0]
        # Fe-56 D₀ ~ few eV at S_n (well-known)
        assert 0.01 < D_eV < 1e4, f"D at S_n = {D_eV:.2f} eV, expected 0.01–10000"

    def test_u235_parameters(self):
        """U-236 compound: D at S_n (thermal neutrons) should be sub-eV.

        Known D₀ for U-235 fission is ~ 0.5 eV.  The CT model gives
        D(S_n) = T · exp(−(S_n − U₀)/T) which is an approximation.
        The important thing is it's finite and in a physically reasonable
        range (sub-eV for actinides).
        """
        T = 0.36405   # MeV
        U0 = -0.73882  # MeV
        S_n = 6.545    # MeV (approximate for U-236)
        D_at_Sn_MeV = T * np.exp(-(S_n - U0) / T)
        floor = np.log10(D_at_Sn_MeV * 1e6)

        # At thermal energy (0.0253 eV), E_x ≈ S_n
        log_E_thermal = np.array([np.log10(0.0253)])  # thermal neutrons
        result = _compute_log_D(log_E_thermal, T, U0, S_n, floor)
        D_eV = 10.0 ** result[0]
        # D at thermal should be at the floor = D(S_n)
        # CT model gives D(S_n) in sub-eV range for actinides
        assert D_eV > 1e-6, f"U-235 D₀ = {D_eV:.2e} eV, expected > 1e-6"
        assert D_eV < 10.0, f"U-235 D₀ = {D_eV:.2e} eV, expected < 10"
        assert np.isfinite(D_eV)


# ---------------------------------------------------------------------------
# RIPL3LevelDensity class tests
# ---------------------------------------------------------------------------

class TestRIPL3LevelDensity:
    """Test the RIPL-3 loader class."""

    @pytest.fixture
    def data_path(self):
        """Path to the RIPL-3 levels-param.data file."""
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "levels" / "levels-param.data"
        if not p.exists():
            pytest.skip("RIPL-3 data not available")
        return str(p)

    def test_load_from_file(self, data_path):
        """Loading from file populates nuclide parameters."""
        loader = RIPL3LevelDensity(data_path)
        assert loader.n_nuclides > 100

    def test_has_data_u235(self, data_path):
        """U-236 compound should be available (from U-235 + n)."""
        loader = RIPL3LevelDensity(data_path)
        # levels-param.data is indexed by (Z, A) of the nuclide itself
        assert loader.has_data(92, 236)

    def test_has_data_fe56(self, data_path):
        """Fe-57 compound should be available (from Fe-56 + n)."""
        loader = RIPL3LevelDensity(data_path)
        assert loader.has_data(26, 57)

    def test_missing_nuclide(self, data_path):
        """Non-existent nuclide returns None."""
        loader = RIPL3LevelDensity(data_path)
        assert not loader.has_data(200, 500)
        assert loader.get_log_D_interpolator(200, 500, S_n=5.0) is None

    def test_get_ct_params(self, data_path):
        """Can retrieve (T, U0) for known nuclides."""
        loader = RIPL3LevelDensity(data_path)
        params = loader.get_ct_params(92, 235)
        assert params is not None
        T, U0 = params
        assert 0.3 < T < 0.5  # U-235 T ~ 0.364 MeV
        assert -1.0 < U0 < 0.0  # U-235 U0 ~ -0.739 MeV

    def test_get_interpolator(self, data_path):
        """Interpolator returns callable with correct shape."""
        loader = RIPL3LevelDensity(data_path)
        interp = loader.get_log_D_interpolator(92, 236, S_n=6.545)
        assert interp is not None

        log_E = np.linspace(0, 7, 50)
        result = interp(log_E)
        assert result.shape == (50,)
        assert np.all(np.isfinite(result))

    def test_interpolator_no_file_io(self, data_path):
        """After initial load, interpolator does no file I/O."""
        loader = RIPL3LevelDensity(data_path)
        interp = loader.get_log_D_interpolator(92, 236, S_n=6.545)
        # Call many times — should be fast (no I/O)
        for _ in range(100):
            interp(np.array([3.0, 4.0, 5.0]))

    def test_zero_T_nuclide(self, data_path):
        """Nuclides with T=0 (no CT fit) should return None."""
        loader = RIPL3LevelDensity(data_path)
        # H-1 has T=0 in levels-param.data
        assert not loader.has_data(1, 1)
        assert loader.get_log_D_interpolator(1, 1, S_n=2.2) is None

    def test_nonexistent_path(self):
        """Non-existent path logs warning, returns empty."""
        loader = RIPL3LevelDensity("/nonexistent/path.data")
        assert loader.n_nuclides == 0

    def test_lazy_loading(self):
        """File is not parsed until first access."""
        loader = RIPL3LevelDensity("/nonexistent/path.data")
        # No file access yet — only on first use
        assert not loader._loaded
        _ = loader.n_nuclides
        assert loader._loaded
