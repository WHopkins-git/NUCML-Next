"""
RIPL-3 Level Density Loader
============================

Loads Constant Temperature (CT) level density parameters from RIPL-3
``levels-param.data`` and provides callable interpolators for the mean
level spacing D(E) per nuclide.

**Purpose:** The Gibbs kernel needs an energy-dependent lengthscale
ℓ(E) that tracks the mean level spacing D(E) ≈ 1/ρ(E).  RIPL-3
provides nuclear-structure-derived CT parameters (T, U₀) per nuclide,
from which D(E) is computed analytically.

**Evaluation independence:** RIPL-3 data comes from nuclear structure
(discrete level schemes, neutron resonances), NOT from cross-section
evaluations (ENDF/B, JEFF, JENDL).  It is safe to use in a tool that
provides independent QA to evaluators.

CT Level Density Formula
------------------------
    ρ(E_x) = (1/T) · exp((E_x − U₀) / T)
    D(E_x) = 1/ρ(E_x) = T · exp(−(E_x − U₀) / T)

where E_x is excitation energy.  For neutron-induced reactions on
target (Z, A), the compound nucleus (Z, A+1) has excitation energy:
    E_x = S_n + E_n

where S_n is the neutron separation energy and E_n is the incident
neutron energy.

Data Source
-----------
    ``data/levels/levels-param.data`` from IAEA RIPL-3:
    https://www-nds.iaea.org/RIPL-3/levels/

Key Classes:
    RIPL3LevelDensity: Loader with per-nuclide D(E) interpolators.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Neutron mass in AMU (for S_n fallback)
_NEUTRON_MASS_MEV = 939.565346  # MeV/c²


class RIPL3LevelDensity:
    """Loader for RIPL-3 Constant Temperature level density parameters.

    Parses ``levels-param.data`` once and caches all (Z, A) → (T, U₀)
    pairs.  Provides callable interpolators that compute log₁₀ D(E) for
    any incident neutron energy, given the neutron separation energy S_n.

    Parameters
    ----------
    data_path : str or None
        Path to ``levels-param.data``.  If None, tries the default
        location ``data/levels/levels-param.data`` relative to the
        package root.

    Examples
    --------
    >>> loader = RIPL3LevelDensity("data/levels/levels-param.data")
    >>> interp = loader.get_log_D_interpolator(92, 236, S_n=6.545)
    >>> log_E = np.array([0.0, 3.0, 6.0])  # log10(eV)
    >>> log_D = interp(log_E)  # log10(D in eV)
    """

    def __init__(self, data_path: Optional[str] = None):
        self._params: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._data_path = data_path
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Parse levels-param.data on first use (lazy loading)."""
        if self._loaded:
            return

        path = self._resolve_path()
        if path is None:
            logger.warning("RIPL-3 levels-param.data not found; "
                           "Gibbs kernel will fall back to RBF")
            self._loaded = True
            return

        self._parse(path)
        self._loaded = True

    def _resolve_path(self) -> Optional[Path]:
        """Find the levels-param.data file."""
        if self._data_path is not None:
            p = Path(self._data_path)
            if p.exists():
                return p
            logger.warning(f"RIPL-3 data path not found: {p}")
            return None

        # Try default locations relative to package
        candidates = [
            Path(__file__).parent.parent.parent / "data" / "levels" / "levels-param.data",
            Path("data") / "levels" / "levels-param.data",
        ]
        for c in candidates:
            if c.exists():
                return c

        return None

    def _parse(self, path: Path) -> None:
        """Parse levels-param.data into {(Z, A): (T, U0)} dict.

        File format (fixed-width, space-separated after comment lines):
            #  Z   A El     T      dT      U0     dU0  Nlev Nmax ...
            1   1 H    0.00000   0.00000   0.00000   0.00000   1   1 ...
        """
        count = 0
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                try:
                    Z = int(parts[0])
                    A = int(parts[1])
                    # parts[2] = element symbol
                    T = float(parts[3])
                    # parts[4] = dT (uncertainty, not used)
                    U0 = float(parts[5])
                    # parts[6] = dU0 (uncertainty, not used)

                    # T = 0 means no CT fit was possible (too few levels)
                    if T > 0:
                        self._params[(Z, A)] = (T, U0)
                        count += 1
                except (ValueError, IndexError):
                    continue

        logger.info(f"RIPL-3: loaded CT parameters for {count} nuclides "
                    f"from {path}")

    def has_data(self, Z: int, A: int) -> bool:
        """Check if CT parameters are available for this nuclide."""
        self._ensure_loaded()
        return (Z, A) in self._params

    def get_ct_params(self, Z: int, A: int) -> Optional[Tuple[float, float]]:
        """Return (T, U0) in MeV for given nuclide, or None."""
        self._ensure_loaded()
        return self._params.get((Z, A))

    def get_log_D_interpolator(
        self,
        Z: int,
        A: int,
        S_n: float,
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Return callable: log₁₀(E_n [eV]) → log₁₀(D [eV]).

        The returned function computes D(E) analytically from the CT
        formula for the compound nucleus (Z, A).  It is a closure over
        (T, U₀, S_n) and does no file I/O.

        Parameters
        ----------
        Z : int
            Proton number of the **compound** nucleus (target Z for
            neutron capture, or target Z for fission — depending on
            the specific reaction).
        A : int
            Mass number of the **compound** nucleus.
        S_n : float
            Neutron separation energy in MeV.

        Returns
        -------
        callable or None
            ``f(log_E) -> log_D`` where log_E and log_D are in
            log₁₀(eV).  Returns None if no CT parameters for (Z, A).
        """
        self._ensure_loaded()
        params = self._params.get((Z, A))
        if params is None:
            return None

        T, U0 = params

        # Precompute the floor: D at S_n (the minimum physical D).
        # This is the mean level spacing at the neutron separation
        # energy, the densest physically meaningful point.
        D_at_Sn_MeV = T * np.exp(-(S_n - U0) / T)
        log10_D_floor_eV = np.log10(D_at_Sn_MeV * 1e6)

        def _log_D(log_E: np.ndarray) -> np.ndarray:
            return _compute_log_D(log_E, T, U0, S_n, log10_D_floor_eV)

        return _log_D

    @property
    def n_nuclides(self) -> int:
        """Number of nuclides with valid CT parameters."""
        self._ensure_loaded()
        return len(self._params)


def _compute_log_D(
    log_E: np.ndarray,
    T: float,
    U0: float,
    S_n: float,
    log10_D_floor_eV: float,
) -> np.ndarray:
    """Vectorised: log₁₀(E_n [eV]) → log₁₀(D(E_x) [eV]).

    Parameters
    ----------
    log_E : ndarray
        log₁₀ of incident neutron energy in eV.
    T : float
        CT nuclear temperature in MeV.
    U0 : float
        CT back-shift energy in MeV.
    S_n : float
        Neutron separation energy in MeV.
    log10_D_floor_eV : float
        Floor value = log₁₀(D(S_n) in eV).  The CT formula is only
        valid above the discrete level region.  At very low E_x the
        exponential blows up giving nonsensically large D.  When
        S_n + E_n < U0, D diverges (unphysical).  The floor ensures
        ℓ never exceeds the thermal-region value.

    Returns
    -------
    ndarray
        log₁₀(D [eV]) at each energy point.
    """
    log_E = np.asarray(log_E, dtype=float)

    # Convert incident neutron energy from eV to MeV
    E_n_MeV = 10.0 ** log_E * 1e-6  # eV → MeV

    # Excitation energy of compound nucleus
    E_x = S_n + E_n_MeV  # MeV

    # CT formula: D(E_x) = T · exp(−(E_x − U0) / T)  [MeV]
    D_MeV = T * np.exp(-(E_x - U0) / T)

    # Convert to eV: D [eV] = D [MeV] × 10⁶
    D_eV = D_MeV * 1e6

    # log₁₀(D [eV])
    # Use np.clip to avoid log(0) or log(negative) from numerical issues
    log10_D = np.log10(np.maximum(D_eV, 1e-30))

    # Apply floor: D should not exceed D(S_n) — the thermal/1v region
    # is the smoothest, so ℓ there should be largest but not infinite.
    log10_D = np.minimum(log10_D, log10_D_floor_eV)

    return log10_D
