"""
ENDF-6 Format Reader
====================

Parser for ENDF-6 formatted evaluated nuclear data files.
Supports reading cross-section data (MF=3) for any reaction type (MT),
with automatic fallback to NNDC Sigma for resonance-reconstructed data.

ENDF-6 Format Overview:
    - Fixed-width columns (11 characters per field, 6 fields per line)
    - FORTRAN scientific notation (e.g., 1.234567+5 = 1.234567e5)
    - Section identified by MAT (material), MF (file type), MT (reaction type)
    - MF=3 contains cross-section data in TAB1 format

MF=3 vs MF=2 Limitation:
    This reader only parses MF=3 (pointwise cross-section) data. It does NOT
    process MF=2 resonance parameters. For isotopes where cross-sections are
    stored as resonance parameters (Reich-Moore, R-Matrix Limited, etc.),
    the MF=3 data may be sparse or contain only above-threshold values.

    Examples:
    - U-235 fission (MT=18): Dense pointwise data in MF=3 (839 points)
    - Cl-35 (n,p) (MT=103): Sparse data, only 79 points above ~1 MeV threshold
      (resonance structure is in MF=2, requires NJOY/PREPRO to process)

Auto-Fallback (get_cross_section_best):
    The recommended API is ``get_cross_section_best(mt, prefer="auto")``.
    When MF=3 data is sparse (< 200 points), it automatically attempts to
    fetch resonance-reconstructed pointwise data from NNDC Sigma.  If NNDC
    is unavailable it falls back gracefully to raw MF=3 with a warning.

    ``CrossSectionFigure.add_endf()`` uses this logic by default so plots
    will show reconstructed resonance detail when available.

Classes:
    ENDFReader: Parser for local ENDF-6 files with auto-fallback
    NNDCSigmaFetcher: HTTP client for NNDC Sigma processed pointwise data

References:
    - ENDF-6 Formats Manual: https://www.oecd-nea.org/dbdata/data/manual-endf/
    - ENDF/B-VIII.0 Library: https://www.nndc.bnl.gov/endf-b8.0/
    - NNDC Sigma: https://www.nndc.bnl.gov/sigma/

Example:
    >>> reader = ENDFReader('data/ENDF-B/neutrons/n-092_U_235.endf')
    >>> energies, xs = reader.get_cross_section(mt=18)  # Fission
    >>> print(f"Fission XS: {len(energies)} points")
    >>>
    >>> # Auto-fallback for sparse data
    >>> reader = ENDFReader('data/ENDF-B/neutrons/n-017_Cl_035.endf')
    >>> e, xs, src = reader.get_cross_section_best(mt=103)
    >>> print(src)  # "nndc_sigma" when sparse
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import re
import warnings


class ENDFReader:
    """
    Reader for ENDF-6 formatted nuclear data files.

    Parses cross-section data (MF=3) from ENDF-6 files. Supports all
    standard reaction types (MT codes) including fission, capture,
    elastic scattering, and threshold reactions.

    Attributes:
        filepath: Path to the ENDF file
        mat: Material number (e.g., 9228 for U-235)
        z: Atomic number
        a: Mass number
        symbol: Element symbol

    Example:
        >>> reader = ENDFReader('data/ENDF-B/neutrons/n-092_U_235.endf')
        >>>
        >>> # Get fission cross section
        >>> energies, xs = reader.get_cross_section(mt=18)
        >>>
        >>> # Get capture cross section
        >>> energies, xs = reader.get_cross_section(mt=102)
        >>>
        >>> # Get all available reaction types
        >>> available_mt = reader.list_reactions()
        >>> print(f"Available MTs: {available_mt}")
    """

    # Common MT code descriptions
    MT_DESCRIPTIONS = {
        1: 'Total',
        2: 'Elastic',
        4: 'Inelastic (total)',
        5: '(n,x) - anything',
        16: '(n,2n)',
        17: '(n,3n)',
        18: 'Fission (total)',
        19: 'First-chance fission',
        20: 'Second-chance fission',
        21: 'Third-chance fission',
        37: '(n,4n)',
        38: 'Fourth-chance fission',
        51: 'Inelastic (1st level)',
        91: 'Inelastic (continuum)',
        102: '(n,gamma) Capture',
        103: '(n,p)',
        104: '(n,d)',
        105: '(n,t)',
        106: '(n,3He)',
        107: '(n,alpha)',
        # Add more as needed
    }

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize ENDF reader.

        Args:
            filepath: Path to ENDF-6 formatted file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid ENDF format
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"ENDF file not found: {self.filepath}")

        # Parse header to get material info
        self._parse_header()

        # Cache for parsed cross-sections
        self._xs_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _parse_header(self) -> None:
        """Parse ENDF file header to extract material information."""
        with open(self.filepath, 'r') as f:
            # Read first few lines to get header info
            lines = [f.readline() for _ in range(10)]

        # Parse first data line (line 2) for ZA and AWR
        # Format: ZA (1000*Z + A) and AWR (atomic weight ratio)
        line = lines[1]
        za_str = line[0:11].strip()
        za = self._parse_endf_float(za_str)

        self.z = int(za // 1000)
        self.a = int(za % 1000)

        # Get MAT number from line identifier (columns 66-70)
        self.mat = int(lines[1][66:70])

        # Element symbol from periodic table
        self.symbol = self._z_to_symbol(self.z)

    @staticmethod
    def _z_to_symbol(z: int) -> str:
        """Convert atomic number to element symbol."""
        symbols = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
            37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
            44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
            51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
            58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
            65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
            72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
            79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
            86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
            93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
        }
        return symbols.get(z, f'Z{z}')

    @staticmethod
    def _parse_endf_float(s: str) -> float:
        """
        Parse ENDF FORTRAN scientific notation.

        ENDF uses compact format: 1.234567+5 instead of 1.234567e+5

        Args:
            s: String in ENDF format

        Returns:
            Parsed float value

        Examples:
            >>> ENDFReader._parse_endf_float('1.234567+5')
            123456.7
            >>> ENDFReader._parse_endf_float('-2.345678-3')
            -0.002345678
        """
        s = s.strip()
        if not s or s == '':
            return 0.0

        # Replace ENDF notation with standard notation
        # Handle patterns like: 1.234567+5, 1.234567-5, -1.234567+5
        # First handle the case where there's no explicit sign before exponent
        s = re.sub(r'(\d)([+-])(\d)', r'\1e\2\3', s)

        try:
            return float(s)
        except ValueError:
            return 0.0

    def _find_section(self, mf: int, mt: int) -> Optional[int]:
        """
        Find the line number where MF/MT section starts.

        Args:
            mf: File type (e.g., 3 for cross-sections)
            mt: Reaction type

        Returns:
            Line number (0-indexed) or None if not found
        """
        target = f"{self.mat:4d}{mf:2d}{mt:3d}"

        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                if len(line) >= 75:
                    # Check columns 66-75 for MAT, MF, MT
                    line_id = line[66:75].replace(' ', '')
                    if line_id.startswith(target.replace(' ', '')):
                        return i
        return None

    def list_reactions(self) -> List[int]:
        """
        List all available reaction types (MT codes) in MF=3.

        Returns:
            List of MT codes with cross-section data

        Example:
            >>> reader = ENDFReader('n-092_U_235.endf')
            >>> mts = reader.list_reactions()
            >>> print(mts)  # [1, 2, 4, 16, 17, 18, 102, ...]
        """
        available = []
        mf = 3  # Cross-section file

        with open(self.filepath, 'r') as f:
            for line in f:
                if len(line) >= 75:
                    try:
                        line_mf = int(line[70:72])
                        line_mt = int(line[72:75])
                        if line_mf == mf and line_mt > 0 and line_mt not in available:
                            available.append(line_mt)
                    except ValueError:
                        continue

        return sorted(available)

    def get_cross_section(
        self,
        mt: int,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cross-section data for a specific reaction.

        Args:
            mt: Reaction type (e.g., 18 for fission, 102 for capture)
            energy_min: Minimum energy in eV (optional filter)
            energy_max: Maximum energy in eV (optional filter)

        Returns:
            Tuple of (energies, cross_sections) as numpy arrays
            - energies: Energy points in eV
            - cross_sections: Cross-section values in barns

        Raises:
            ValueError: If MT reaction not found in file

        Example:
            >>> reader = ENDFReader('n-092_U_235.endf')
            >>> energies, xs = reader.get_cross_section(mt=18)  # Fission
            >>> print(f"Energy range: {energies[0]:.2e} - {energies[-1]:.2e} eV")
            >>> print(f"XS range: {xs.min():.4f} - {xs.max():.4f} barns")
        """
        # Check cache
        if mt in self._xs_cache:
            energies, xs = self._xs_cache[mt]
        else:
            # Parse from file
            energies, xs = self._parse_mf3_section(mt)
            self._xs_cache[mt] = (energies, xs)

        # Apply energy filters
        if energy_min is not None or energy_max is not None:
            mask = np.ones(len(energies), dtype=bool)
            if energy_min is not None:
                mask &= energies >= energy_min
            if energy_max is not None:
                mask &= energies <= energy_max
            energies = energies[mask]
            xs = xs[mask]

        return energies, xs

    def _parse_mf3_section(self, mt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse MF=3 (cross-section) data for given MT.

        MF=3 uses TAB1 record format:
        - HEAD record: ZA, AWR, 0, 0, 0, 0
        - TAB1 record: C1, C2, L1, L2, NR, NP
          - NR = number of interpolation regions
          - NP = number of energy-XS points
        - Interpolation table (if NR > 0)
        - Energy-XS pairs (6 values per line)
        """
        start_line = self._find_section(3, mt)
        if start_line is None:
            raise ValueError(
                f"MT={mt} not found in {self.filepath.name}. "
                f"Available MTs: {self.list_reactions()}"
            )

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Skip HEAD record (first line of section)
        line_idx = start_line + 1

        # Parse TAB1 header
        # Line format: C1, C2, L1, L2, NR, NP (then MAT MF MT)
        header_line = lines[line_idx]
        nr = int(self._parse_endf_float(header_line[44:55]))  # Number of interpolation regions
        np_points = int(self._parse_endf_float(header_line[55:66]))  # Number of points
        line_idx += 1

        # Skip interpolation table (NR regions, 2 values each = NR pairs)
        # These are packed 3 pairs per line
        if nr > 0:
            interp_lines = (nr * 2 + 5) // 6  # Number of lines for interpolation table
            line_idx += interp_lines

        # Parse energy-XS pairs
        # 6 values per line: E1, XS1, E2, XS2, E3, XS3
        energies = []
        cross_sections = []

        values_needed = np_points * 2
        values_read = 0

        while values_read < values_needed:
            line = lines[line_idx]

            # Parse 6 values from the line (11 characters each)
            for i in range(6):
                if values_read >= values_needed:
                    break
                start = i * 11
                end = start + 11
                val = self._parse_endf_float(line[start:end])

                if values_read % 2 == 0:
                    energies.append(val)
                else:
                    cross_sections.append(val)
                values_read += 1

            line_idx += 1

        return np.array(energies), np.array(cross_sections)

    def get_mt_description(self, mt: int) -> str:
        """
        Get human-readable description of MT code.

        Args:
            mt: Reaction type number

        Returns:
            Description string
        """
        return self.MT_DESCRIPTIONS.get(mt, f'MT-{mt}')

    def get_data_info(self, mt: int) -> Dict[str, any]:
        """
        Get information about the cross-section data for an MT code.

        Useful for understanding data quality and limitations.
        Sparse data (< 200 points) often indicates resonance-parameter-based
        evaluations where detailed structure requires NJOY processing.

        Args:
            mt: Reaction type number

        Returns:
            Dictionary with:
            - n_points: Number of data points
            - energy_range: (min, max) energy in eV
            - xs_range: (min, max) cross-section in barns
            - is_sparse: True if < 200 points (may lack resonance detail)
            - has_placeholders: True if data contains very small values (<1e-15)
              indicating below-threshold placeholders

        Example:
            >>> reader = ENDFReader('n-017_Cl_035.endf')
            >>> info = reader.get_data_info(mt=103)
            >>> if info['is_sparse']:
            ...     print("Warning: Data may lack resonance detail")
        """
        energies, xs = self.get_cross_section(mt)

        # Check for placeholder zeros (ENDF uses ~1e-20 for "zero")
        has_placeholders = (xs < 1e-15).any()

        return {
            'n_points': len(energies),
            'energy_range': (float(energies.min()), float(energies.max())),
            'xs_range': (float(xs.min()), float(xs.max())),
            'is_sparse': len(energies) < 200,
            'has_placeholders': bool(has_placeholders),
        }

    def get_cross_section_best(
        self,
        mt: int,
        prefer: str = "auto",
        z: Optional[int] = None,
        a: Optional[int] = None,
        library: str = "endfb8.0",
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get the best available cross-section data, with auto-fallback.

        In ``"auto"`` mode the method inspects the local MF=3 data via
        :meth:`get_data_info`.  If the data is *sparse* (< 200 points)
        **and** the non-placeholder region covers fewer than 200 physical
        points, the method attempts to fetch resonance-reconstructed
        pointwise data from NNDC Sigma instead.

        Args:
            mt: Reaction type (MT code).
            prefer: Data source preference:

                * ``"auto"`` – use NNDC when MF=3 is sparse (default)
                * ``"endf"`` – always use local MF=3
                * ``"nndc"`` – always fetch from NNDC Sigma
            z: Atomic number (defaults to file header).
            a: Mass number (defaults to file header).
            library: NNDC library id (``"endfb8.0"``, ``"endfb7.1"``, …).
            energy_min: Optional low-energy filter (eV).
            energy_max: Optional high-energy filter (eV).

        Returns:
            ``(energies, cross_sections, source)`` where *source* is one of
            ``"endf_mf3"`` or ``"nndc_sigma"``.

        Example:
            >>> reader = ENDFReader('n-017_Cl_035.endf')
            >>> e, xs, src = reader.get_cross_section_best(mt=103)
            >>> print(src)  # "nndc_sigma" (sparse MF=3 triggers NNDC)
        """
        if prefer not in ("auto", "endf", "nndc"):
            raise ValueError(f"prefer must be 'auto', 'endf', or 'nndc', got {prefer!r}")

        z = z or self.z
        a = a or self.a

        # ---- forced ENDF path ----
        if prefer == "endf":
            energies, xs = self.get_cross_section(mt, energy_min, energy_max)
            return energies, xs, "endf_mf3"

        # ---- forced NNDC path ----
        if prefer == "nndc":
            return self._fetch_nndc(z, a, mt, library, energy_min, energy_max)

        # ---- auto mode: decide from data quality ----
        info = self.get_data_info(mt)
        needs_nndc = info["is_sparse"]

        if not needs_nndc:
            # Dense data – use MF=3 directly
            energies, xs = self.get_cross_section(mt, energy_min, energy_max)
            return energies, xs, "endf_mf3"

        # Sparse – try NNDC, fall back to MF=3 on error
        warnings.warn(
            f"ENDF MF=3 data for MT={mt} is sparse ({info['n_points']} points). "
            f"Attempting to fetch resonance-reconstructed pointwise data from NNDC Sigma.",
            UserWarning,
        )
        try:
            return self._fetch_nndc(z, a, mt, library, energy_min, energy_max)
        except Exception as exc:
            warnings.warn(
                f"NNDC Sigma fetch failed: {exc}\n"
                f"Falling back to raw ENDF MF=3 data. The plot may omit resonance structure.",
                UserWarning,
            )
            energies, xs = self.get_cross_section(mt, energy_min, energy_max)
            return energies, xs, "endf_mf3"

    # ------------------------------------------------------------------
    # internal helper used by get_cross_section_best
    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_nndc(
        z: int, a: int, mt: int, library: str,
        energy_min: Optional[float], energy_max: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Fetch from NNDCSigmaFetcher and return the 3-tuple."""
        fetcher = NNDCSigmaFetcher(library=library)
        energies, xs = fetcher.get_cross_section(z, a, mt, energy_min, energy_max)
        return energies, xs, "nndc_sigma"

    def __repr__(self) -> str:
        """String representation."""
        return f"ENDFReader({self.symbol}-{self.a}, MAT={self.mat})"

    @classmethod
    def find_file(
        cls,
        z: int,
        a: int,
        endf_dir: Union[str, Path] = 'data/ENDF-B/neutrons',
    ) -> Optional[Path]:
        """
        Find ENDF file for a given isotope.

        Args:
            z: Atomic number
            a: Mass number
            endf_dir: Directory containing ENDF files

        Returns:
            Path to ENDF file or None if not found

        Example:
            >>> path = ENDFReader.find_file(z=92, a=235)
            >>> print(path)  # data/ENDF-B/neutrons/n-092_U_235.endf
        """
        endf_dir = Path(endf_dir)
        if not endf_dir.exists():
            return None

        # Try standard naming pattern: n-ZZZ_Sym_AAA.endf
        symbol = cls._z_to_symbol(z)
        pattern = f"n-{z:03d}_{symbol}_{a:03d}*.endf"

        matches = list(endf_dir.glob(pattern))
        if matches:
            return matches[0]

        # Try alternative patterns
        for pattern in [f"n-{z:03d}*{a:03d}.endf", f"*{symbol}*{a}*.endf"]:
            matches = list(endf_dir.glob(pattern))
            if matches:
                return matches[0]

        return None


class NNDCSigmaFetcher:
    """
    Fetch processed pointwise cross-section data from NNDC Sigma.

    NNDC Sigma (https://www.nndc.bnl.gov/sigma/) provides processed pointwise
    nuclear cross-section data with full resonance reconstruction. Unlike raw
    ENDF files which may store cross-sections as resonance parameters (MF=2),
    NNDC Sigma returns data that has been processed through NJOY/PREPRO to
    reconstruct the full energy-dependent cross-section.

    This is particularly useful for:
    - Threshold reactions like (n,p), (n,α) that have resonance structure
    - Capturing full resonance detail in the thermal and epithermal region
    - Getting processed data without installing NJOY locally

    The data is fetched via HTTP from the NNDC web service and cached locally
    to avoid repeated downloads.

    Attributes:
        cache_dir: Directory for caching downloaded data
        temperature: Temperature for Doppler broadening (default 0K = 0K cold)

    Example:
        >>> from nucml_next.visualization import NNDCSigmaFetcher
        >>>
        >>> fetcher = NNDCSigmaFetcher()
        >>>
        >>> # Fetch Cl-35 (n,p) with full resonance detail
        >>> energies, xs = fetcher.get_cross_section(z=17, a=35, mt=103)
        >>> print(f"Got {len(energies)} points with resonance structure")
        >>>
        >>> # Clear cache if needed
        >>> fetcher.clear_cache()
    """

    # NNDC Sigma base URL for pointwise data retrieval
    BASE_URL = "https://www.nndc.bnl.gov/sigma/getPlotData.jsp"

    # Element symbols (same as ENDFReader)
    SYMBOLS = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
        93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
    }

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        library: str = "endfb8.0",
    ):
        """
        Initialize NNDC Sigma fetcher.

        Args:
            cache_dir: Directory for caching downloaded data.
                       Default: ~/.nucml_next/nndc_cache
            library: Nuclear data library to use. Options:
                     - "endfb8.0" (default): ENDF/B-VIII.0
                     - "endfb7.1": ENDF/B-VII.1
                     - "jendl5.0": JENDL-5.0
                     - "jeff3.3": JEFF-3.3
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".nucml_next" / "nndc_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.library = library

        # Map library names to NNDC library codes
        self._library_codes = {
            "endfb8.0": "endfb8.0",
            "endfb7.1": "endfb7.1",
            "jendl5.0": "jendl5.0",
            "jeff3.3": "jeff3.3",
        }

    def _get_cache_path(self, z: int, a: int, mt: int) -> Path:
        """Get cache file path for given isotope and reaction."""
        symbol = self.SYMBOLS.get(z, f"Z{z}")
        return self.cache_dir / f"{symbol}-{a}_MT{mt}_{self.library}.npz"

    def get_cross_section(
        self,
        z: int,
        a: int,
        mt: int,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get processed pointwise cross-section data from NNDC Sigma.

        Fetches data from NNDC's web service, which provides cross-sections
        that have been processed through NJOY to reconstruct resonance
        parameters into pointwise data.

        Args:
            z: Atomic number (e.g., 17 for Cl)
            a: Mass number (e.g., 35 for Cl-35)
            mt: Reaction type (MT code, e.g., 103 for (n,p))
            energy_min: Minimum energy filter in eV (optional)
            energy_max: Maximum energy filter in eV (optional)
            use_cache: Use cached data if available (default True)

        Returns:
            Tuple of (energies, cross_sections) as numpy arrays
            - energies: Energy points in eV
            - cross_sections: Cross-section values in barns

        Raises:
            RuntimeError: If data cannot be fetched from NNDC
            ValueError: If the requested data is not available

        Example:
            >>> fetcher = NNDCSigmaFetcher()
            >>> energies, xs = fetcher.get_cross_section(z=17, a=35, mt=103)
            >>> print(f"Cl-35 (n,p): {len(energies)} points")
        """
        cache_path = self._get_cache_path(z, a, mt)

        # Check cache
        if use_cache and cache_path.exists():
            data = np.load(cache_path)
            energies = data['energies']
            xs = data['xs']
        else:
            # Fetch from NNDC
            energies, xs = self._fetch_from_nndc(z, a, mt)

            # Save to cache
            np.savez_compressed(cache_path, energies=energies, xs=xs)

        # Apply energy filters
        if energy_min is not None or energy_max is not None:
            mask = np.ones(len(energies), dtype=bool)
            if energy_min is not None:
                mask &= energies >= energy_min
            if energy_max is not None:
                mask &= energies <= energy_max
            energies = energies[mask]
            xs = xs[mask]

        return energies, xs

    def _fetch_from_nndc(self, z: int, a: int, mt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch cross-section data from NNDC Sigma web service.

        Uses the NNDC getPlotData.jsp endpoint to retrieve processed
        pointwise cross-section data.  Retries up to 3 times with
        exponential back-off (90 s timeout per attempt).
        """
        import time
        try:
            import urllib.request
            import urllib.parse
        except ImportError:
            raise RuntimeError("urllib is required for NNDC data fetching")

        symbol = self.SYMBOLS.get(z, f"Z{z}")

        # Construct the URL for NNDC Sigma
        # The format is: getPlotData.jsp?lib=<library>&target=<ZAM>&mt=<MT>&nPts=<points>
        # ZAM format: Z*1000 + A (e.g., Cl-35 = 17035)
        zam = z * 1000 + a

        # NNDC uses specific request format
        params = {
            'lib': self.library,
            'ZAM': str(zam),
            'MT': str(mt),
            'nPts': '10000',  # Request high resolution
            'fmt': '2',  # Column format: energy(eV), xs(b)
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        max_retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    content = response.read().decode('utf-8')
                break  # success
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    warnings.warn(
                        f"NNDC fetch attempt {attempt + 1}/{max_retries} failed "
                        f"for {symbol}-{a} MT={mt}: {e}. Retrying in {wait}s...",
                        UserWarning,
                    )
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"Failed to fetch data from NNDC Sigma for {symbol}-{a} MT={mt} "
                f"after {max_retries} attempts: {last_error}\n"
                f"URL: {url}\n"
                f"You may need to check your internet connection or try the NNDC website directly:\n"
                f"https://www.nndc.bnl.gov/sigma/"
            )

        # Parse the response
        energies = []
        xs_values = []

        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    energy = float(parts[0])
                    xs = float(parts[1])
                    if energy > 0 and xs >= 0:
                        energies.append(energy)
                        xs_values.append(xs)
            except ValueError:
                continue

        if len(energies) == 0:
            raise ValueError(
                f"No data returned from NNDC Sigma for {symbol}-{a} MT={mt}.\n"
                f"The reaction may not be available in the {self.library} library.\n"
                f"Try checking the NNDC website: https://www.nndc.bnl.gov/sigma/"
            )

        return np.array(energies), np.array(xs_values)

    def clear_cache(self, z: Optional[int] = None, a: Optional[int] = None) -> int:
        """
        Clear cached NNDC data.

        Args:
            z: Atomic number (optional, clears all if not specified)
            a: Mass number (optional, clears all if not specified)

        Returns:
            Number of cache files removed

        Example:
            >>> fetcher = NNDCSigmaFetcher()
            >>> fetcher.clear_cache()  # Clear all
            >>> fetcher.clear_cache(z=17, a=35)  # Clear only Cl-35
        """
        count = 0
        pattern = "*.npz"

        if z is not None and a is not None:
            symbol = self.SYMBOLS.get(z, f"Z{z}")
            pattern = f"{symbol}-{a}_*.npz"
        elif z is not None:
            symbol = self.SYMBOLS.get(z, f"Z{z}")
            pattern = f"{symbol}-*_*.npz"

        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            count += 1

        return count

    def list_cached(self) -> List[str]:
        """
        List all cached datasets.

        Returns:
            List of cached dataset identifiers (e.g., "Cl-35_MT103_endfb8.0")
        """
        return [f.stem for f in self.cache_dir.glob("*.npz")]

    def __repr__(self) -> str:
        """String representation."""
        return f"NNDCSigmaFetcher(library={self.library}, cache={self.cache_dir})"
