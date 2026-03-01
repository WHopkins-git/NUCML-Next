"""
Cross-Section Figure Class
==========================

Flexible, publication-quality figure class for nuclear cross-section visualization.
Supports multiple data sources: EXFOR, ENDF-B, ML models, and custom literature data.

Features:
    - Automatic log-log scaling for nuclear data
    - Multiple data series with customizable styles
    - EXFOR experimental data with uncertainty bands
    - ENDF-B evaluated data overlay
    - ML model predictions comparison
    - Custom literature data points
    - Publication-ready export (PNG, PDF, SVG)

Example:
    >>> from nucml_next.visualization import CrossSectionFigure
    >>>
    >>> # Quick plot with EXFOR and ENDF
    >>> fig = CrossSectionFigure(isotope='U-235', mt=18, title='U-235 Fission')
    >>> fig.add_exfor(exfor_df)
    >>> fig.add_endf('data/ENDF-B/neutrons/n-092_U_235.endf')
    >>> fig.show()
    >>>
    >>> # Compare multiple models
    >>> fig = CrossSectionFigure(isotope='Cl-35', mt=103)
    >>> fig.add_exfor(exfor_df, label='EXFOR (n,p)')
    >>> fig.add_model(energies, dt_pred, label='Decision Tree', color='red', linestyle='--')
    >>> fig.add_model(energies, xgb_pred, label='XGBoost', color='green')
    >>> fig.add_endf_auto()  # Auto-find ENDF file
    >>> fig.save('cl35_np_comparison.png', dpi=300)
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors

from .endf_reader import ENDFReader, NNDCSigmaFetcher


class CrossSectionFigure:
    """
    Publication-quality cross-section visualization.

    A flexible figure class for creating nuclear cross-section plots with
    support for multiple data sources, customizable styles, and various
    export formats.

    Attributes:
        isotope: Isotope identifier (e.g., 'U-235', 'Cl-35')
        z: Atomic number (parsed from isotope)
        a: Mass number (parsed from isotope)
        mt: Reaction type (MT code)
        title: Figure title
        fig: Matplotlib Figure object
        ax: Matplotlib Axes object

    Example:
        >>> # Basic usage
        >>> fig = CrossSectionFigure('U-235', mt=18)
        >>> fig.add_exfor(df)
        >>> fig.add_endf('path/to/endf.endf')
        >>> fig.show()

        >>> # Advanced customization
        >>> fig = CrossSectionFigure(
        ...     isotope='U-235',
        ...     mt=18,
        ...     title='U-235 Fission Cross Section',
        ...     figsize=(12, 8),
        ...     style='seaborn-v0_8-darkgrid',
        ...     energy_units='eV',
        ...     xs_units='barns',
        ... )
        >>> fig.add_exfor(df, color='blue', alpha=0.5, marker='o', s=20)
        >>> fig.add_endf_auto(color='black', linewidth=2, label='ENDF/B-VIII.0')
        >>> fig.add_model(e, xs, color='red', linestyle='--', label='ML Model')
        >>> fig.set_energy_range(1e-5, 2e7)
        >>> fig.add_legend(loc='best')
        >>> fig.save('figure.pdf', dpi=300)
    """

    # Default color palette for multiple series
    DEFAULT_COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]

    # MT code descriptions for auto-labels
    MT_NAMES = {
        1: 'Total',
        2: 'Elastic',
        4: 'Inelastic',
        16: '(n,2n)',
        17: '(n,3n)',
        18: 'Fission',
        102: '(n,γ) Capture',
        103: '(n,p)',
        104: '(n,d)',
        105: '(n,t)',
        106: '(n,³He)',
        107: '(n,α)',
    }

    def __init__(
        self,
        isotope: Optional[str] = None,
        z: Optional[int] = None,
        a: Optional[int] = None,
        mt: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 7),
        style: Optional[str] = 'seaborn-v0_8-darkgrid',
        energy_units: str = 'eV',
        xs_units: str = 'barns',
        log_x: bool = True,
        log_y: bool = True,
        grid: bool = True,
        grid_alpha: float = 0.3,
        endf_dir: Union[str, Path] = 'data/ENDF-B/neutrons',
    ):
        """
        Initialize cross-section figure.

        Args:
            isotope: Isotope string (e.g., 'U-235', 'Cl-35'). Parses Z and A automatically.
            z: Atomic number (alternative to isotope string)
            a: Mass number (alternative to isotope string)
            mt: Reaction type (MT code). Used for auto-labeling and ENDF lookup.
            title: Figure title. Auto-generated if None.
            figsize: Figure size in inches (width, height)
            style: Matplotlib style name. None for default.
            energy_units: Units for energy axis label
            xs_units: Units for cross-section axis label
            log_x: Use logarithmic x-axis
            log_y: Use logarithmic y-axis
            grid: Show grid lines
            grid_alpha: Grid transparency
            endf_dir: Directory containing ENDF files for auto-lookup

        Example:
            >>> fig = CrossSectionFigure('U-235', mt=18)
            >>> fig = CrossSectionFigure(z=92, a=235, mt=18)
        """
        # Parse isotope
        if isotope is not None:
            self.z, self.a, self.symbol = self._parse_isotope(isotope)
        elif z is not None and a is not None:
            self.z = z
            self.a = a
            self.symbol = self._z_to_symbol(z)
        else:
            self.z = None
            self.a = None
            self.symbol = None

        self.isotope = f"{self.symbol}-{self.a}" if self.symbol else None
        self.mt = mt
        self.endf_dir = Path(endf_dir)

        # Figure settings
        self.energy_units = energy_units
        self.xs_units = xs_units
        self.log_x = log_x
        self.log_y = log_y

        # Apply style
        if style:
            try:
                plt.style.use(style)
            except OSError:
                pass  # Style not available, use default

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Configure axes
        if log_x:
            self.ax.set_xscale('log')
        if log_y:
            self.ax.set_yscale('log')

        if grid:
            self.ax.grid(True, alpha=grid_alpha, which='both')

        # Set labels
        self.ax.set_xlabel(f'Energy ({energy_units})', fontsize=12, fontweight='bold')
        self.ax.set_ylabel(f'Cross Section ({xs_units})', fontsize=12, fontweight='bold')

        # Set title
        if title:
            self.ax.set_title(title, fontsize=14, fontweight='bold')
        elif self.isotope and self.mt:
            mt_name = self.MT_NAMES.get(self.mt, f'MT-{self.mt}')
            self.ax.set_title(f'{self.isotope} {mt_name} Cross Section',
                             fontsize=14, fontweight='bold')

        # Track added series for legend and colors
        self._series_count = 0
        self._legend_handles = []
        self._legend_labels = []

    @staticmethod
    def _parse_isotope(isotope: str) -> Tuple[int, int, str]:
        """
        Parse isotope string to Z, A, symbol.

        Supports formats: 'U-235', 'U235', '235U', 'Uranium-235'

        Args:
            isotope: Isotope string

        Returns:
            Tuple of (Z, A, symbol)
        """
        import re

        # Element symbols to Z
        symbol_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
            'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
        }

        # Try different patterns
        # Pattern 1: 'U-235', 'Cl-35'
        match = re.match(r'^([A-Z][a-z]?)-?(\d+)$', isotope)
        if match:
            symbol = match.group(1)
            a = int(match.group(2))
            z = symbol_to_z.get(symbol)
            if z:
                return z, a, symbol

        # Pattern 2: '235U'
        match = re.match(r'^(\d+)([A-Z][a-z]?)$', isotope)
        if match:
            a = int(match.group(1))
            symbol = match.group(2)
            z = symbol_to_z.get(symbol)
            if z:
                return z, a, symbol

        raise ValueError(f"Could not parse isotope: {isotope}")

    @staticmethod
    def _z_to_symbol(z: int) -> str:
        """Convert atomic number to element symbol."""
        symbols = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
        }
        return symbols.get(z, f'Z{z}')

    def _get_next_color(self) -> str:
        """Get next color from palette."""
        color = self.DEFAULT_COLORS[self._series_count % len(self.DEFAULT_COLORS)]
        self._series_count += 1
        return color

    @staticmethod
    def _clean_data(
        energies: np.ndarray,
        xs: np.ndarray,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean data for log-scale plotting.

        Removes NaN, non-positive values, and applies energy filters.
        """
        energies = np.asarray(energies)
        xs = np.asarray(xs)

        # Remove NaN
        mask = np.isfinite(energies) & np.isfinite(xs)

        # Remove non-positive (required for log scale)
        mask &= (energies > 0) & (xs > 0)

        # Apply energy range
        if energy_min is not None:
            mask &= energies >= energy_min
        if energy_max is not None:
            mask &= energies <= energy_max

        # Sort by energy
        sort_idx = np.argsort(energies[mask])

        return energies[mask][sort_idx], xs[mask][sort_idx]

    # =========================================================================
    # DATA ADDITION METHODS
    # =========================================================================

    def add_exfor(
        self,
        data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
        energy_column: str = 'Energy',
        xs_column: str = 'CrossSection',
        uncertainty_column: Optional[str] = 'Uncertainty',
        label: Optional[str] = None,
        color: Optional[str] = None,
        marker: str = 'o',
        s: float = 30,
        alpha: float = 0.6,
        edgecolors: str = 'none',
        show_uncertainty: bool = False,
        uncertainty_alpha: float = 0.2,
        zorder: int = 1,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add EXFOR experimental data points.

        Args:
            data: DataFrame with energy/XS columns, or tuple of (energies, xs) arrays
            energy_column: Column name for energy values
            xs_column: Column name for cross-section values
            uncertainty_column: Column name for uncertainty (optional)
            label: Legend label. Auto-generated if None.
            color: Point color. Auto-assigned if None.
            marker: Marker style
            s: Marker size
            alpha: Marker transparency
            edgecolors: Marker edge color
            show_uncertainty: Show error bars or uncertainty band
            uncertainty_alpha: Uncertainty band transparency
            zorder: Drawing order (lower = behind)
            energy_min: Minimum energy filter (eV)
            energy_max: Maximum energy filter (eV)
            **kwargs: Additional arguments passed to ax.scatter()

        Returns:
            self for method chaining

        Example:
            >>> fig.add_exfor(df, color='blue', s=40, label='EXFOR Data')
            >>> fig.add_exfor((energies, xs), marker='s', alpha=0.7)
        """
        # Parse input
        if isinstance(data, pd.DataFrame):
            energies = data[energy_column].values
            xs = data[xs_column].values
            if show_uncertainty and uncertainty_column in data.columns:
                uncertainties = data[uncertainty_column].values
            else:
                uncertainties = None
        else:
            energies, xs = data
            uncertainties = None

        # Clean data
        energies, xs = self._clean_data(energies, xs, energy_min, energy_max)

        # Auto-assign color and label
        if color is None:
            color = self._get_next_color()
        if label is None:
            label = f'EXFOR ({len(energies)} pts)'

        # Plot scatter
        scatter = self.ax.scatter(
            energies, xs,
            c=color, marker=marker, s=s, alpha=alpha,
            edgecolors=edgecolors, zorder=zorder, label=label,
            **kwargs
        )

        # Add uncertainty bars if requested
        if show_uncertainty and uncertainties is not None:
            # Clean uncertainties to match
            valid_mask = (uncertainties > 0) & np.isfinite(uncertainties)
            if valid_mask.any():
                self.ax.errorbar(
                    energies[valid_mask], xs[valid_mask],
                    yerr=uncertainties[valid_mask],
                    fmt='none', color=color, alpha=uncertainty_alpha * 2,
                    zorder=zorder - 0.5
                )

        self._legend_handles.append(scatter)
        self._legend_labels.append(label)

        return self

    def add_endf(
        self,
        filepath: Union[str, Path],
        mt: Optional[int] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linewidth: float = 2.0,
        linestyle: str = '-',
        alpha: float = 0.9,
        zorder: int = 2,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        xs_min: float = 1e-15,
        prefer: str = "auto",
        library: str = "endfb8.0",
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add ENDF-B evaluated data from file, with automatic NNDC fallback.

        When ``prefer="auto"`` (default), the method inspects the MF=3
        pointwise data.  If the data is *sparse* (< 200 points, indicating
        resonance parameters stored in MF=2), it automatically attempts to
        fetch resonance-reconstructed pointwise data from NNDC Sigma.  If
        the NNDC fetch fails (timeout / no connectivity), it falls back to
        raw MF=3 with a clear warning.

        Args:
            filepath: Path to ENDF-6 formatted file
            mt: Reaction type. Uses figure's MT if None.
            label: Legend label. Auto-generated from data source if None.
            color: Line color. Auto-assigned if None.
            linewidth: Line width
            linestyle: Line style ('-', '--', '-.', ':')
            alpha: Line transparency
            zorder: Drawing order
            energy_min: Minimum energy filter (eV)
            energy_max: Maximum energy filter (eV)
            xs_min: Minimum cross-section filter (barns). ENDF files use very
                    small placeholder values (e.g., 1e-20) for zero cross-section
                    below threshold. This filter removes those placeholders.
                    Default: 1e-15 barns. Set to 0 to include all data.
                    Only applied to raw MF=3 data (not NNDC data).
            prefer: Data source preference:

                * ``"auto"`` – use NNDC when MF=3 is sparse (default)
                * ``"endf"`` – always use local MF=3
                * ``"nndc"`` – always fetch from NNDC Sigma
            library: NNDC library id (``"endfb8.0"``, ``"endfb7.1"``, …).
            **kwargs: Additional arguments passed to ax.plot()

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If ENDF file not found
            ValueError: If MT reaction not in file

        Example:
            >>> fig.add_endf('data/ENDF-B/neutrons/n-092_U_235.endf', mt=18)
            >>> fig.add_endf(endf_path, color='black', linewidth=2.5)
            >>> # Force raw MF=3 even for sparse data:
            >>> fig.add_endf(endf_path, mt=103, prefer='endf')
            >>> # Force NNDC Sigma reconstructed data:
            >>> fig.add_endf(endf_path, mt=103, prefer='nndc')
        """
        # Use figure's MT if not specified
        if mt is None:
            mt = self.mt
        if mt is None:
            raise ValueError("MT must be specified (either in add_endf or figure constructor)")

        # Read ENDF file and decide data source
        reader = ENDFReader(filepath)
        energies, xs, source = reader.get_cross_section_best(
            mt, prefer=prefer, library=library,
            energy_min=energy_min, energy_max=energy_max,
        )

        # Filter out placeholder zeros and non-positive values
        # (essential for log-scale plotting; ENDF files use tiny placeholders
        # like 1e-20 for below-threshold cross-sections)
        pos_mask = (energies > 0) & (xs > 0)
        if source == "endf_mf3" and xs_min > 0:
            pos_mask &= xs >= xs_min
        energies = energies[pos_mask]
        xs = xs[pos_mask]

        if len(energies) == 0:
            import warnings
            warnings.warn(f"No positive ENDF data for MT={mt} after filtering")
            return self

        # Auto-assign color
        if color is None:
            color = self._get_next_color()

        # Auto-generate label reflecting actual data source
        if label is None:
            mt_desc = reader.get_mt_description(mt)
            if source == "nndc_sigma":
                label = f'NNDC Sigma {mt_desc} (reconstructed pointwise)'
            else:
                label = f'ENDF MF=3 {mt_desc} (raw)'

        # Plot line
        line, = self.ax.plot(
            energies, xs,
            color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=alpha, zorder=zorder, label=label,
            **kwargs
        )

        self._legend_handles.append(line)
        self._legend_labels.append(label)

        return self

    def add_endf_auto(
        self,
        mt: Optional[int] = None,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Automatically find and add ENDF-B data for the figure's isotope.

        Uses the isotope (Z, A) set in the constructor to find the
        corresponding ENDF file in the configured ENDF directory.

        Args:
            mt: Reaction type. Uses figure's MT if None.
            **kwargs: Arguments passed to add_endf()

        Returns:
            self for method chaining

        Raises:
            ValueError: If isotope not set or ENDF file not found

        Example:
            >>> fig = CrossSectionFigure('U-235', mt=18)
            >>> fig.add_endf_auto()  # Auto-finds n-092_U_235.endf
        """
        if self.z is None or self.a is None:
            raise ValueError("Isotope must be set to use add_endf_auto()")

        filepath = ENDFReader.find_file(self.z, self.a, self.endf_dir)
        if filepath is None:
            raise ValueError(
                f"ENDF file not found for {self.isotope} in {self.endf_dir}. "
                f"Use add_endf() with explicit path."
            )

        return self.add_endf(filepath, mt=mt, **kwargs)

    def add_nndc_sigma(
        self,
        z: Optional[int] = None,
        a: Optional[int] = None,
        mt: Optional[int] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linewidth: float = 2.0,
        linestyle: str = '-',
        alpha: float = 0.9,
        zorder: int = 2,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        library: str = "endfb8.0",
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add processed pointwise cross-section data from NNDC Sigma.

        NNDC Sigma provides cross-section data that has been processed through
        NJOY to reconstruct resonance parameters into pointwise data. This is
        particularly useful for reactions where the local ENDF file only contains
        sparse MF=3 data (like Cl-35 (n,p)).

        Requires internet connection on first call; data is cached locally.

        Args:
            z: Atomic number. Uses figure's Z if None.
            a: Mass number. Uses figure's A if None.
            mt: Reaction type. Uses figure's MT if None.
            label: Legend label. Auto-generated if None.
            color: Line color. Auto-assigned if None.
            linewidth: Line width
            linestyle: Line style ('-', '--', '-.', ':')
            alpha: Line transparency
            zorder: Drawing order
            energy_min: Minimum energy filter (eV)
            energy_max: Maximum energy filter (eV)
            library: Nuclear data library ("endfb8.0", "endfb7.1", "jendl5.0", "jeff3.3")
            **kwargs: Additional arguments passed to ax.plot()

        Returns:
            self for method chaining

        Raises:
            ValueError: If Z, A, or MT not specified
            RuntimeError: If data cannot be fetched from NNDC

        Example:
            >>> # For reactions with resonance structure in MF=2
            >>> fig = CrossSectionFigure('Cl-35', mt=103)
            >>> fig.add_exfor(exfor_df)
            >>> fig.add_nndc_sigma()  # Gets full resonance-resolved data
            >>> fig.add_legend()
            >>> fig.show()
        """
        # Use figure's values if not specified
        if z is None:
            z = self.z
        if a is None:
            a = self.a
        if mt is None:
            mt = self.mt

        if z is None or a is None:
            raise ValueError("Z and A must be specified (either in add_nndc_sigma or figure constructor)")
        if mt is None:
            raise ValueError("MT must be specified (either in add_nndc_sigma or figure constructor)")

        # Fetch from NNDC Sigma
        fetcher = NNDCSigmaFetcher(library=library)
        energies, xs = fetcher.get_cross_section(z, a, mt, energy_min, energy_max)

        # Auto-assign color and label
        if color is None:
            color = self._get_next_color()
        if label is None:
            symbol = fetcher.SYMBOLS.get(z, f"Z{z}")
            mt_name = self.MT_NAMES.get(mt, f'MT-{mt}')
            label = f'NNDC Sigma {library.upper()} {mt_name}'

        # Plot line
        line, = self.ax.plot(
            energies, xs,
            color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=alpha, zorder=zorder, label=label,
            **kwargs
        )

        self._legend_handles.append(line)
        self._legend_labels.append(label)

        return self

    def add_model(
        self,
        energies: np.ndarray,
        predictions: np.ndarray,
        label: str = 'ML Model',
        color: Optional[str] = None,
        linewidth: float = 2.0,
        linestyle: str = '-',
        alpha: float = 0.8,
        zorder: int = 3,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add ML model predictions.

        Args:
            energies: Energy points (eV)
            predictions: Predicted cross-sections (barns)
            label: Legend label
            color: Line color. Auto-assigned if None.
            linewidth: Line width
            linestyle: Line style
            alpha: Line transparency
            zorder: Drawing order (higher = on top)
            energy_min: Minimum energy filter (eV)
            energy_max: Maximum energy filter (eV)
            **kwargs: Additional arguments passed to ax.plot()

        Returns:
            self for method chaining

        Example:
            >>> fig.add_model(e, dt_pred, label='Decision Tree', color='red', linestyle='--')
            >>> fig.add_model(e, xgb_pred, label='XGBoost', color='green')
        """
        # Clean data
        energies, predictions = self._clean_data(
            np.asarray(energies),
            np.asarray(predictions),
            energy_min, energy_max
        )

        # Auto-assign color
        if color is None:
            color = self._get_next_color()

        # Plot line
        line, = self.ax.plot(
            energies, predictions,
            color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=alpha, zorder=zorder, label=label,
            **kwargs
        )

        self._legend_handles.append(line)
        self._legend_labels.append(label)

        return self

    def add_literature(
        self,
        energies: np.ndarray,
        xs: np.ndarray,
        label: str = 'Literature',
        uncertainties: Optional[np.ndarray] = None,
        color: Optional[str] = None,
        marker: str = 's',
        s: float = 60,
        alpha: float = 0.8,
        edgecolors: str = 'black',
        linewidths: float = 1.0,
        show_uncertainty: bool = True,
        zorder: int = 4,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add custom literature data points.

        For adding data from publications, standards, or other sources
        that aren't in EXFOR or ENDF.

        Args:
            energies: Energy points (eV)
            xs: Cross-section values (barns)
            label: Legend label (include citation)
            uncertainties: Uncertainty values (optional)
            color: Point color. Auto-assigned if None.
            marker: Marker style ('s'=square, '^'=triangle, 'D'=diamond, etc.)
            s: Marker size
            alpha: Marker transparency
            edgecolors: Marker edge color
            linewidths: Marker edge width
            show_uncertainty: Show error bars
            zorder: Drawing order (high = on top)
            **kwargs: Additional arguments passed to ax.scatter()

        Returns:
            self for method chaining

        Example:
            >>> # Add data from a publication
            >>> fig.add_literature(
            ...     energies=[1e6, 2e6, 5e6],
            ...     xs=[1.2, 1.1, 0.9],
            ...     uncertainties=[0.1, 0.08, 0.12],
            ...     label='Smith et al. (2023)',
            ...     marker='D',
            ...     color='purple',
            ... )
        """
        energies = np.asarray(energies)
        xs = np.asarray(xs)

        if color is None:
            color = self._get_next_color()

        # Plot points
        scatter = self.ax.scatter(
            energies, xs,
            c=color, marker=marker, s=s, alpha=alpha,
            edgecolors=edgecolors, linewidths=linewidths,
            zorder=zorder, label=label,
            **kwargs
        )

        # Add error bars
        if show_uncertainty and uncertainties is not None:
            uncertainties = np.asarray(uncertainties)
            self.ax.errorbar(
                energies, xs, yerr=uncertainties,
                fmt='none', color=color, alpha=alpha,
                zorder=zorder - 0.5, capsize=3
            )

        self._legend_handles.append(scatter)
        self._legend_labels.append(label)

        return self

    # =========================================================================
    # SVGP / OUTLIER VISUALIZATION
    # =========================================================================

    def add_gp_fit(
        self,
        energies: np.ndarray,
        gp_mean: np.ndarray,
        gp_std: np.ndarray,
        n_sigma: float = 2.0,
        label: str = 'GP fit',
        color: str = '#2ca02c',
        fill_alpha: float = 0.15,
        line_alpha: float = 0.8,
        linewidth: float = 2.0,
        zorder: int = 2,
        log_space: bool = True,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add GP mean curve with uncertainty band.

        Plots the Gaussian Process fit from SVGP outlier detection,
        showing the smooth trend and confidence interval.

        Args:
            energies: Energy values (log10 if log_space=True, linear otherwise)
            gp_mean: GP predicted mean (log10(sigma) if log_space=True)
            gp_std: GP predicted std (log10(sigma) space if log_space=True)
            n_sigma: Number of standard deviations for uncertainty band
            label: Legend label
            color: Line and fill color
            fill_alpha: Uncertainty band transparency
            line_alpha: Mean line transparency
            linewidth: Mean line width
            zorder: Drawing order
            log_space: If True, energies/gp_mean/gp_std are in log10 space
                      and will be converted to linear for plotting.
                      If False, values are already in linear space.
            **kwargs: Additional arguments passed to ax.plot()

        Returns:
            self for method chaining

        Example:
            >>> # Plot GP fit from Parquet z_score columns
            >>> subset = df[(df['Z'] == 92) & (df['A'] == 235) & (df['MT'] == 2)]
            >>> fig.add_gp_fit(
            ...     subset['log_E'].values,
            ...     subset['gp_mean'].values,
            ...     subset['gp_std'].values,
            ... )
        """
        energies = np.asarray(energies)
        gp_mean = np.asarray(gp_mean)
        gp_std = np.asarray(gp_std)

        if log_space:
            # Convert from log10 to linear for plotting
            E_plot = 10 ** energies
            mean_plot = 10 ** gp_mean
            upper_plot = 10 ** (gp_mean + n_sigma * gp_std)
            lower_plot = 10 ** (gp_mean - n_sigma * gp_std)
        else:
            E_plot = energies
            mean_plot = gp_mean
            upper_plot = gp_mean + n_sigma * gp_std
            lower_plot = gp_mean - n_sigma * gp_std

        # Sort by energy
        sort_idx = np.argsort(E_plot)
        E_plot = E_plot[sort_idx]
        mean_plot = mean_plot[sort_idx]
        upper_plot = upper_plot[sort_idx]
        lower_plot = lower_plot[sort_idx]

        # Plot mean line
        line, = self.ax.plot(
            E_plot, mean_plot,
            color=color, linewidth=linewidth,
            alpha=line_alpha, zorder=zorder, label=label,
            **kwargs
        )

        # Plot uncertainty band
        self.ax.fill_between(
            E_plot, lower_plot, upper_plot,
            color=color, alpha=fill_alpha, zorder=zorder - 0.5,
        )

        self._legend_handles.append(line)
        self._legend_labels.append(label)

        return self

    def add_exfor_outliers(
        self,
        data: pd.DataFrame,
        z_threshold: float = 3.0,
        energy_column: str = 'Energy',
        xs_column: str = 'CrossSection',
        z_score_column: str = 'z_score',
        inlier_color: str = 'tab:blue',
        outlier_color: str = 'tab:red',
        inlier_alpha: float = 0.5,
        outlier_alpha: float = 0.9,
        inlier_size: float = 25,
        outlier_size: float = 60,
        outlier_marker: str = 'x',
        inlier_marker: str = 'o',
        zorder: int = 1,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add EXFOR data with outlier highlighting based on z_score.

        Plots inlier and outlier points in different colors/markers for
        visual inspection of SVGP outlier detection results.

        Args:
            data: DataFrame with energy, cross-section, and z_score columns
            z_threshold: Z-score threshold for outlier classification
            energy_column: Column name for energy values
            xs_column: Column name for cross-section values
            z_score_column: Column name for z-score values
            inlier_color: Color for inlier points
            outlier_color: Color for outlier points
            inlier_alpha: Transparency for inlier points
            outlier_alpha: Transparency for outlier points
            inlier_size: Marker size for inlier points
            outlier_size: Marker size for outlier points
            outlier_marker: Marker style for outliers ('x', 'D', '^', etc.)
            inlier_marker: Marker style for inliers ('o', 's', etc.)
            zorder: Drawing order
            **kwargs: Additional arguments passed to ax.scatter()

        Returns:
            self for method chaining

        Example:
            >>> subset = df[(df['Z'] == 92) & (df['A'] == 235) & (df['MT'] == 2)]
            >>> fig.add_exfor_outliers(subset, z_threshold=3.0)
        """
        if z_score_column not in data.columns:
            raise ValueError(
                f"Column '{z_score_column}' not found in data. "
                f"Run ingestion with --run-svgp to add z_score column."
            )

        inliers = data[data[z_score_column] <= z_threshold]
        outliers = data[data[z_score_column] > z_threshold]

        # Plot inliers
        if len(inliers) > 0:
            s_in = self.ax.scatter(
                inliers[energy_column], inliers[xs_column],
                c=inlier_color, marker=inlier_marker, s=inlier_size,
                alpha=inlier_alpha, zorder=zorder,
                label=f'EXFOR inliers ({len(inliers):,})',
                edgecolors='none', **kwargs
            )
            self._legend_handles.append(s_in)
            self._legend_labels.append(f'EXFOR inliers ({len(inliers):,})')

        # Plot outliers
        if len(outliers) > 0:
            s_out = self.ax.scatter(
                outliers[energy_column], outliers[xs_column],
                c=outlier_color, marker=outlier_marker, s=outlier_size,
                alpha=outlier_alpha, zorder=zorder + 1,
                label=f'Outliers z>{z_threshold} ({len(outliers):,})',
                linewidths=1.5,
            )
            self._legend_handles.append(s_out)
            self._legend_labels.append(f'Outliers z>{z_threshold} ({len(outliers):,})')

        return self

    # =========================================================================
    # ANNOTATION AND CUSTOMIZATION
    # =========================================================================

    def add_annotation(
        self,
        text: str,
        xy: Tuple[float, float],
        xytext: Optional[Tuple[float, float]] = None,
        fontsize: int = 10,
        color: str = 'black',
        fontweight: str = 'bold',
        arrowprops: Optional[Dict] = None,
        bbox: Optional[Dict] = None,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add text annotation to the figure.

        Args:
            text: Annotation text
            xy: Point to annotate (x, y in data coordinates)
            xytext: Text position (x, y). If None, placed at xy.
            fontsize: Font size
            color: Text color
            fontweight: Font weight
            arrowprops: Arrow properties dict (e.g., {'arrowstyle': '->'})
            bbox: Text box properties dict
            **kwargs: Additional arguments for ax.annotate()

        Returns:
            self for method chaining

        Example:
            >>> fig.add_annotation(
            ...     'Resonance peak',
            ...     xy=(1e3, 500),
            ...     xytext=(1e4, 1000),
            ...     arrowprops={'arrowstyle': '->', 'color': 'red'},
            ...     bbox={'boxstyle': 'round', 'facecolor': 'yellow', 'alpha': 0.7},
            ... )
        """
        if arrowprops is None and xytext is not None:
            arrowprops = {'arrowstyle': '->', 'color': color, 'lw': 1.5}

        if bbox is None:
            bbox = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}

        self.ax.annotate(
            text, xy=xy, xytext=xytext,
            fontsize=fontsize, color=color, fontweight=fontweight,
            arrowprops=arrowprops, bbox=bbox,
            **kwargs
        )

        return self

    def add_legend(
        self,
        loc: str = 'best',
        fontsize: int = 11,
        framealpha: float = 0.9,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Add legend to the figure.

        Args:
            loc: Legend location ('best', 'upper right', 'lower left', etc.)
            fontsize: Legend font size
            framealpha: Legend frame transparency
            **kwargs: Additional arguments for ax.legend()

        Returns:
            self for method chaining
        """
        self.ax.legend(loc=loc, fontsize=fontsize, framealpha=framealpha, **kwargs)
        return self

    def set_energy_range(
        self,
        energy_min: float,
        energy_max: float,
    ) -> 'CrossSectionFigure':
        """
        Set the energy axis range.

        Args:
            energy_min: Minimum energy (eV)
            energy_max: Maximum energy (eV)

        Returns:
            self for method chaining
        """
        self.ax.set_xlim(energy_min, energy_max)
        return self

    def set_xs_range(
        self,
        xs_min: float,
        xs_max: float,
    ) -> 'CrossSectionFigure':
        """
        Set the cross-section axis range.

        Args:
            xs_min: Minimum cross-section (barns)
            xs_max: Maximum cross-section (barns)

        Returns:
            self for method chaining
        """
        self.ax.set_ylim(xs_min, xs_max)
        return self

    def set_title(self, title: str, fontsize: int = 14, **kwargs) -> 'CrossSectionFigure':
        """Set figure title."""
        self.ax.set_title(title, fontsize=fontsize, fontweight='bold', **kwargs)
        return self

    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================

    def show(self) -> None:
        """Display the figure."""
        plt.tight_layout()
        plt.show()

    def save(
        self,
        filepath: Union[str, Path],
        dpi: int = 300,
        bbox_inches: str = 'tight',
        transparent: bool = False,
        **kwargs,
    ) -> 'CrossSectionFigure':
        """
        Save figure to file.

        Supports PNG, PDF, SVG, and other formats based on file extension.

        Args:
            filepath: Output file path
            dpi: Resolution (dots per inch)
            bbox_inches: Bounding box option ('tight' recommended)
            transparent: Transparent background
            **kwargs: Additional arguments for fig.savefig()

        Returns:
            self for method chaining

        Example:
            >>> fig.save('figure.png', dpi=300)
            >>> fig.save('figure.pdf')  # Vector format for publications
            >>> fig.save('figure.svg')  # Scalable vector graphics
        """
        plt.tight_layout()
        self.fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            **kwargs
        )
        print(f"[OK] Figure saved to {filepath}")
        return self

    def get_figure(self) -> Tuple[Figure, Axes]:
        """
        Get the matplotlib Figure and Axes objects for further customization.

        Returns:
            Tuple of (Figure, Axes)

        Example:
            >>> fig, ax = cross_fig.get_figure()
            >>> ax.axvline(x=1e6, color='gray', linestyle='--')
        """
        return self.fig, self.ax

    def close(self) -> None:
        """Close the figure and free resources."""
        plt.close(self.fig)


# Convenience function for quick plots
def plot_cross_section(
    isotope: str,
    mt: int,
    exfor_df: Optional[pd.DataFrame] = None,
    endf_path: Optional[Union[str, Path]] = None,
    model_data: Optional[List[Tuple[np.ndarray, np.ndarray, str]]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> CrossSectionFigure:
    """
    Quick function to create a cross-section plot.

    Args:
        isotope: Isotope identifier (e.g., 'U-235')
        mt: Reaction type (MT code)
        exfor_df: EXFOR DataFrame with Energy and CrossSection columns
        endf_path: Path to ENDF file (or auto-find if None)
        model_data: List of (energies, predictions, label) tuples
        title: Figure title
        save_path: Path to save figure (optional)
        **kwargs: Additional arguments for CrossSectionFigure

    Returns:
        CrossSectionFigure object

    Example:
        >>> # Quick plot with EXFOR and ENDF
        >>> fig = plot_cross_section('U-235', mt=18, exfor_df=df)
        >>>
        >>> # Compare models
        >>> fig = plot_cross_section(
        ...     'Cl-35', mt=103,
        ...     exfor_df=df,
        ...     model_data=[
        ...         (e, dt_pred, 'Decision Tree'),
        ...         (e, xgb_pred, 'XGBoost'),
        ...     ],
        ...     save_path='comparison.png',
        ... )
    """
    fig = CrossSectionFigure(isotope=isotope, mt=mt, title=title, **kwargs)

    # Add EXFOR
    if exfor_df is not None:
        fig.add_exfor(exfor_df)

    # Add ENDF
    if endf_path is not None:
        fig.add_endf(endf_path)
    else:
        try:
            fig.add_endf_auto()
        except ValueError:
            pass  # ENDF not found, skip

    # Add models
    if model_data is not None:
        for energies, predictions, label in model_data:
            fig.add_model(energies, predictions, label=label)

    # Add legend
    fig.add_legend()

    # Save if requested
    if save_path is not None:
        fig.save(save_path)

    return fig
