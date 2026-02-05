"""
Isotope Plotter -- One-Line Cross-Section Visualization
========================================================

High-level convenience class for plotting ML model predictions against
EXFOR experimental data for specific isotopes and reaction channels.

Wraps :class:`CrossSectionFigure` with automatic data filtering, prediction
generation, and optional ENDF/NNDC overlay.  Constructed once with shared
context (training data, models, defaults), then reused for any number of
``.plot()`` calls.

Example::

    from nucml_next.visualization import IsotopePlotter

    plotter = IsotopePlotter(
        training_df=df_tier,
        models={'Decision Tree': dt_model, 'XGBoost': xgb_model},
        energy_range=(1e-4, 1e7),
    )

    # One line per isotope
    plotter.plot(Z=92, A=233, MT=1)
    plotter.plot(Z=17, A=35, MT=103, show_endf=True, save_path='cl35.png')

    # Show only a subset of models
    plotter.plot(Z=92, A=233, MT=1, models=['Decision Tree'])

    # Discover available data
    plotter.available_isotopes()
    plotter.available_reactions(Z=92, A=233)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cross_section_figure import CrossSectionFigure


# -------------------------------------------------------------------------
# MT code descriptions (subset -- CrossSectionFigure.MT_NAMES has the full
# set, but we duplicate a small lookup here for error messages so that
# IsotopePlotter has no dependency on CSF internals).
# -------------------------------------------------------------------------
_MT_NAMES: Dict[int, str] = {
    1: 'Total', 2: 'Elastic', 4: 'Inelastic', 16: '(n,2n)',
    17: '(n,3n)', 18: 'Fission', 102: '(n,g) Capture',
    103: '(n,p)', 104: '(n,d)', 105: '(n,t)', 106: '(n,3He)',
    107: '(n,a)',
}

# Element symbols for auto-titles  (Z = 1..99)
_SYMBOLS: Dict[int, str] = {
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


def _isotope_str(Z: int, A: int) -> str:
    """Return a human-readable isotope string, e.g. 'U-233'."""
    sym = _SYMBOLS.get(Z, f'Z{Z}')
    return f'{sym}-{A}'


def _mt_str(mt: int) -> str:
    """Return a human-readable reaction string, e.g. '(n,p)' or 'MT-103'."""
    return _MT_NAMES.get(mt, f'MT-{mt}')


class IsotopePlotter:
    """
    One-line cross-section plotting for isotope + reaction combinations.

    Encapsulates the full pipeline: filter training data by (Z, A),
    generate predictions from one or more trained models, extract EXFOR
    ground truth for the specific MT, and produce a publication-quality
    :class:`CrossSectionFigure`.

    Parameters
    ----------
    training_df : pd.DataFrame
        Full training DataFrame (output of ``NucmlDataset.to_tabular()``).
        Must contain columns ``Z``, ``A``, ``MT``, ``Energy``,
        ``CrossSection``, and all feature columns the models expect.
    models : dict
        Mapping of ``{display_name: trained_model}``.  Each model must have
        a ``.predict(df) -> np.ndarray`` method returning cross-section
        values in original (barns) scale.
    energy_range : tuple of float
        Default ``(E_min, E_max)`` in eV applied to both data filtering
        and axis limits.  Can be overridden per-plot.
    endf_dir : str or Path
        Directory containing ENDF-6 files for ``add_endf_auto()``.
    clip_floor : float
        Minimum positive value for predictions (prevents log-scale errors).
    figsize : tuple of float
        Default ``(width, height)`` in inches.
    exfor_color : str
        Default scatter colour for EXFOR data points.
    exfor_alpha : float
        Default transparency for EXFOR scatter.
    exfor_size : float
        Default marker size for EXFOR scatter.
    experiment_dir : str or Path, optional
        Experiment directory.  When set, every ``plot()`` call
        auto-saves ``.png`` and ``.pdf`` into ``experiment_dir/figures/``.
    """

    # Colour / linestyle rotation for multiple models
    MODEL_COLORS = [
        'tab:red', 'tab:green', 'tab:purple',
        'tab:orange', 'tab:brown',
    ]
    MODEL_LINESTYLES = ['-', '--', '-.', ':', '-']

    def __init__(
        self,
        training_df: pd.DataFrame,
        models: Dict[str, Any],
        energy_range: Tuple[float, float] = (1e-5, 2e7),
        endf_dir: Union[str, Path] = 'data/ENDF-B/neutrons',
        clip_floor: float = 1e-30,
        figsize: Tuple[float, float] = (14, 7),
        exfor_color: str = 'tab:blue',
        exfor_alpha: float = 0.5,
        exfor_size: float = 30,
        experiment_dir: Optional[Union[str, Path]] = None,
    ):
        # Validate required columns
        required = {'Z', 'A', 'MT', 'Energy', 'CrossSection'}
        missing = required - set(training_df.columns)
        if missing:
            raise ValueError(
                f"training_df is missing required columns: {missing}. "
                f"Ensure you pass the output of NucmlDataset.to_tabular()."
            )

        if not models:
            raise ValueError("models dict must contain at least one entry.")

        self.training_df = training_df
        self.models = models
        self.energy_range = energy_range
        self.endf_dir = str(endf_dir)
        self.clip_floor = clip_floor
        self.figsize = figsize
        self.exfor_color = exfor_color
        self.exfor_alpha = exfor_alpha
        self.exfor_size = exfor_size
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def plot(
        self,
        Z: int,
        A: int,
        MT: int,
        *,
        models: Optional[List[str]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        show_endf: bool = False,
        endf_prefer: str = 'auto',
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        save_dpi: int = 300,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        exfor_label: Optional[str] = None,
        exfor_color: Optional[str] = None,
        exfor_size: Optional[float] = None,
    ) -> CrossSectionFigure:
        """
        Plot cross-section predictions vs EXFOR data for one isotope + MT.

        Parameters
        ----------
        Z : int
            Atomic number of the target isotope.
        A : int
            Mass number of the target isotope.
        MT : int
            Reaction type (MT code).
        models : list of str, optional
            Subset of registered model names to include.  ``None`` = all.
        energy_range : tuple of float, optional
            Override the default energy range for this plot.
        show_endf : bool
            Overlay ENDF/NNDC evaluated data (auto-fallback).
        endf_prefer : str
            ENDF source preference (``'auto'``, ``'endf'``, ``'nndc'``).
        title : str, optional
            Custom figure title.  Auto-generated if ``None``.
        save_path : str or Path, optional
            Save the figure to this path.
        save_dpi : int
            DPI for saved figure.
        show : bool
            Call ``fig.show()`` at the end.  Set ``False`` for batch mode.
        figsize : tuple of float, optional
            Override the default figure size for this plot.
        exfor_label : str, optional
            Custom EXFOR legend label.
        exfor_color : str, optional
            Override EXFOR scatter colour for this plot.
        exfor_size : float, optional
            Override EXFOR marker size for this plot.

        Returns
        -------
        CrossSectionFigure
            The figure instance (for further customisation if needed).

        Raises
        ------
        ValueError
            If no data is found for (Z, A), no EXFOR for the MT, or an
            unknown model name is given.
        RuntimeError
            If a model's ``predict()`` call fails.
        """
        e_range = energy_range or self.energy_range
        _figsize = figsize or self.figsize
        _exfor_color = exfor_color or self.exfor_color
        _exfor_size = exfor_size or self.exfor_size

        # Resolve model list
        model_names = self._resolve_model_names(models)

        # Step 1: filter isotope data (all MTs)
        df_isotope = self._filter_isotope(Z, A, e_range)

        # Step 2: extract EXFOR ground truth for this MT
        df_exfor = self._extract_exfor_ground_truth(df_isotope, MT)

        # Step 3: generate predictions for this MT
        predictions_list = self._generate_predictions(
            df_isotope, MT, model_names,
        )

        # Step 4: build CrossSectionFigure
        iso_str = _isotope_str(Z, A)
        mt_str = _mt_str(MT)
        if title is None:
            if len(model_names) == 1:
                title = f'{iso_str} {mt_str}: {model_names[0]}'
            else:
                title = f'{iso_str} {mt_str}'

        fig = CrossSectionFigure(
            z=Z, a=A, mt=MT,
            title=title,
            figsize=_figsize,
            endf_dir=self.endf_dir,
        )

        # EXFOR scatter
        _label = exfor_label or f'EXFOR ({len(df_exfor):,} pts)'
        fig.add_exfor(
            df_exfor,
            label=_label,
            color=_exfor_color,
            s=_exfor_size,
            alpha=self.exfor_alpha,
            zorder=1,
        )

        # Model prediction lines
        for i, (energies, preds, name) in enumerate(predictions_list):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            ls = self.MODEL_LINESTYLES[i % len(self.MODEL_LINESTYLES)]
            fig.add_model(
                energies, preds,
                label=name,
                color=color,
                linewidth=2.5,
                linestyle=ls,
                alpha=0.9,
                zorder=2 + i,
            )

        # Step 5 (optional): ENDF overlay
        if show_endf:
            try:
                fig.add_endf_auto(prefer=endf_prefer)
            except Exception as exc:
                warnings.warn(
                    f"ENDF overlay skipped for {iso_str} MT={MT}: {exc}"
                )

        # Finalise
        fig.set_energy_range(e_range[0], e_range[1])
        fig.add_legend(loc='best', fontsize=11)

        if save_path is not None:
            fig.save(save_path, dpi=save_dpi)
        if show:
            fig.show()

        # Auto-save to experiment figures directory
        if self.experiment_dir is not None:
            fig_dir = self.experiment_dir / 'figures'
            fig_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{_isotope_str(Z, A)}_MT{MT}".replace('-', '')
            fig.save(fig_dir / f'{slug}.png', dpi=300)
            fig.save(fig_dir / f'{slug}.pdf')

        return fig

    def available_isotopes(self) -> pd.DataFrame:
        """
        List all (Z, A) combinations in the training data with counts.

        Returns
        -------
        pd.DataFrame
            Columns: ``Z``, ``A``, ``Symbol``, ``Isotope``, ``Count``.
            Sorted by count descending.
        """
        counts = (
            self.training_df
            .groupby(['Z', 'A'])
            .size()
            .reset_index(name='Count')
            .sort_values('Count', ascending=False)
            .reset_index(drop=True)
        )
        counts['Symbol'] = counts['Z'].map(
            lambda z: _SYMBOLS.get(z, f'Z{z}')
        )
        counts['Isotope'] = counts.apply(
            lambda r: f"{r['Symbol']}-{r['A']}", axis=1
        )
        return counts[['Z', 'A', 'Symbol', 'Isotope', 'Count']]

    def available_reactions(self, Z: int, A: int) -> pd.DataFrame:
        """
        List all MT codes available for a specific isotope.

        Parameters
        ----------
        Z : int
            Atomic number.
        A : int
            Mass number.

        Returns
        -------
        pd.DataFrame
            Columns: ``MT``, ``Name``, ``Count``.
        """
        mask = (self.training_df['Z'] == Z) & (self.training_df['A'] == A)
        subset = self.training_df.loc[mask]
        if subset.empty:
            raise ValueError(
                f"No data found for Z={Z}, A={A} "
                f"({_isotope_str(Z, A)}). "
                f"Use plotter.available_isotopes() to see available isotopes."
            )
        counts = (
            subset
            .groupby('MT')
            .size()
            .reset_index(name='Count')
            .sort_values('Count', ascending=False)
            .reset_index(drop=True)
        )
        counts['Name'] = counts['MT'].map(
            lambda mt: _MT_NAMES.get(mt, f'MT-{mt}')
        )
        return counts[['MT', 'Name', 'Count']]

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _resolve_model_names(self, names: Optional[List[str]]) -> List[str]:
        """Validate and return the list of model names to plot."""
        if names is None:
            return list(self.models.keys())
        unknown = set(names) - set(self.models.keys())
        if unknown:
            raise ValueError(
                f"Model(s) not found: {unknown}. "
                f"Registered models: {list(self.models.keys())}"
            )
        return list(names)

    def _filter_isotope(
        self,
        Z: int,
        A: int,
        energy_range: Tuple[float, float],
    ) -> pd.DataFrame:
        """
        Filter training_df to rows for a specific isotope and energy range.

        Returns a DataFrame sorted by Energy containing ALL MT codes for
        this isotope (models predict on all feature columns, which include
        the particle emission vector derived from MT).
        """
        mask = (
            (self.training_df['Z'] == Z)
            & (self.training_df['A'] == A)
            & (self.training_df['Energy'] >= energy_range[0])
            & (self.training_df['Energy'] <= energy_range[1])
        )
        df = self.training_df.loc[mask].copy()
        if df.empty:
            raise ValueError(
                f"No data found for {_isotope_str(Z, A)} (Z={Z}, A={A}) "
                f"in energy range [{energy_range[0]:.2e}, {energy_range[1]:.2e}] eV. "
                f"Use plotter.available_isotopes() to see available isotopes."
            )
        return df.sort_values('Energy').reset_index(drop=True)

    def _extract_exfor_ground_truth(
        self,
        df_isotope: pd.DataFrame,
        MT: int,
    ) -> pd.DataFrame:
        """
        Extract EXFOR ground truth for a specific MT from the isotope data.

        Filters by MT and removes rows with non-positive CrossSection
        (required for log-scale scatter).
        """
        df_mt = df_isotope[df_isotope['MT'] == MT].copy()
        df_mt = df_mt[df_mt['CrossSection'] > 0]

        if df_mt.empty:
            iso = _isotope_str(
                df_isotope['Z'].iloc[0], df_isotope['A'].iloc[0],
            )
            available_mts = sorted(df_isotope['MT'].unique())
            mt_strs = [f"{m} ({_mt_str(m)})" for m in available_mts[:15]]
            extra = f" ... and {len(available_mts) - 15} more" if len(available_mts) > 15 else ""
            raise ValueError(
                f"No EXFOR measurements found for {iso} MT={MT} "
                f"({_mt_str(MT)}). "
                f"Available MTs for this isotope: {', '.join(mt_strs)}{extra}. "
                f"Use plotter.available_reactions({df_isotope['Z'].iloc[0]}, "
                f"{df_isotope['A'].iloc[0]}) for the full list."
            )
        return df_mt

    def _generate_predictions(
        self,
        df_isotope: pd.DataFrame,
        MT: int,
        model_names: List[str],
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Generate model predictions for a specific isotope and MT.

        Filters ``df_isotope`` to the requested MT, calls ``predict()``
        on each model, and clips predictions to ``self.clip_floor``.

        Returns a list of ``(energies, predictions, model_label)`` tuples.
        """
        df_mt = df_isotope[df_isotope['MT'] == MT].copy()
        df_mt = df_mt.sort_values('Energy').reset_index(drop=True)

        if df_mt.empty:
            return []

        energies = df_mt['Energy'].values
        results: List[Tuple[np.ndarray, np.ndarray, str]] = []

        for name in model_names:
            model = self.models[name]
            try:
                preds = model.predict(df_mt)
            except Exception as exc:
                raise RuntimeError(
                    f"Prediction failed for model '{name}' on "
                    f"{_isotope_str(df_mt['Z'].iloc[0], df_mt['A'].iloc[0])} "
                    f"MT={MT}: {exc}"
                ) from exc

            preds = np.asarray(preds, dtype=float)
            preds = np.clip(preds, self.clip_floor, None)

            # Warn if all predictions were clipped
            if np.all(preds <= self.clip_floor):
                warnings.warn(
                    f"All predictions for '{name}' on "
                    f"{_isotope_str(df_mt['Z'].iloc[0], df_mt['A'].iloc[0])} "
                    f"MT={MT} were non-positive after clipping. "
                    f"Model line will not appear on log-scale plot."
                )

            results.append((energies, preds, name))

        return results
