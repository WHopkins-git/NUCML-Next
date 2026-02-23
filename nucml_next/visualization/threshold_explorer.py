"""
Interactive Outlier Threshold Explorer
=======================================

ipywidgets-based interactive explorer for SVGP outlier detection results.
Provides cascading dropdowns for (Z, A, MT) selection, a z-score threshold
slider, a GP probability surface heatmap, and a z-score band plot with
auto-annotated extreme outliers.

Usage (notebook)::

    from nucml_next.visualization.threshold_explorer import ThresholdExplorer
    explorer = ThresholdExplorer('../data/exfor_processed.parquet')
    explorer.show()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.stats import norm

import ipywidgets as widgets
from IPython.display import display

# ── Colourblind-friendly palette ─────────────────────────────────────────────
GP_MEAN_COLOR = '#4682B4'      # steel blue
CONFIDENCE_COLOR = '#4682B4'   # same hue, varied alpha
OUTLIER_ZONE_CLR = '#CD5C5C'   # coral
INLIER_COLOR = '#2E8B8B'       # teal
OUTLIER_COLOR = '#DC143C'      # crimson (legacy, kept for compatibility)
POINT_OUTLIER_COLOR = '#FF8C00'  # dark orange for point outliers (z > threshold)
DISCREPANT_EXP_COLOR = '#DC143C'  # crimson for discrepant experiments
INLIER_SIZE = 10
OUTLIER_SIZE = 30

# ── Standard neutron energy region boundaries (eV) ──────────────────────────
ENERGY_REGIONS: List[Tuple[str, float, float, str]] = [
    ('Thermal',    1e-5,  0.625, '#DAA520'),
    ('Epithermal', 0.625, 1.0e3, '#228B22'),
    ('Resonance',  1.0e3, 1.0e5, '#4682B4'),
    ('Fast',       1.0e5, 2.0e7, '#9370DB'),
]

# ── MT code descriptions ────────────────────────────────────────────────────
_MT_NAMES: Dict[int, str] = {
    1: 'Total', 2: 'Elastic', 4: 'Inelastic', 16: '(n,2n)',
    17: '(n,3n)', 18: 'Fission', 102: r'(n,$\gamma$) Capture',
    103: '(n,p)', 104: '(n,d)', 105: '(n,t)', 106: r'(n,$^3$He)',
    107: r'(n,$\alpha$)',
}

# ── Element symbols (Z = 1..99) ─────────────────────────────────────────────
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

_REQUIRED_COLUMNS = [
    'Z', 'A', 'MT', 'Energy', 'CrossSection',
    'z_score', 'gp_mean', 'gp_std', 'Entry',
]

_OPTIONAL_COLUMNS = [
    'experiment_outlier', 'point_outlier', 'experiment_id', 'calibration_metric',
    'Uncertainty',  # For error bars on filtered data plot
    'sf8',          # For RAW data toggle (SF8=RAW = uncorrected measurements)
]


def _isotope_str(Z: int, A: int) -> str:
    sym = _SYMBOLS.get(Z, f'Z{Z}')
    return f'{sym}-{A}'


def _mt_str(mt: int) -> str:
    return _MT_NAMES.get(mt, f'MT-{mt}')


class ThresholdExplorer:
    """Interactive widget-based outlier threshold explorer for SVGP results.

    Parameters
    ----------
    data : str, Path, or pd.DataFrame
        Path to Parquet file or pre-loaded DataFrame.  Required columns:
        Z, A, MT, Energy, CrossSection, z_score, gp_mean, gp_std, Entry.
    default_threshold : float
        Initial z-score threshold for the slider.
    figsize : tuple of float
        (width, height) for the figure.
    """

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        default_threshold: float = 3.0,
        figsize: Tuple[float, float] = (16, 7),
    ):
        self._figsize = figsize
        self._default_threshold = default_threshold

        # ── Load data ────────────────────────────────────────────────────
        if isinstance(data, (str, Path)):
            # Load required columns plus any optional columns that exist
            all_cols_in_file = pd.read_parquet(str(data), columns=[]).columns.tolist()
            # Workaround: read with no row limit to get column names
            import pyarrow.parquet as pq
            try:
                pq_file = pq.ParquetFile(str(data))
                all_cols_in_file = pq_file.schema.names
            except Exception:
                all_cols_in_file = _REQUIRED_COLUMNS  # Fallback

            cols_to_load = list(_REQUIRED_COLUMNS)
            for opt_col in _OPTIONAL_COLUMNS:
                if opt_col in all_cols_in_file:
                    cols_to_load.append(opt_col)

            self._df = pd.read_parquet(str(data), columns=cols_to_load)
        else:
            self._df = data.copy()

        # Track which optional columns are available
        self._has_experiment_outlier = 'experiment_outlier' in self._df.columns
        self._has_experiment_id = 'experiment_id' in self._df.columns or 'Entry' in self._df.columns
        self._has_uncertainty = 'Uncertainty' in self._df.columns
        self._has_sf8 = 'sf8' in self._df.columns

        # Validate
        if 'z_score' not in self._df.columns:
            raise ValueError(
                "z_score column not found in data. "
                "Run ingestion with --run-svgp to add z_score column."
            )

        # Drop rows with missing z_score
        self._df = self._df.dropna(subset=['z_score'])
        if len(self._df) == 0:
            raise ValueError("No valid z_score values found in data.")

        # Pre-compute log columns
        self._df['log_E'] = np.log10(
            self._df['Energy'].clip(lower=1e-30)
        )
        self._df['log_sigma'] = np.log10(
            self._df['CrossSection'].clip(lower=1e-30)
        )

        # ── Build cascading lookup structures ────────────────────────────
        self._z_values: List[int] = sorted(self._df['Z'].unique().tolist())

        grouped_a = self._df.groupby('Z')['A'].unique()
        self._a_for_z: Dict[int, List[int]] = {
            z: sorted(a_arr.tolist()) for z, a_arr in grouped_a.items()
        }

        grouped_mt = self._df.groupby(['Z', 'A'])['MT'].unique()
        self._mt_for_za: Dict[Tuple[int, int], List[int]] = {
            (z, a): sorted(mt_arr.tolist())
            for (z, a), mt_arr in grouped_mt.items()
        }

        # ── Build widgets ────────────────────────────────────────────────
        self._output = widgets.Output()
        self._build_widgets()

    # =====================================================================
    # Widget construction
    # =====================================================================

    def _build_widgets(self) -> None:
        z_options = [
            (f'{_SYMBOLS.get(z, "Z" + str(z))} ({z})', z)
            for z in self._z_values
        ]

        self._w_z = widgets.Dropdown(
            options=z_options,
            description='Element (Z):',
            style={'description_width': '90px'},
        )
        self._w_a = widgets.Dropdown(
            description='Mass (A):',
            style={'description_width': '80px'},
        )
        self._w_mt = widgets.Dropdown(
            description='Reaction (MT):',
            style={'description_width': '100px'},
        )
        # Generate threshold options: 1.0, 1.5, 2.0, ..., 25.0
        threshold_options = [round(x * 0.5, 1) for x in range(2, 51)]
        self._w_threshold = widgets.Dropdown(
            options=threshold_options,
            value=self._default_threshold,
            description='z threshold:',
            style={'description_width': '90px'},
        )

        # ── Filtering checkboxes ─────────────────────────────────────────
        self._w_exclude_point_outliers = widgets.Checkbox(
            value=True,
            description='Exclude point outliers (z > threshold)',
            style={'description_width': 'initial'},
        )
        self._w_exclude_discrepant = widgets.Checkbox(
            value=False,
            description='Exclude discrepant experiments',
            style={'description_width': 'initial'},
            disabled=not self._has_experiment_outlier,
        )
        self._w_color_by_experiment = widgets.Checkbox(
            value=False,
            description='Color by experiment',
            style={'description_width': 'initial'},
            disabled=not self._has_experiment_id,
        )
        self._w_exclude_raw = widgets.Checkbox(
            value=False,
            description='Exclude RAW data',
            style={'description_width': 'initial'},
            disabled=not self._has_sf8,
        )

        # Wire observers
        self._w_z.observe(self._on_z_change, names='value')
        self._w_a.observe(self._on_a_change, names='value')
        self._w_mt.observe(self._on_mt_change, names='value')
        self._w_threshold.observe(self._on_threshold_change, names='value')
        self._w_exclude_point_outliers.observe(self._on_filter_change, names='value')
        self._w_exclude_discrepant.observe(self._on_filter_change, names='value')
        self._w_color_by_experiment.observe(self._on_experiment_toggle, names='value')
        self._w_exclude_raw.observe(self._on_filter_change, names='value')

        # Build control rows
        row1 = widgets.HBox(
            [self._w_z, self._w_a, self._w_mt, self._w_threshold],
            layout=widgets.Layout(margin='0 0 5px 0'),
        )

        # Second row with filter checkboxes
        row2 = widgets.HBox(
            [
                self._w_exclude_point_outliers,
                self._w_exclude_discrepant,
                self._w_exclude_raw,
                self._w_color_by_experiment,
            ],
            layout=widgets.Layout(margin='0 0 5px 0'),
        )

        # Help text explaining RAW data and discrepant experiments
        help_text = widgets.HTML(
            value=(
                '<div style="font-size:11px; color:#555; margin:0 0 8px 5px; line-height:1.4;">'
                '<b>RAW</b> = uncorrected experimental measurements (SF8=RAW); real data but may have systematic offsets. '
                '<b>Discrepant experiments</b> = entire EXFOR entries whose GP posterior deviates from the multi-experiment consensus.'
                '</div>'
            ),
        )

        self._ui = widgets.VBox([row1, row2, help_text, self._output])

        # Initialise cascade (triggers A -> MT -> plot)
        self._on_z_change(None)

    # =====================================================================
    # Cascading dropdown observers
    # =====================================================================

    def _on_z_change(self, change) -> None:
        z = self._w_z.value
        a_options = self._a_for_z.get(z, [])

        self._w_a.unobserve(self._on_a_change, names='value')
        self._w_a.options = a_options
        if a_options:
            self._w_a.value = a_options[0]
        self._w_a.observe(self._on_a_change, names='value')

        self._on_a_change(None)

    def _on_a_change(self, change) -> None:
        z = self._w_z.value
        a = self._w_a.value
        mt_options = self._mt_for_za.get((z, a), [])

        mt_labeled = [(f'{_mt_str(mt)} ({mt})', mt) for mt in mt_options]

        self._w_mt.unobserve(self._on_mt_change, names='value')
        self._w_mt.options = mt_labeled
        if mt_labeled:
            self._w_mt.value = mt_labeled[0][1]
        self._w_mt.observe(self._on_mt_change, names='value')

        self._on_mt_change(None)

    def _on_mt_change(self, change) -> None:
        self._update_plot()

    def _on_threshold_change(self, change) -> None:
        self._update_plot()

    def _on_filter_change(self, change) -> None:
        self._update_plot()

    def _on_experiment_toggle(self, change) -> None:
        self._update_plot()

    # =====================================================================
    # Public API
    # =====================================================================

    def show(self) -> None:
        """Display the interactive explorer in a Jupyter notebook."""
        display(self._ui)

    def get_filter_settings(self) -> dict:
        """Return current filter settings for use in DataSelection.

        Call this after user has configured the explorer to get the settings
        they chose, then pass to DataSelection.

        Returns
        -------
        dict
            Dictionary with keys:
            - z_threshold: float, current z-score threshold
            - exclude_point_outliers: bool, whether to exclude z > threshold
            - exclude_discrepant_experiments: bool, whether to exclude discrepant experiments
        """
        return {
            'z_threshold': self._w_threshold.value,
            'exclude_point_outliers': self._w_exclude_point_outliers.value,
            'exclude_discrepant_experiments': self._w_exclude_discrepant.value,
            'exclude_raw': self._w_exclude_raw.value,
        }

    # =====================================================================
    # Plot orchestration
    # =====================================================================

    def _is_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean mask for rows where sf8 contains 'RAW'."""
        if not self._has_sf8 or 'sf8' not in df.columns:
            return np.zeros(len(df), dtype=bool)
        return df['sf8'].fillna('').str.contains('RAW', na=False).values

    def _update_plot(self) -> None:
        z = self._w_z.value
        a = self._w_a.value
        mt = self._w_mt.value

        if z is None or a is None or mt is None:
            return

        group = self._df[
            (self._df['Z'] == z)
            & (self._df['A'] == a)
            & (self._df['MT'] == mt)
        ].copy()

        if len(group) == 0:
            with self._output:
                self._output.clear_output(wait=True)
                print(f"No data for {_isotope_str(z, a)} MT={mt}")
            return

        # Detect MAD fallback or single-point
        is_single = len(group) == 1
        is_mad = (
            not is_single
            and group['gp_mean'].nunique() == 1
            and group['gp_std'].nunique() == 1
        )

        if is_single or is_mad:
            self._plot_fallback(group, is_single=is_single)
        else:
            self._plot_full(group)

    # =====================================================================
    # Full GP plot (probability surface + z-score bands)
    # =====================================================================

    def _plot_full(self, df: pd.DataFrame) -> None:
        threshold = self._w_threshold.value
        stats = self._compute_statistics(df, threshold)
        z, a, mt = int(df['Z'].iloc[0]), int(df['A'].iloc[0]), int(df['MT'].iloc[0])
        iso = _isotope_str(z, a)
        reaction = _mt_str(mt)

        with self._output:
            self._output.clear_output(wait=True)

            # 3-panel layout: GP surface | All data highlighted | Filtered data
            fig = plt.figure(figsize=(self._figsize[0] + 4, self._figsize[1] + 1.5))

            gs = gridspec.GridSpec(
                2, 3,
                width_ratios=[1, 1, 1],
                height_ratios=[1, 0.08],
                hspace=0.35,
                wspace=0.30,
            )

            ax_zscore = fig.add_subplot(gs[0, 0])
            ax_all = fig.add_subplot(gs[0, 1])
            ax_filtered = fig.add_subplot(gs[0, 2])
            ax_stats = fig.add_subplot(gs[1, :])

            # Draw panels
            self._draw_zscore_heatmap(ax_zscore, df, threshold)
            self._draw_all_data_highlighted(ax_all, df, threshold)
            self._draw_filtered_with_errorbars(ax_filtered, df, threshold)

            # Sync y-axis limits across all 3 main panels
            y_min = min(ax_zscore.get_ylim()[0], ax_all.get_ylim()[0], ax_filtered.get_ylim()[0])
            y_max = max(ax_zscore.get_ylim()[1], ax_all.get_ylim()[1], ax_filtered.get_ylim()[1])
            for ax in [ax_zscore, ax_all, ax_filtered]:
                ax.set_ylim(y_min, y_max)

            # ── Statistics text bar ──────────────────────────────────────
            ax_stats.axis('off')

            # Build stats text showing filtering effects
            stats_text = (
                f"Total: {stats['total']:,}   |   "
            )
            if stats.get('n_experiments') is not None:
                stats_text += f"Experiments: {stats['n_experiments']}"
                if stats.get('n_discrepant') is not None:
                    stats_text += f" ({stats['n_discrepant']} discrepant)"
                stats_text += "   |   "

            stats_text += (
                f"Point outliers (z>{threshold}): {stats['n_outliers']:,} ({stats['pct_outliers']:.1f}%)   |   "
                f"Remaining: {stats['n_remaining']:,} ({stats['pct_remaining']:.1f}%)"
            )

            ax_stats.text(
                0.5, 0.5, stats_text,
                transform=ax_stats.transAxes,
                ha='center', va='center', fontsize=11,
                family='sans-serif',
                bbox=dict(
                    boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                    alpha=0.8, edgecolor='#cccccc',
                ),
            )

            fig.suptitle(
                f'{iso} {reaction} (MT {mt}) -- '
                f'Outlier Threshold Explorer (z = {threshold:.1f})',
                fontsize=13, fontweight='bold', family='sans-serif',
            )

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            plt.show()

    # =====================================================================
    # Fallback plot (MAD / single-point groups)
    # =====================================================================

    def _plot_fallback(
        self, df: pd.DataFrame, is_single: bool = False,
    ) -> None:
        threshold = self._w_threshold.value
        z, a, mt = int(df['Z'].iloc[0]), int(df['A'].iloc[0]), int(df['MT'].iloc[0])
        iso = _isotope_str(z, a)
        reaction = _mt_str(mt)

        with self._output:
            self._output.clear_output(wait=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            if is_single:
                msg = (
                    f"Single-point group (N=1).\n"
                    f"Cannot assess outlier status (z_score = 0)."
                )
            else:
                msg = (
                    f"MAD fallback group (N={len(df)} < min_group_size).\n"
                    f"Probability surface not available.\n"
                    f"GP mean and std are constant (median / MAD)."
                )

            # Plot data
            log_E = df['log_E'].values
            log_sigma = df['log_sigma'].values
            gp_mean_val = df['gp_mean'].iloc[0]
            gp_std_val = df['gp_std'].iloc[0]

            # Bands (constant)
            E_range = np.array([log_E.min() - 0.5, log_E.max() + 0.5])
            ax.fill_between(
                E_range,
                gp_mean_val - threshold * gp_std_val,
                gp_mean_val + threshold * gp_std_val,
                color=OUTLIER_ZONE_CLR, alpha=0.1,
                label=f'$\\pm {threshold:.1f}\\sigma$',
            )
            ax.axhline(
                gp_mean_val, color=GP_MEAN_COLOR, linewidth=2,
                label='GP mean (median)',
            )
            ax.axhline(
                gp_mean_val + threshold * gp_std_val,
                color=OUTLIER_ZONE_CLR, linestyle='--', linewidth=1, alpha=0.7,
            )
            ax.axhline(
                gp_mean_val - threshold * gp_std_val,
                color=OUTLIER_ZONE_CLR, linestyle='--', linewidth=1, alpha=0.7,
            )

            # Scatter
            z_scores = df['z_score'].values
            inlier_mask = z_scores <= threshold
            outlier_mask = ~inlier_mask

            ax.scatter(
                log_E[inlier_mask], log_sigma[inlier_mask],
                c=INLIER_COLOR, s=INLIER_SIZE * 3, alpha=0.6,
                edgecolors='none', label=f'Inliers ({inlier_mask.sum():,})',
            )
            if outlier_mask.any():
                ax.scatter(
                    log_E[outlier_mask], log_sigma[outlier_mask],
                    c=OUTLIER_COLOR, s=OUTLIER_SIZE, alpha=0.9,
                    edgecolors='black', linewidths=0.5,
                    label=f'Outliers ({outlier_mask.sum():,})',
                )

            # Message box
            ax.text(
                0.5, 0.97, msg,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, family='sans-serif', fontweight='bold',
                color='#666666',
                bbox=dict(
                    boxstyle='round,pad=0.5', facecolor='#fff3cd',
                    edgecolor='#ffc107', alpha=0.9,
                ),
            )

            ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11,
                          family='sans-serif')
            ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11,
                          family='sans-serif')
            ax.set_title(
                f'{iso} {reaction} (MT {mt})',
                fontsize=12, fontweight='bold', family='sans-serif',
            )
            ax.legend(fontsize=9, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show()

    # =====================================================================
    # Z-Score heatmap panel (replaces probability surface)
    # =====================================================================

    def _draw_zscore_heatmap(
        self, ax: Axes, df: pd.DataFrame, threshold: float,
    ) -> None:
        """Draw data points colored by z-score magnitude.

        Uses RdYlBu_r colormap where:
        - Blue = low z-score (good fit to GP)
        - Yellow = moderate z-score
        - Red = high z-score (outlier)
        """
        log_E = df['log_E'].values
        log_sigma = df['log_sigma'].values
        z_scores = df['z_score'].values

        # Clip z-scores for colormap range (0 to 2*threshold)
        z_clipped = np.clip(z_scores, 0, threshold * 2)

        scatter = ax.scatter(
            log_E, log_sigma,
            c=z_clipped, cmap='RdYlBu_r',  # Red = high z-score
            s=INLIER_SIZE * 1.5, alpha=0.7,
            edgecolors='none',
            vmin=0, vmax=threshold * 2,
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label('z-score', fontsize=10, family='sans-serif')

        # Add threshold marker on colorbar
        cbar.ax.axhline(threshold, color='black', linewidth=2, linestyle='--')
        cbar.ax.text(
            1.5, threshold, f'z={threshold}',
            va='center', fontsize=8, family='sans-serif',
        )

        # ── Energy region vertical spans ─────────────────────────────────
        y_lo, y_hi = ax.get_ylim()
        for name, lo_ev, hi_ev, color in ENERGY_REGIONS:
            lo_log = np.log10(max(lo_ev, 1e-30))
            hi_log = np.log10(hi_ev)
            x_min, x_max = log_E.min(), log_E.max()
            if hi_log < x_min or lo_log > x_max:
                continue
            lo_clipped = max(lo_log, x_min)
            hi_clipped = min(hi_log, x_max)
            ax.axvspan(lo_clipped, hi_clipped, alpha=0.06, color=color, zorder=0)
            mid = (lo_clipped + hi_clipped) / 2
            ax.text(
                mid, y_hi, name, ha='center', va='bottom', fontsize=7,
                alpha=0.6, family='sans-serif', style='italic',
            )
        ax.set_ylim(y_lo, y_hi)

        # Style
        ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11, family='sans-serif')
        ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11, family='sans-serif')
        ax.set_title('Z-Score Distribution', fontsize=12,
                     fontweight='bold', family='sans-serif')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # =====================================================================
    # Probability surface panel (legacy, kept for reference)
    # =====================================================================

    def _draw_probability_surface(
        self, ax: Axes, df: pd.DataFrame, threshold: float,
    ) -> None:
        log_E = df['log_E'].values
        log_sigma = df['log_sigma'].values
        gp_mean = df['gp_mean'].values
        gp_std = df['gp_std'].values

        # Sort and de-duplicate for interp1d
        sort_idx = np.argsort(log_E)
        log_E_s = log_E[sort_idx]
        gp_mean_s = gp_mean[sort_idx]
        gp_std_s = gp_std[sort_idx]

        unique_mask = np.concatenate(
            [[True], np.diff(log_E_s) > 1e-12]
        )
        log_E_u = log_E_s[unique_mask]
        gp_mean_u = gp_mean_s[unique_mask]
        gp_std_u = gp_std_s[unique_mask]

        if len(log_E_u) < 2:
            ax.text(
                0.5, 0.5, 'Insufficient unique energy\npoints for surface',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='#666666',
            )
            return

        # Build interpolators
        f_mean = interp1d(
            log_E_u, gp_mean_u, kind='linear',
            fill_value='extrapolate',
        )
        f_std = interp1d(
            log_E_u, gp_std_u, kind='linear',
            fill_value='extrapolate',
        )

        # 200x200 grid
        n_grid = 200
        E_grid = np.linspace(log_E_u.min(), log_E_u.max(), n_grid)
        sigma_pad = 1.0
        sigma_min = log_sigma.min() - sigma_pad
        sigma_max = log_sigma.max() + sigma_pad
        S_grid = np.linspace(sigma_min, sigma_max, n_grid)

        E_mesh, S_mesh = np.meshgrid(E_grid, S_grid)

        mu_at_E = f_mean(E_grid)
        std_at_E = np.clip(f_std(E_grid), 1e-10, None)

        # Gaussian PDF: P(sigma | E)
        P = norm.pdf(
            S_mesh,
            loc=mu_at_E[np.newaxis, :],
            scale=std_at_E[np.newaxis, :],
        )

        # Plot surface
        pcm = ax.pcolormesh(
            E_mesh, S_mesh, P,
            cmap='viridis', shading='auto', rasterized=True,
        )
        cbar = plt.colorbar(pcm, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label(r'$P(\sigma | E)$', fontsize=10, family='sans-serif')

        # Contour lines
        contour_levels = [0.01, 0.05, 0.1, 0.25]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = ax.contour(
                E_mesh, S_mesh, P,
                levels=contour_levels,
                colors='white', linewidths=0.7, alpha=0.7,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

        # Overlay data points
        ax.scatter(
            log_E, log_sigma,
            c='white', s=5, alpha=0.5, edgecolors='none', zorder=5,
        )

        # Style
        ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11,
                      family='sans-serif')
        ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11,
                      family='sans-serif')
        ax.set_title('GP Predictive Distribution', fontsize=12,
                     fontweight='bold', family='sans-serif')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # =====================================================================
    # Z-score bands panel
    # =====================================================================

    def _draw_zscore_bands(
        self, ax: Axes, df: pd.DataFrame, threshold: float,
    ) -> None:
        log_E = df['log_E'].values
        log_sigma = df['log_sigma'].values
        gp_mean = df['gp_mean'].values
        gp_std = df['gp_std'].values
        z_scores = df['z_score'].values

        # Sort for line plotting
        sort_idx = np.argsort(log_E)
        E_s = log_E[sort_idx]
        mu_s = gp_mean[sort_idx]
        std_s = gp_std[sort_idx]

        # GP mean line
        ax.plot(
            E_s, mu_s, color=GP_MEAN_COLOR, linewidth=2,
            label='GP mean', zorder=5,
        )

        # Confidence bands
        ax.fill_between(
            E_s, mu_s - std_s, mu_s + std_s,
            color=CONFIDENCE_COLOR, alpha=0.3,
            label=r'$\pm 1\sigma$', zorder=2,
        )
        ax.fill_between(
            E_s, mu_s - 2 * std_s, mu_s + 2 * std_s,
            color=CONFIDENCE_COLOR, alpha=0.15,
            label=r'$\pm 2\sigma$', zorder=1,
        )

        # Threshold band (outlier zone)
        ax.fill_between(
            E_s, mu_s - threshold * std_s, mu_s + threshold * std_s,
            color=OUTLIER_ZONE_CLR, alpha=0.1,
            label=f'$\\pm {threshold:.1f}\\sigma$ (outlier zone)',
            zorder=0,
        )

        # Dashed lines at threshold boundary
        ax.plot(
            E_s, mu_s + threshold * std_s, '--',
            color=OUTLIER_ZONE_CLR, linewidth=1, alpha=0.7,
        )
        ax.plot(
            E_s, mu_s - threshold * std_s, '--',
            color=OUTLIER_ZONE_CLR, linewidth=1, alpha=0.7,
        )

        # Compute outlier mask (always needed for auto-annotation)
        outlier_mask = z_scores > threshold

        # Scatter: different modes based on "Color by experiment" toggle
        color_by_exp = self._w_color_by_experiment.value

        if color_by_exp and self._has_experiment_id:
            # Per-experiment coloring mode
            self._draw_experiment_scatter(ax, df, threshold)
        else:
            # Default mode: inliers vs outliers
            inlier_mask = ~outlier_mask

            ax.scatter(
                log_E[inlier_mask], log_sigma[inlier_mask],
                c=INLIER_COLOR, s=INLIER_SIZE, alpha=0.5, edgecolors='none',
                label=f'Inliers ({inlier_mask.sum():,})', zorder=6,
            )

            if outlier_mask.any():
                ax.scatter(
                    log_E[outlier_mask], log_sigma[outlier_mask],
                    c=OUTLIER_COLOR, s=OUTLIER_SIZE, alpha=0.9,
                    edgecolors='black', linewidths=0.5,
                    label=f'Outliers ({outlier_mask.sum():,})', zorder=7,
                )

        # ── Energy region vertical spans ─────────────────────────────────
        y_lo, y_hi = ax.get_ylim()
        for name, lo_ev, hi_ev, color in ENERGY_REGIONS:
            lo_log = np.log10(max(lo_ev, 1e-30))
            hi_log = np.log10(hi_ev)
            if hi_log < E_s.min() or lo_log > E_s.max():
                continue
            lo_clipped = max(lo_log, E_s.min())
            hi_clipped = min(hi_log, E_s.max())
            ax.axvspan(lo_clipped, hi_clipped, alpha=0.06, color=color,
                       zorder=0)
            mid = (lo_clipped + hi_clipped) / 2
            ax.text(
                mid, y_hi, name, ha='center', va='bottom', fontsize=7,
                alpha=0.6, family='sans-serif', style='italic',
            )

        # ── Rug plot along x-axis ────────────────────────────────────────
        ax.plot(
            log_E, np.full_like(log_E, y_lo), '|',
            color='gray', markersize=3, alpha=0.3, zorder=3,
        )
        ax.set_ylim(y_lo, y_hi)  # restore after rug

        # ── Auto-annotate top 3 extreme outliers ─────────────────────────
        if outlier_mask.any():
            outlier_df = df[outlier_mask].nlargest(
                min(3, outlier_mask.sum()), 'z_score'
            )
            for _, row in outlier_df.iterrows():
                entry = str(row.get('Entry', 'unknown'))
                # Truncate long entry IDs
                if len(entry) > 12:
                    entry = entry[:12] + '..'
                ax.annotate(
                    f'{entry}\nz={row["z_score"]:.1f}',
                    xy=(row['log_E'], row['log_sigma']),
                    xytext=(12, 12),
                    textcoords='offset points',
                    fontsize=7, color=OUTLIER_COLOR, fontweight='bold',
                    family='sans-serif',
                    arrowprops=dict(
                        arrowstyle='->', color=OUTLIER_COLOR, lw=0.8,
                    ),
                    bbox=dict(
                        boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                        edgecolor=OUTLIER_COLOR, linewidth=0.5,
                    ),
                )

        # Style
        ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11,
                      family='sans-serif')
        ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11,
                      family='sans-serif')
        ax.set_title('GP Fit + Z-Score Bands', fontsize=12,
                     fontweight='bold', family='sans-serif')
        ax.legend(loc='best', fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # =====================================================================
    # All data with outliers highlighted panel
    # =====================================================================

    def _draw_all_data_highlighted(
        self, ax: Axes, df: pd.DataFrame, threshold: float,
    ) -> None:
        """Draw all data points, highlighting only outliers that WILL be removed.

        Only highlights outliers based on ENABLED filter settings:
        - If "Exclude point outliers" is ON: show point outliers in orange
        - If "Exclude discrepant experiments" is ON: show discrepant exp as red X
        - Retained points shown translucent in background
        """
        log_E = df['log_E'].values
        log_sigma = df['log_sigma'].values
        z_scores = df['z_score'].values

        # Color by experiment mode
        color_by_exp = self._w_color_by_experiment.value

        if color_by_exp and self._has_experiment_id:
            # Per-experiment coloring mode (delegate to existing method)
            self._draw_experiment_scatter(ax, df, threshold)
        else:
            # Only compute masks for ENABLED filters
            exclude_points = self._w_exclude_point_outliers.value
            exclude_exp = self._w_exclude_discrepant.value
            exclude_raw = self._w_exclude_raw.value

            # Point outlier mask: only if filter is enabled
            point_outlier_mask = (
                (z_scores > threshold) if exclude_points
                else np.zeros(len(df), dtype=bool)
            )

            # Experiment outlier mask: only if filter is enabled
            exp_outlier_mask = (
                df['experiment_outlier'].values
                if (exclude_exp and 'experiment_outlier' in df.columns)
                else np.zeros(len(df), dtype=bool)
            )

            # RAW data mask: only if filter is enabled
            raw_mask = (
                self._is_raw(df) if exclude_raw
                else np.zeros(len(df), dtype=bool)
            )

            # Compute what will be removed vs retained
            will_be_removed = point_outlier_mask | exp_outlier_mask | raw_mask
            retained_mask = ~will_be_removed

            # 1. Retained points: translucent background
            if retained_mask.any():
                ax.scatter(
                    log_E[retained_mask], log_sigma[retained_mask],
                    c=INLIER_COLOR, s=INLIER_SIZE, alpha=0.3,
                    edgecolors='none', zorder=5,
                    label=f'Retained ({retained_mask.sum():,})',
                )

            # 2. Point outliers: orange, opaque foreground (only if filter enabled)
            if exclude_points:
                point_only = point_outlier_mask & ~exp_outlier_mask
                if point_only.any():
                    ax.scatter(
                        log_E[point_only], log_sigma[point_only],
                        c=POINT_OUTLIER_COLOR, s=OUTLIER_SIZE, alpha=0.9,
                        edgecolors='black', linewidths=0.8,
                        marker='o', zorder=10,
                        label=f'Point outliers z>{threshold} ({point_only.sum():,})',
                    )

            # 3. Discrepant exp: red X, opaque foreground (only if filter enabled)
            if exclude_exp and exp_outlier_mask.any():
                ax.scatter(
                    log_E[exp_outlier_mask], log_sigma[exp_outlier_mask],
                    c=DISCREPANT_EXP_COLOR, s=OUTLIER_SIZE * 1.2, alpha=0.9,
                    edgecolors='darkred', linewidths=1.0,
                    marker='X', zorder=11,
                    label=f'Discrepant exp ({exp_outlier_mask.sum():,})',
                )

            ax.legend(loc='best', fontsize=8, frameon=True, framealpha=0.9)

        # ── Energy region vertical spans ─────────────────────────────────
        y_lo, y_hi = ax.get_ylim()
        for name, lo_ev, hi_ev, color in ENERGY_REGIONS:
            lo_log = np.log10(max(lo_ev, 1e-30))
            hi_log = np.log10(hi_ev)
            x_min, x_max = log_E.min(), log_E.max()
            if hi_log < x_min or lo_log > x_max:
                continue
            lo_clipped = max(lo_log, x_min)
            hi_clipped = min(hi_log, x_max)
            ax.axvspan(lo_clipped, hi_clipped, alpha=0.06, color=color, zorder=0)
            mid = (lo_clipped + hi_clipped) / 2
            ax.text(
                mid, y_hi, name, ha='center', va='bottom', fontsize=7,
                alpha=0.6, family='sans-serif', style='italic',
            )
        ax.set_ylim(y_lo, y_hi)

        # Style
        ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11, family='sans-serif')
        ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11, family='sans-serif')
        ax.set_title('All Data (outliers highlighted)', fontsize=12,
                     fontweight='bold', family='sans-serif')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # =====================================================================
    # Filtered data with error bars panel
    # =====================================================================

    def _draw_filtered_with_errorbars(
        self, ax: Axes, df: pd.DataFrame, threshold: float,
    ) -> None:
        """Draw only the remaining points after filtering, with error bars.

        Applies current filter settings:
        - If exclude_point_outliers: remove z > threshold
        - If exclude_discrepant: remove experiment_outlier=True

        Shows error bars using 'Uncertainty' column if available.
        """
        log_E = df['log_E'].values
        z_scores = df['z_score'].values

        # Apply filters based on checkbox states
        mask = np.ones(len(df), dtype=bool)

        if self._w_exclude_point_outliers.value:
            mask &= z_scores <= threshold

        if self._w_exclude_discrepant.value and 'experiment_outlier' in df.columns:
            mask &= ~df['experiment_outlier'].values

        if self._w_exclude_raw.value:
            mask &= ~self._is_raw(df)

        filtered = df[mask].copy()
        n_filtered = len(filtered)

        if n_filtered == 0:
            ax.text(
                0.5, 0.5, 'No points remain\nafter filtering',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='#666666',
            )
            ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11, family='sans-serif')
            ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11, family='sans-serif')
            ax.set_title('Filtered Data (0 pts)', fontsize=12,
                         fontweight='bold', family='sans-serif')
            return

        # Plot with error bars if uncertainty available
        if self._has_uncertainty and 'Uncertainty' in filtered.columns:
            xs = filtered['CrossSection'].values
            unc = filtered['Uncertainty'].values
            log_xs = np.log10(np.clip(xs, 1e-30, None))
            filt_log_E = filtered['log_E'].values

            # Filter out invalid uncertainties
            valid_unc = np.isfinite(unc) & (unc > 0)

            if valid_unc.any():
                # Asymmetric error bars in log space
                # Cap error bars at MAX_LOG_ERROR to prevent huge bars when unc >= xs
                MAX_LOG_ERROR = 2.0  # Maximum 2 orders of magnitude

                xs_valid = xs[valid_unc]
                unc_valid = unc[valid_unc]
                log_xs_valid = log_xs[valid_unc]
                filt_log_E_valid = filt_log_E[valid_unc]

                # Upper error: log10(xs + unc) - log10(xs), capped
                log_upper = np.log10(np.clip(xs_valid + unc_valid, 1e-30, None)) - log_xs_valid
                log_upper = np.clip(log_upper, 0, MAX_LOG_ERROR)

                # Lower error: when xs - unc <= 0, cap at MAX_LOG_ERROR
                lower_vals = xs_valid - unc_valid
                log_lower = np.where(
                    lower_vals > 0,
                    log_xs_valid - np.log10(lower_vals),
                    MAX_LOG_ERROR  # Cap when uncertainty >= value
                )
                log_lower = np.clip(log_lower, 0, MAX_LOG_ERROR)

                ax.errorbar(
                    filt_log_E_valid, log_xs_valid,
                    yerr=[log_lower, log_upper],
                    fmt='o', color=INLIER_COLOR, markersize=4,
                    ecolor='gray', elinewidth=0.5, capsize=0,
                    alpha=0.7,
                    label=f'With uncertainty ({valid_unc.sum():,})',
                )

                # Plot points without valid uncertainty as plain scatter
                no_unc = ~valid_unc
                if no_unc.any():
                    ax.scatter(
                        filt_log_E[no_unc], log_xs[no_unc],
                        c=INLIER_COLOR, s=INLIER_SIZE, alpha=0.4,
                        edgecolors='none',
                        label=f'No uncertainty ({no_unc.sum():,})',
                    )
            else:
                # No valid uncertainties - fall back to plain scatter
                ax.scatter(
                    filt_log_E, log_xs,
                    c=INLIER_COLOR, s=INLIER_SIZE * 1.5, alpha=0.6,
                    edgecolors='white', linewidths=0.3,
                    label=f'Filtered ({n_filtered:,})',
                )
        else:
            # No uncertainty column - plain scatter
            ax.scatter(
                filtered['log_E'], filtered['log_sigma'],
                c=INLIER_COLOR, s=INLIER_SIZE * 1.5, alpha=0.6,
                edgecolors='white', linewidths=0.3,
                label=f'Filtered ({n_filtered:,})',
            )

        # ── Energy region vertical spans ─────────────────────────────────
        y_lo, y_hi = ax.get_ylim()
        filt_log_E = filtered['log_E'].values
        for name, lo_ev, hi_ev, color in ENERGY_REGIONS:
            lo_log = np.log10(max(lo_ev, 1e-30))
            hi_log = np.log10(hi_ev)
            x_min, x_max = filt_log_E.min(), filt_log_E.max()
            if hi_log < x_min or lo_log > x_max:
                continue
            lo_clipped = max(lo_log, x_min)
            hi_clipped = min(hi_log, x_max)
            ax.axvspan(lo_clipped, hi_clipped, alpha=0.06, color=color, zorder=0)
            mid = (lo_clipped + hi_clipped) / 2
            ax.text(
                mid, y_hi, name, ha='center', va='bottom', fontsize=7,
                alpha=0.6, family='sans-serif', style='italic',
            )
        ax.set_ylim(y_lo, y_hi)

        # Style
        ax.set_xlabel(r'$\log_{10}(E)$ [eV]', fontsize=11, family='sans-serif')
        ax.set_ylabel(r'$\log_{10}(\sigma)$ [b]', fontsize=11, family='sans-serif')
        ax.set_title(f'Filtered Data ({n_filtered:,} pts)', fontsize=12,
                     fontweight='bold', family='sans-serif')
        ax.legend(loc='best', fontsize=8, frameon=True, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # =====================================================================
    # Per-experiment scatter (for Color by experiment mode)
    # =====================================================================

    def _draw_experiment_scatter(
        self, ax: Axes, df: pd.DataFrame, threshold: float,
    ) -> None:
        """Draw scatter plot colored by experiment, highlighting discrepant ones.

        When "Color by experiment" is enabled:
        - Each experiment gets a distinct color from tab10 colormap
        - Discrepant experiments (experiment_outlier=True) get red X markers
        - Point outliers within experiments get larger size + black edge
        - Legend shows experiment IDs with checkmark/X status
        """
        log_E = df['log_E'].values
        log_sigma = df['log_sigma'].values
        z_scores = df['z_score'].values

        # Get experiment ID column
        exp_col = 'experiment_id' if 'experiment_id' in df.columns else 'Entry'
        exp_ids = df[exp_col].values

        # Get unique experiments
        unique_exps = df[exp_col].unique()
        n_experiments = len(unique_exps)

        # Use tab10 colormap for distinct experiment colors
        cmap = plt.cm.get_cmap('tab10')

        # Check for per-experiment outlier columns
        has_exp_outlier = 'experiment_outlier' in df.columns
        has_point_outlier = 'point_outlier' in df.columns

        # Track which experiments are discrepant for legend
        discrepant_set = set()
        if has_exp_outlier:
            discrepant_ids = df[df['experiment_outlier']][exp_col].unique()
            discrepant_set = set(discrepant_ids)

        # Plot each experiment separately
        legend_handles = []
        legend_labels = []

        for i, exp_id in enumerate(unique_exps):
            mask = exp_ids == exp_id
            color = cmap(i % 10)

            exp_log_E = log_E[mask]
            exp_log_sigma = log_sigma[mask]
            exp_z = z_scores[mask]

            is_discrepant = exp_id in discrepant_set

            # Determine marker style based on experiment status
            if is_discrepant:
                # Discrepant experiments: red X markers
                marker = 'X'
                base_color = OUTLIER_COLOR
                edge_color = 'darkred'
                size = OUTLIER_SIZE
                alpha = 0.9
            else:
                # Normal experiments: colored circles
                marker = 'o'
                base_color = color
                edge_color = 'none'
                size = INLIER_SIZE
                alpha = 0.6

            # Check for point outliers within this experiment
            if has_point_outlier:
                point_outlier_mask = df.loc[mask, 'point_outlier'].values
            else:
                # Fall back to z-score threshold
                point_outlier_mask = exp_z > threshold

            # Plot non-point-outlier points
            normal_mask = ~point_outlier_mask
            if normal_mask.any():
                scatter = ax.scatter(
                    exp_log_E[normal_mask], exp_log_sigma[normal_mask],
                    c=[base_color], s=size, alpha=alpha,
                    marker=marker, edgecolors=edge_color, linewidths=0.5,
                    zorder=6,
                )

            # Plot point outliers with larger size + black edge
            if point_outlier_mask.any():
                ax.scatter(
                    exp_log_E[point_outlier_mask], exp_log_sigma[point_outlier_mask],
                    c=[base_color], s=size * 2.5, alpha=0.9,
                    marker=marker, edgecolors='black', linewidths=1.5,
                    zorder=7,
                )

            # Build legend entry (limit to first 8 experiments to avoid clutter)
            if i < 8:
                # Truncate long experiment IDs
                exp_str = str(exp_id)
                if len(exp_str) > 10:
                    exp_str = exp_str[:10] + '..'

                if is_discrepant:
                    label = f'{exp_str} \u2717'  # ✗ mark
                    handle = plt.Line2D(
                        [0], [0], marker='X', color='w',
                        markerfacecolor=OUTLIER_COLOR, markersize=8,
                        markeredgecolor='darkred', markeredgewidth=0.5,
                        linestyle='None',
                    )
                else:
                    label = f'{exp_str} \u2713'  # ✓ mark
                    handle = plt.Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=6,
                        linestyle='None',
                    )
                legend_handles.append(handle)
                legend_labels.append(label)

        # Add "more experiments" indicator if needed
        if n_experiments > 8:
            more_label = f'... +{n_experiments - 8} more'
            more_handle = plt.Line2D(
                [0], [0], marker='', color='gray', linestyle='None',
            )
            legend_handles.append(more_handle)
            legend_labels.append(more_label)

        # Add legend for point outlier indicator
        if has_point_outlier or (z_scores > threshold).any():
            outlier_handle = plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor='gray', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5,
                linestyle='None',
            )
            legend_handles.append(outlier_handle)
            legend_labels.append('Point outlier (z > thr)')

        # Create legend with experiment info
        ax.legend(
            legend_handles, legend_labels,
            loc='upper right', fontsize=7, frameon=True,
            framealpha=0.9, ncol=2 if n_experiments > 4 else 1,
            title='Experiments', title_fontsize=8,
        )

    # =====================================================================
    # Statistics computation
    # =====================================================================

    def _compute_statistics(
        self, df: pd.DataFrame, threshold: float,
    ) -> dict:
        total = len(df)
        z_scores = df['z_score'].values
        n_outliers = int((z_scores > threshold).sum())
        n_inliers = total - n_outliers

        # Compute remaining after current filter settings
        remaining_mask = np.ones(total, dtype=bool)
        if self._w_exclude_point_outliers.value:
            remaining_mask &= z_scores <= threshold
        if self._w_exclude_discrepant.value and 'experiment_outlier' in df.columns:
            remaining_mask &= ~df['experiment_outlier'].values
        if self._w_exclude_raw.value:
            remaining_mask &= ~self._is_raw(df)
        n_remaining = int(remaining_mask.sum())

        stats = {
            'total': total,
            'n_inliers': n_inliers,
            'n_outliers': n_outliers,
            'pct_inliers': 100.0 * n_inliers / total if total > 0 else 0.0,
            'pct_outliers': 100.0 * n_outliers / total if total > 0 else 0.0,
            'n_remaining': n_remaining,
            'pct_remaining': 100.0 * n_remaining / total if total > 0 else 0.0,
            'z_min': float(z_scores.min()),
            'z_max': float(z_scores.max()),
        }

        # Add per-experiment stats if available
        if self._has_experiment_outlier and 'experiment_outlier' in df.columns:
            exp_col = 'experiment_id' if 'experiment_id' in df.columns else 'Entry'
            n_experiments = df[exp_col].nunique()
            discrepant_ids = df[df['experiment_outlier']][exp_col].unique()
            n_discrepant = len(discrepant_ids)
            n_points_in_discrepant = df['experiment_outlier'].sum()

            stats['n_experiments'] = n_experiments
            stats['n_discrepant'] = n_discrepant
            stats['n_points_in_discrepant'] = int(n_points_in_discrepant)
        else:
            # No per-experiment data
            exp_col = 'Entry' if 'Entry' in df.columns else None
            if exp_col:
                stats['n_experiments'] = df[exp_col].nunique()
            stats['n_discrepant'] = None
            stats['n_points_in_discrepant'] = None

        return stats
