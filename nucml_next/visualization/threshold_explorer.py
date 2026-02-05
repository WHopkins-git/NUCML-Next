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
OUTLIER_COLOR = '#DC143C'      # crimson
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
            cols = [c for c in _REQUIRED_COLUMNS]
            self._df = pd.read_parquet(str(data), columns=cols)
        else:
            self._df = data.copy()

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

        # Wire observers
        self._w_z.observe(self._on_z_change, names='value')
        self._w_a.observe(self._on_a_change, names='value')
        self._w_mt.observe(self._on_mt_change, names='value')
        self._w_threshold.observe(self._on_threshold_change, names='value')

        controls = widgets.HBox(
            [self._w_z, self._w_a, self._w_mt, self._w_threshold],
            layout=widgets.Layout(margin='0 0 10px 0'),
        )
        self._ui = widgets.VBox([controls, self._output])

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

    # =====================================================================
    # Public API
    # =====================================================================

    def show(self) -> None:
        """Display the interactive explorer in a Jupyter notebook."""
        display(self._ui)

    # =====================================================================
    # Plot orchestration
    # =====================================================================

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

            fig = plt.figure(figsize=(self._figsize[0], self._figsize[1] + 1.5))

            gs = gridspec.GridSpec(
                2, 3,
                width_ratios=[1, 1, 0.12],
                height_ratios=[1, 0.08],
                hspace=0.35,
                wspace=0.35,
            )

            ax_prob = fig.add_subplot(gs[0, 0])
            ax_band = fig.add_subplot(gs[0, 1])
            ax_hist = fig.add_subplot(gs[0, 2])
            ax_stats = fig.add_subplot(gs[1, :])

            # Draw panels
            self._draw_probability_surface(ax_prob, df, threshold)
            self._draw_zscore_bands(ax_band, df, threshold)

            # ── Marginal z-score histogram ───────────────────────────────
            z_scores = df['z_score'].values
            z_max_hist = max(z_scores.max(), threshold + 1)
            bins = np.linspace(0, z_max_hist, 40)
            ax_hist.hist(
                z_scores, bins=bins, orientation='horizontal',
                color=GP_MEAN_COLOR, alpha=0.7, edgecolor='white',
                linewidth=0.5,
            )
            ax_hist.axhline(
                y=threshold, color=OUTLIER_COLOR, linestyle='--',
                linewidth=1.5, label=f'z = {threshold:.1f}',
            )
            ax_hist.set_xlabel('Count', fontsize=9, family='sans-serif')
            ax_hist.set_ylabel('z-score', fontsize=9, family='sans-serif')
            ax_hist.set_title('z-score dist.', fontsize=10,
                              fontweight='bold', family='sans-serif')
            ax_hist.legend(fontsize=7, frameon=False)
            ax_hist.spines['top'].set_visible(False)
            ax_hist.spines['right'].set_visible(False)

            # ── Statistics text bar ──────────────────────────────────────
            ax_stats.axis('off')
            stats_text = (
                f"Total: {stats['total']:,}   |   "
                f"Inliers: {stats['n_inliers']:,} ({stats['pct_inliers']:.1f}%)   |   "
                f"Outliers: {stats['n_outliers']:,} ({stats['pct_outliers']:.1f}%)   |   "
                f"z-score range: [{stats['z_min']:.2f}, {stats['z_max']:.2f}]"
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
    # Probability surface panel
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

        # Scatter: inliers vs outliers
        inlier_mask = z_scores <= threshold
        outlier_mask = ~inlier_mask

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
        ax.set_title('Data + Z-Score Bands', fontsize=12,
                     fontweight='bold', family='sans-serif')
        ax.legend(loc='best', fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

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
        return {
            'total': total,
            'n_inliers': n_inliers,
            'n_outliers': n_outliers,
            'pct_inliers': 100.0 * n_inliers / total if total > 0 else 0.0,
            'pct_outliers': 100.0 * n_outliers / total if total > 0 else 0.0,
            'z_min': float(z_scores.min()),
            'z_max': float(z_scores.max()),
        }
