"""Plotting utilities for NUCML-Next visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PlottingHelpers:
    """Visualization tools for educational demonstrations."""

    @staticmethod
    def plot_staircase_comparison(energies, true_xs, dt_xs, xgb_xs, gnn_xs=None):
        """Compare predictions showing staircase effect."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(energies, true_xs, 'b-', lw=3, label='Ground Truth', alpha=0.7)
        ax.plot(energies, dt_xs, 'r--', lw=2, label='Decision Tree (Stairs)', alpha=0.6)
        ax.plot(energies, xgb_xs, 'g-', lw=2, label='XGBoost', alpha=0.7)
        if gnn_xs is not None:
            ax.plot(energies, gnn_xs, 'm-', lw=2.5, label='GNN-Transformer (Smooth)', alpha=0.8)
        ax.set_xlabel('Energy (eV)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross Section (barns)', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: Classical ML vs Deep Learning', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax
