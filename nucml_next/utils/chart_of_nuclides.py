"""Chart of Nuclides visualization."""
import matplotlib.pyplot as plt
import numpy as np

class ChartOfNuclides:
    """Visualize the Chart of Nuclides as a graph."""

    @staticmethod
    def plot_chart(isotopes_df, highlight_isotopes=None):
        """Plot Z vs N chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(isotopes_df['N'], isotopes_df['Z'], alpha=0.6, s=50)
        if highlight_isotopes:
            for Z, A in highlight_isotopes:
                N = A - Z
                ax.scatter([N], [Z], color='red', s=200, marker='*', zorder=5)
        ax.set_xlabel('Neutron Number (N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Atomic Number (Z)', fontsize=12, fontweight='bold')
        ax.set_title('Chart of Nuclides', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        return fig, ax
