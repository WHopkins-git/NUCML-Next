"""
Utilities Module
================

Visualization and helper functions for nuclear data analysis.

Key Components:
    PlottingHelpers: Comparative visualizations (XGBoost vs GNN-Transformer)
    MetricsCalculator: Compute MSE, MAE, physics violations
    ChartOfNuclides: Visualize the Chart of Nuclides as a graph

Educational Utilities:
    - Plot "staircase effect" from Decision Trees
    - Overlay smooth GNN-Transformer predictions
    - Visualize sensitivity coefficients
    - Show physics constraint violations
"""

from nucml_next.utils.plotting_helpers import PlottingHelpers
from nucml_next.utils.metrics_calculator import MetricsCalculator
from nucml_next.utils.chart_of_nuclides import ChartOfNuclides

__all__ = [
    "PlottingHelpers",
    "MetricsCalculator",
    "ChartOfNuclides",
]
