"""Metrics calculation utilities."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MetricsCalculator:
    """Calculate performance metrics."""

    @staticmethod
    def compute_metrics(y_true, y_pred, log_space=True):
        """Compute MSE, MAE, R2."""
        if log_space:
            y_true = np.log10(y_true + 1e-10)
            y_pred = np.log10(y_pred + 1e-10)
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
