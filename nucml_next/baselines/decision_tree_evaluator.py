"""
Decision Tree Evaluator
=======================

Educational baseline that demonstrates the "Staircase Effect."

Why Decision Trees Fail for Nuclear Data:
    1. Piecewise constant predictions → jagged steps
    2. No smoothness in resonance regions (unphysical)
    3. Poor extrapolation beyond training data
    4. Ignores continuity of physical processes

This is the pedagogical "villain" that motivates deep learning.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


class DecisionTreeEvaluator:
    """
    Decision Tree baseline for cross-section prediction.

    Configured to explicitly show the staircase effect:
    - Limited depth → exaggerated steps
    - Min samples per leaf → coarse predictions

    Educational Purpose:
        Students will see jagged predictions in resonance regions
        and understand why continuity matters in physics.
    """

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize Decision Tree evaluator.

        Args:
            max_depth: Maximum tree depth (lower = more stairs)
            min_samples_leaf: Minimum samples per leaf (higher = coarser steps)
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        self.is_trained = False
        self.feature_columns = None
        self.metrics = {}

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        test_size: float = 0.2,
        exclude_columns: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Train the decision tree model.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Name of target column
            test_size: Fraction of data for testing
            exclude_columns: Columns to exclude from features

        Returns:
            Dictionary of training metrics
        """
        # Prepare features and target
        if exclude_columns is None:
            exclude_columns = [target_column, 'Isotope', 'Reaction']

        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_columns
        ]

        X = df[self.feature_columns].values
        y = df[target_column].values

        # Log-transform target for better numerical stability
        y_log = np.log10(y + 1e-10)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.random_state
        )

        # Train model
        print(f"Training Decision Tree (max_depth={self.max_depth}, "
              f"min_samples_leaf={self.min_samples_leaf})...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Convert back from log space
        y_train_pred = 10 ** y_train_pred
        y_test_pred = 10 ** y_test_pred
        y_train_orig = 10 ** y_train
        y_test_orig = 10 ** y_test

        train_mse = mean_squared_error(y_train_orig, y_train_pred)
        test_mse = mean_squared_error(y_test_orig, y_test_pred)
        train_mae = mean_absolute_error(y_train_orig, y_train_pred)
        test_mae = mean_absolute_error(y_test_orig, y_test_pred)

        self.metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'num_leaves': self.model.get_n_leaves(),
            'tree_depth': self.model.get_depth(),
        }

        self.is_trained = True

        print(f"✓ Training complete!")
        print(f"  Test MSE: {test_mse:.4e}")
        print(f"  Test MAE: {test_mae:.4e}")
        print(f"  Tree depth: {self.metrics['tree_depth']}")
        print(f"  Number of leaves: {self.metrics['num_leaves']}")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Input features (same format as training)

        Returns:
            Predicted cross sections

        Note:
            The predictions will exhibit the "staircase effect"
            - especially visible when plotting vs. energy!
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X = df[self.feature_columns].values
        y_pred_log = self.model.predict(X)
        y_pred = 10 ** y_pred_log

        return y_pred

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained tree.

        Returns:
            DataFrame with feature importances sorted by importance

        Educational Note:
            This shows what the tree "thinks" is important.
            Often surprises students when physical features
            are ranked lower than expected!
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importances

    def predict_resonance_region(
        self,
        Z: int,
        A: int,
        mt_code: int,
        energy_range: Tuple[float, float],
        num_points: int = 1000,
        mode: str = 'naive',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cross section in a resonance region.

        This is specifically designed to show the staircase effect!

        Args:
            Z: Atomic number
            A: Mass number
            mt_code: Reaction type
            energy_range: (E_min, E_max) in eV
            num_points: Number of energy points
            mode: Feature mode ('naive' or 'physics')

        Returns:
            (energies, predictions) showing jagged steps

        Educational Purpose:
            Plot this result to show students why continuity matters.
            Real cross sections have smooth resonance curves,
            not discrete steps!
        """
        energies = np.linspace(energy_range[0], energy_range[1], num_points)

        # Build feature matrix
        if mode == 'naive':
            # One-hot encoding
            features = []
            for energy in energies:
                feat_dict = {'Z': Z, 'A': A, 'Energy': energy}
                # Add MT one-hot (assuming MT codes are known)
                for mt in [2, 16, 18, 102]:  # Common reactions
                    feat_dict[f'MT_{mt}'] = 1.0 if mt == mt_code else 0.0
                features.append(feat_dict)
            df = pd.DataFrame(features)

        else:  # physics mode
            features = []
            for energy in energies:
                feat_dict = {
                    'Z': Z,
                    'A': A,
                    'N': A - Z,
                    'Energy': np.log10(energy + 1.0),
                    'Q_Value': 0.0,  # Simplified
                    'Threshold': 0.0,
                    'Delta_Z': 0,
                    'Delta_A': 0,
                    'MT': mt_code / 100.0,
                }
                features.append(feat_dict)
            df = pd.DataFrame(features)

        # Predict
        predictions = self.predict(df)

        return energies, predictions

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.max_depth = model_data['max_depth']
        self.min_samples_leaf = model_data['min_samples_leaf']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
