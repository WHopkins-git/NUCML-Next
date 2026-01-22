"""
XGBoost Evaluator
=================

Enhanced baseline using gradient boosting.

Compared to Decision Trees:
    - Better accuracy (ensemble of trees)
    - Still exhibits staircase effect (milder)
    - Still poor extrapolation
    - Can't learn smooth resonance curves

Educational Purpose:
    Show that even state-of-the-art classical ML can't match
    physics-informed deep learning for smooth predictions.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib


class XGBoostEvaluator:
    """
    XGBoost baseline for cross-section prediction.

    Better than Decision Trees but still fundamentally limited:
    - Ensemble smooths out some steps
    - Feature importance reveals physics insights
    - But still can't guarantee smoothness or physical constraints
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost evaluator.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Fast histogram-based method
        }

        self.model = xgb.XGBRegressor(**self.params)
        self.is_trained = False
        self.feature_columns = None
        self.metrics = {}

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        test_size: float = 0.2,
        exclude_columns: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 10,
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Name of target column
            test_size: Fraction of data for testing
            exclude_columns: Columns to exclude from features
            early_stopping_rounds: Enable early stopping

        Returns:
            Dictionary of training metrics
        """
        # Prepare features and target
        if exclude_columns is None:
            exclude_columns = [target_column, 'Isotope', 'Reaction']

        # Filter out non-numeric columns (includes pandas sparse arrays)
        numeric_columns = df.select_dtypes(include=[np.number, pd.SparseDtype]).columns
        self.feature_columns = [
            col for col in numeric_columns
            if col not in exclude_columns
        ]

        # Handle sparse DataFrames efficiently (avoid memory explosion)
        X_df = df[self.feature_columns]
        is_sparse = any(isinstance(dtype, pd.SparseDtype) for dtype in X_df.dtypes)

        if is_sparse:
            # XGBoost natively supports scipy sparse matrices
            import scipy.sparse as sp
            X = sp.csr_matrix(X_df.sparse.to_coo())
            print(f"  → Using sparse matrix format (memory efficient)")
        else:
            X = X_df.values

        y = df[target_column].values

        # Log-transform target for better numerical stability
        y_log = np.log10(y + 1e-10)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.params['random_state']
        )

        # Train model with optional early stopping
        print(f"Training XGBoost ({self.params['n_estimators']} trees, "
              f"max_depth={self.params['max_depth']})...")

        eval_set = [(X_test, y_test)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

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

        # Calculate R² score
        train_r2 = 1 - (np.sum((y_train_orig - y_train_pred) ** 2) /
                        np.sum((y_train_orig - y_train_orig.mean()) ** 2))
        test_r2 = 1 - (np.sum((y_test_orig - y_test_pred) ** 2) /
                       np.sum((y_test_orig - y_test_orig.mean()) ** 2))

        self.metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
        }

        self.is_trained = True

        print(f"✓ Training complete!")
        print(f"  Test MSE: {test_mse:.4e}")
        print(f"  Test MAE: {test_mae:.4e}")
        print(f"  Test R²: {test_r2:.4f}")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Input features (same format as training)

        Returns:
            Predicted cross sections
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Handle sparse DataFrames efficiently
        X_df = df[self.feature_columns]
        is_sparse = any(isinstance(dtype, pd.SparseDtype) for dtype in X_df.dtypes)

        if is_sparse:
            # Convert to scipy sparse matrix
            import scipy.sparse as sp
            X = sp.csr_matrix(X_df.sparse.to_coo())
        else:
            X = X_df.values

        y_pred_log = self.model.predict(X)
        y_pred = 10 ** y_pred_log

        return y_pred

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            importance_type: 'gain', 'weight', or 'cover'
                - gain: Average gain across all splits (recommended)
                - weight: Number of times feature is used
                - cover: Average coverage of splits

        Returns:
            DataFrame with feature importances

        Educational Note:
            Compare this with physics intuition:
            - Energy should be highly important
            - Q-value and threshold should matter for physics mode
            - If Z/A dominate, model is just memorizing isotopes!
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        # Get importance scores
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)

        # Map to feature names
        feature_importance = []
        for i, feat_name in enumerate(self.feature_columns):
            # XGBoost uses 'f0', 'f1', ... as feature names internally
            xgb_feat_name = f'f{i}'
            importance = importance_dict.get(xgb_feat_name, 0.0)
            feature_importance.append({'Feature': feat_name, 'Importance': importance})

        df_importance = pd.DataFrame(feature_importance).sort_values('Importance', ascending=False)

        return df_importance

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

        Shows that XGBoost is smoother than Decision Tree
        but still not physics-compliant.

        Args:
            Z: Atomic number
            A: Mass number
            mt_code: Reaction type
            energy_range: (E_min, E_max) in eV
            num_points: Number of energy points
            mode: Feature mode ('naive' or 'physics')

        Returns:
            (energies, predictions)
        """
        energies = np.linspace(energy_range[0], energy_range[1], num_points)

        # Build feature matrix
        if mode == 'naive':
            # One-hot encoding
            features = []
            for energy in energies:
                feat_dict = {'Z': Z, 'A': A, 'Energy': energy}
                # Add MT one-hot
                for mt in [2, 16, 18, 102]:
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
                    'Q_Value': 0.0,
                    'Threshold': 0.0,
                    'Delta_Z': 0,
                    'Delta_A': 0,
                    'MT': mt_code / 100.0,
                }
                features.append(feat_dict)
            df = pd.DataFrame(features)

        # Ensure all training features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

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
            'params': self.params,
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.params = model_data['params']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")

    def analyze_predictions(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
    ) -> Dict[str, Any]:
        """
        Analyze prediction quality across different regimes.

        Args:
            df: Test data with ground truth
            target_column: Name of target column

        Returns:
            Dictionary with analysis results

        Educational Analysis:
            - Low energy (thermal): XGBoost usually good
            - Resonance region: May show artifacts
            - High energy (fast): Extrapolation poor
        """
        predictions = self.predict(df)
        y_true = df[target_column].values

        # Split into energy regimes
        thermal_mask = df['Energy'] < 1.0  # < 1 eV
        resonance_mask = (df['Energy'] >= 1.0) & (df['Energy'] < 1e3)  # 1 eV - 1 keV
        fast_mask = df['Energy'] >= 1e3  # > 1 keV

        analysis = {}

        for name, mask in [('thermal', thermal_mask), ('resonance', resonance_mask), ('fast', fast_mask)]:
            if mask.sum() > 0:
                y_true_regime = y_true[mask]
                y_pred_regime = predictions[mask]

                mse = mean_squared_error(y_true_regime, y_pred_regime)
                mae = mean_absolute_error(y_true_regime, y_pred_regime)
                max_error = np.max(np.abs(y_true_regime - y_pred_regime))

                analysis[name] = {
                    'mse': mse,
                    'mae': mae,
                    'max_error': max_error,
                    'num_points': mask.sum(),
                }

        return analysis
