"""
Decision Tree Evaluator - Research Baseline
============================================

Production-grade Decision Tree baseline with full pipeline integration.

Features:
- TransformationPipeline integration with configurable scalers
- Automatic particle emission vector handling (MT codes → 9 features)
- Hyperparameter optimization via Bayesian optimization
- Comprehensive feature importance analysis
- Robust handling of AME-enriched data

Educational Purpose:
    Demonstrates the "staircase effect" and motivates smooth physics-informed models.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from nucml_next.data.transformations import TransformationPipeline
from nucml_next.data.selection import TransformationConfig


class DecisionTreeEvaluator:
    """
    Decision Tree baseline for nuclear cross-section prediction.

    Integrates with NUCML-Next TransformationPipeline for:
    - Configurable log transforms (log₁₀, ln, log₂)
    - Multiple scaler types (standard, minmax, robust, none)
    - Automatic particle emission vector handling (MT → 9 features)
    - Reversible transformations for predictions

    Example:
        >>> from nucml_next.data import NucmlDataset, DataSelection
        >>>
        >>> # Load data with optimal configuration
        >>> selection = DataSelection(tiers=['A', 'B', 'C'])
        >>> dataset = NucmlDataset('data.parquet', selection=selection)
        >>> df = dataset.to_tabular(mode='tier')  # MT codes → particle vectors
        >>>
        >>> # Train with automatic transformation
        >>> evaluator = DecisionTreeEvaluator()
        >>> metrics = evaluator.train(df, pipeline=dataset.get_transformation_pipeline())
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize Decision Tree evaluator.

        Args:
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf node
            min_samples_split: Minimum samples to split a node
            min_impurity_decrease: Minimum impurity decrease for split
            max_features: Features per split ('sqrt', 'log2', None)
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=random_state,
        )

        self.is_trained = False
        self.feature_columns = None
        self.pipeline = None
        self.metrics = {}

    def optimize_hyperparameters(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        energy_column: str = 'Energy',
        exclude_columns: Optional[list] = None,
        pipeline: Optional[TransformationPipeline] = None,
        transformation_config: Optional[TransformationConfig] = None,
        max_evals: int = 100,
        cv_folds: int = 3,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Target column name
            energy_column: Energy column name
            exclude_columns: Columns to exclude from features
            pipeline: Pre-configured TransformationPipeline (recommended)
            transformation_config: Config for new pipeline (if pipeline=None)
            max_evals: Maximum optimization iterations
            cv_folds: Cross-validation folds
            test_size: Test set fraction
            verbose: Print progress

        Returns:
            Dictionary with best_params, best_cv_score, test_mse, trials
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError(
                "hyperopt required for optimization. Install: pip install hyperopt"
            )

        if verbose:
            print("\n" + "=" * 80)
            print("HYPERPARAMETER OPTIMIZATION - Decision Tree")
            print("=" * 80)
            print(f"Dataset size: {len(df):,} samples")
            print(f"Max evaluations: {max_evals}")
            print(f"Cross-validation folds: {cv_folds}")
            print()

        # Prepare features
        if exclude_columns is None:
            exclude_columns = [target_column, 'Uncertainty', 'Entry', 'MT']

        # Auto-detect numeric columns (includes sparse particle vectors)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sparse_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)]
        all_numeric = list(set(numeric_cols + sparse_cols))
        feature_columns = [col for col in all_numeric if col not in exclude_columns]

        # Remove energy column - will be transformed separately
        if energy_column in feature_columns:
            feature_columns.remove(energy_column)

        non_numeric = [col for col in df.columns if col not in all_numeric and col not in exclude_columns]
        if len(non_numeric) > 0 and verbose:
            print(f"⚠️  Excluding {len(non_numeric)} non-numeric columns:")
            for col in non_numeric[:5]:
                print(f"    - {col} (dtype: {df[col].dtype})")
            if len(non_numeric) > 5:
                print(f"    ... and {len(non_numeric) - 5} more")
            print()

        X_features = df[feature_columns]
        y = df[target_column]
        energy = df[energy_column] if energy_column in df.columns else None

        # Create or use pipeline
        if pipeline is None:
            if transformation_config is None:
                transformation_config = TransformationConfig()
            pipeline = TransformationPipeline(config=transformation_config)
            pipeline.fit(X_features, y, energy, feature_columns=feature_columns)

        # Transform data
        X_transformed = pipeline.transform(X_features, energy)
        y_transformed = pipeline.transform_target(y)

        # Handle inf/NaN (critical for AME-enriched data)
        X_arr = X_transformed[feature_columns].values
        y_arr = y_transformed.values

        finite_mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
        if not finite_mask.all():
            n_invalid = (~finite_mask).sum()
            if verbose:
                print(f"⚠️  Removing {n_invalid:,} rows with inf/NaN ({n_invalid/len(X_arr)*100:.2f}%)")
            X_arr = X_arr[finite_mask]
            y_arr = y_arr[finite_mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=42
        )

        # Hyperparameter search space
        space = {
            'max_depth': hp.quniform('max_depth', 5, 30, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 50, 1),
            'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-8), np.log(1e-2)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        }

        def objective(params):
            params['max_depth'] = int(params['max_depth'])
            params['min_samples_split'] = int(params['min_samples_split'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])

            model = DecisionTreeRegressor(
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                min_impurity_decrease=params['min_impurity_decrease'],
                max_features=params['max_features'],
                random_state=42,
            )

            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            return {'loss': -cv_scores.mean(), 'status': STATUS_OK, 'params': params}

        # Run optimization
        trials = Trials()
        if verbose:
            print("Starting Bayesian optimization...")
            print("-" * 80)

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=verbose,
            rstate=np.random.default_rng(42),
        )

        best_params = space_eval(space, best)
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        best_params['random_state'] = 42

        # Evaluate on test set
        final_model = DecisionTreeRegressor(**best_params)
        final_model.fit(X_train, y_train)
        y_test_pred = final_model.predict(X_test)

        # Inverse transform if using log
        if pipeline.config.log_target:
            y_test_pred = pipeline.inverse_transform_target(pd.Series(y_test_pred)).values
            y_test_orig = pipeline.inverse_transform_target(pd.Series(y_test)).values
        else:
            y_test_pred = y_test_pred
            y_test_orig = y_test

        test_mse = mean_squared_error(y_test_orig, y_test_pred)
        best_cv_score = -trials.best_trial['result']['loss']

        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE")
            print("=" * 80)
            print(f"Best CV MSE (transformed space): {best_cv_score:.6f}")
            print(f"Test MSE (original space): {test_mse:.4e}")
            print()
            print("Optimal Hyperparameters:")
            for key, value in best_params.items():
                if key != 'random_state':
                    print(f"  {key:25s}: {value}")
            print("=" * 80)

        return {
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'test_mse': test_mse,
            'trials': trials,
        }

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        energy_column: str = 'Energy',
        test_size: float = 0.2,
        exclude_columns: Optional[list] = None,
        pipeline: Optional[TransformationPipeline] = None,
        transformation_config: Optional[TransformationConfig] = None,
    ) -> Dict[str, float]:
        """
        Train the Decision Tree model with full pipeline integration.

        Args:
            df: Training data with particle emission vectors (from to_tabular())
            target_column: Target column name
            energy_column: Energy column name
            test_size: Test set fraction
            exclude_columns: Columns to exclude
            pipeline: Pre-configured pipeline (recommended)
            transformation_config: Config for new pipeline (if pipeline=None)

        Returns:
            Training metrics dictionary
        """
        # Prepare features
        if exclude_columns is None:
            exclude_columns = [target_column, 'Uncertainty', 'Entry', 'MT']

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sparse_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)]
        all_numeric = list(set(numeric_cols + sparse_cols))
        self.feature_columns = [col for col in all_numeric if col not in exclude_columns]

        if energy_column in self.feature_columns:
            self.feature_columns.remove(energy_column)

        X_features = df[self.feature_columns]
        y = df[target_column]
        energy = df[energy_column] if energy_column in df.columns else None

        # Create or use pipeline
        if pipeline is None:
            if transformation_config is None:
                transformation_config = TransformationConfig()
            pipeline = TransformationPipeline(config=transformation_config)
            pipeline.fit(X_features, y, energy, feature_columns=self.feature_columns)

        self.pipeline = pipeline

        # Transform data
        X_transformed = pipeline.transform(X_features, energy)
        y_transformed = pipeline.transform_target(y)

        # Extract arrays
        X_arr = X_transformed[self.feature_columns].values
        y_arr = y_transformed.values

        # Handle inf/NaN
        finite_mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
        if not finite_mask.all():
            n_invalid = (~finite_mask).sum()
            print(f"  ⚠️  Removing {n_invalid:,} rows with inf/NaN ({n_invalid/len(X_arr)*100:.2f}%)")
            X_arr = X_arr[finite_mask]
            y_arr = y_arr[finite_mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=self.random_state
        )

        # Train model
        print(f"Training Decision Tree (max_depth={self.max_depth}, "
              f"min_samples_leaf={self.min_samples_leaf})...")
        self.model.fit(X_train, y_train)

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Inverse transform if using log
        if self.pipeline.config.log_target:
            y_train_pred = self.pipeline.inverse_transform_target(pd.Series(y_train_pred)).values
            y_test_pred = self.pipeline.inverse_transform_target(pd.Series(y_test_pred)).values
            y_train_orig = self.pipeline.inverse_transform_target(pd.Series(y_train)).values
            y_test_orig = self.pipeline.inverse_transform_target(pd.Series(y_test)).values
        else:
            y_train_orig = y_train
            y_test_orig = y_test

        # Metrics
        train_mse = mean_squared_error(y_train_orig, y_train_pred)
        test_mse = mean_squared_error(y_test_orig, y_test_pred)
        train_mae = mean_absolute_error(y_train_orig, y_train_pred)
        test_mae = mean_absolute_error(y_test_orig, y_test_pred)
        test_r2 = r2_score(y_test_orig, y_test_pred)

        self.metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'num_leaves': self.model.get_n_leaves(),
            'tree_depth': self.model.get_depth(),
        }

        self.is_trained = True

        print(f"✓ Training complete!")
        print(f"  Test MSE: {test_mse:.4e}")
        print(f"  Test MAE: {test_mae:.4e}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Tree depth: {self.metrics['tree_depth']}")
        print(f"  Leaves: {self.metrics['num_leaves']}")

        return self.metrics

    def predict(self, df: pd.DataFrame, energy_column: str = 'Energy') -> np.ndarray:
        """
        Predict cross-sections with automatic transformation.

        Args:
            df: Input features (must match training format)
            energy_column: Energy column name

        Returns:
            Predicted cross-sections in original scale
        """
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model must be trained before prediction")

        X_features = df[self.feature_columns]
        energy = df[energy_column] if energy_column in df.columns else None

        # Transform
        X_transformed = self.pipeline.transform(X_features, energy)
        X_arr = X_transformed[self.feature_columns].values

        # Predict in transformed space
        y_pred_transformed = self.model.predict(X_arr)

        # Inverse transform
        if self.pipeline.config.log_target:
            y_pred = self.pipeline.inverse_transform_target(pd.Series(y_pred_transformed)).values
        else:
            y_pred = y_pred_transformed

        return y_pred

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with features and importance scores

        Note:
            Particle emission features (out_n, out_p, etc.) will appear
            instead of raw MT codes, showing physics-informed encoding.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importances

    def save(self, filepath: str) -> None:
        """Save model and pipeline to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'hyperparams': {
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'min_samples_split': self.min_samples_split,
            }
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model and pipeline from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.pipeline = model_data['pipeline']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']

        hyperparams = model_data.get('hyperparams', {})
        self.max_depth = hyperparams.get('max_depth', 10)
        self.min_samples_leaf = hyperparams.get('min_samples_leaf', 10)
        self.min_samples_split = hyperparams.get('min_samples_split', 2)

        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
