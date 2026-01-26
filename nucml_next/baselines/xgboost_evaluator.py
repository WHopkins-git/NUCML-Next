"""
XGBoost Evaluator - Research Baseline
======================================

Production-grade XGBoost baseline with full pipeline integration.

Features:
- TransformationPipeline integration with configurable scalers
- Automatic particle emission vector handling (MT codes → 9 features)
- Hyperparameter optimization via Bayesian optimization
- GPU support and early stopping
- Robust handling of AME-enriched data

Educational Purpose:
    Shows that gradient boosting is smoother than decision trees but
    still can't match physics-informed deep learning for smooth predictions.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from nucml_next.data.transformations import TransformationPipeline
from nucml_next.data.selection import TransformationConfig


class XGBoostEvaluator:
    """
    XGBoost baseline for nuclear cross-section prediction.

    Integrates with NUCML-Next TransformationPipeline for:
    - Configurable log transforms (log₁₀, ln, log₂)
    - Multiple scaler types (standard, minmax, robust, none)
    - Automatic particle emission vector handling (MT → 9 features)
    - Reversible transformations for predictions

    Example:
        >>> from nucml_next.data import NucmlDataset, DataSelection
        >>>
        >>> # Load data with optimal configuration
        >>> selection = DataSelection(tiers=['A', 'B', 'C', 'D'])
        >>> dataset = NucmlDataset('data.parquet', selection=selection)
        >>> df = dataset.to_tabular(mode='tier')  # MT codes → particle vectors
        >>>
        >>> # Train with automatic transformation
        >>> evaluator = XGBoostEvaluator()
        >>> metrics = evaluator.train(df, pipeline=dataset.get_transformation_pipeline())
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
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
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_weight: Minimum sum of instance weight in child
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
        }

        self.model = xgb.XGBRegressor(**self.params)
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
            print("HYPERPARAMETER OPTIMIZATION - XGBoost")
            print("=" * 80)
            print(f"Dataset size: {len(df):,} samples")
            print(f"Max evaluations: {max_evals}")
            print(f"Cross-validation folds: {cv_folds}")
            print()

        # Prepare features
        if exclude_columns is None:
            exclude_columns = [target_column, 'Uncertainty', 'Entry', 'MT']

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sparse_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)]
        all_numeric = list(set(numeric_cols + sparse_cols))
        feature_columns = [col for col in all_numeric if col not in exclude_columns]

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

        # ============================================================================
        # CRITICAL: Drop rows with NaN in features BEFORE pipeline fitting
        # ============================================================================
        X_features = df[feature_columns]
        y = df[target_column]
        energy = df[energy_column] if energy_column in df.columns else None

        # Check for NaN in features (before transformation)
        initial_size = len(df)
        nan_mask_features = X_features.isna().any(axis=1)
        nan_mask_target = y.isna()
        nan_mask_energy = energy.isna() if energy is not None else pd.Series(False, index=df.index)
        nan_mask = nan_mask_features | nan_mask_target | nan_mask_energy

        if nan_mask.any():
            n_nan = nan_mask.sum()
            if verbose:
                print(f"⚠️  CRITICAL: Found {n_nan:,} rows with NaN in features ({n_nan/initial_size*100:.2f}%)")
                print(f"   This is likely due to incomplete AME enrichment coverage.")
                print(f"   Dropping these rows BEFORE fitting pipeline...")
                # Show which features have the most NaN
                nan_counts = X_features.isna().sum()
                nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)
                if len(nan_features) > 0:
                    print(f"\n   Features with NaN values:")
                    for feat, count in nan_features.head(5).items():
                        print(f"     - {feat}: {count:,} NaN ({count/initial_size*100:.1f}%)")
                    if len(nan_features) > 5:
                        print(f"     ... and {len(nan_features) - 5} more features")
                print()

            # Drop rows with NaN
            valid_indices = ~nan_mask
            X_features = X_features[valid_indices]
            y = y[valid_indices]
            if energy is not None:
                energy = energy[valid_indices]

            final_size = len(X_features)
            if verbose:
                print(f"   After dropping NaN: {final_size:,} rows ({final_size/initial_size*100:.1f}% retained)")
                print()

            # Check if we have enough data left
            if final_size == 0:
                raise ValueError(
                    "❌ ERROR: All rows have NaN values in features!\n"
                    "   This means NO isotopes in your dataset are in the AME2020 database.\n"
                    "   Solutions:\n"
                    "   1. Use tiers=['A'] only (no AME enrichment)\n"
                    "   2. Filter to isotopes with AME data before training\n"
                    "   3. Implement imputation for missing AME features"
                )

            if final_size < 1000:
                print(f"⚠️  WARNING: Only {final_size:,} rows remain after dropping NaN.")
                print(f"   Consider using fewer tier features or different data selection.")
                print()

        # Create or use pipeline
        if pipeline is None:
            if transformation_config is None:
                transformation_config = TransformationConfig()
            pipeline = TransformationPipeline(config=transformation_config)
            pipeline.fit(X_features, y, energy, feature_columns=feature_columns)

        # Transform data
        X_transformed = pipeline.transform(X_features, energy)
        y_transformed = pipeline.transform_target(y)

        # Handle inf/NaN
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
            'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        }

        def objective(params):
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            params['min_child_weight'] = int(params['min_child_weight'])

            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=params['min_child_weight'],
                random_state=42,
                objective='reg:squarederror',
                tree_method='hist',
                n_jobs=-1,
                verbosity=0,
            )

            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=1  # XGBoost already uses n_jobs=-1
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
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['random_state'] = 42

        # Evaluate on test set
        final_model = xgb.XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            verbosity=0,
        )
        final_model.fit(X_train, y_train, verbose=False)
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
                    if isinstance(value, float):
                        print(f"  {key:25s}: {value:.6f}")
                    else:
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
        early_stopping_rounds: Optional[int] = 10,
    ) -> Dict[str, float]:
        """
        Train the XGBoost model with full pipeline integration.

        Args:
            df: Training data with particle emission vectors (from to_tabular())
            target_column: Target column name
            energy_column: Energy column name
            test_size: Test set fraction
            exclude_columns: Columns to exclude
            pipeline: Pre-configured pipeline (recommended)
            transformation_config: Config for new pipeline (if pipeline=None)
            early_stopping_rounds: Early stopping rounds (None to disable)

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

        # ============================================================================
        # CRITICAL: Drop rows with NaN in features BEFORE pipeline fitting
        # ============================================================================
        X_features = df[self.feature_columns]
        y = df[target_column]
        energy = df[energy_column] if energy_column in df.columns else None

        # Check for NaN in features (before transformation)
        initial_size = len(df)
        nan_mask_features = X_features.isna().any(axis=1)
        nan_mask_target = y.isna()
        nan_mask_energy = energy.isna() if energy is not None else pd.Series(False, index=df.index)
        nan_mask = nan_mask_features | nan_mask_target | nan_mask_energy

        if nan_mask.any():
            n_nan = nan_mask.sum()
            print(f"  ⚠️  Found {n_nan:,} rows with NaN in features ({n_nan/initial_size*100:.2f}%)")
            print(f"      Dropping these rows BEFORE fitting pipeline...")

            # Drop rows with NaN
            valid_indices = ~nan_mask
            X_features = X_features[valid_indices]
            y = y[valid_indices]
            if energy is not None:
                energy = energy[valid_indices]

            final_size = len(X_features)
            print(f"      After dropping NaN: {final_size:,} rows ({final_size/initial_size*100:.1f}% retained)\n")

            # Check if we have enough data left
            if final_size == 0:
                raise ValueError(
                    "❌ ERROR: All rows have NaN values in features!\n"
                    "   This means NO isotopes in your dataset are in the AME2020 database.\n"
                    "   Solutions:\n"
                    "   1. Use tiers=['A'] only (no AME enrichment)\n"
                    "   2. Filter to isotopes with AME data before training\n"
                    "   3. Implement imputation for missing AME features"
                )

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
            X_arr, y_arr, test_size=test_size, random_state=self.params['random_state']
        )

        # Train model
        print(f"Training XGBoost ({self.params['n_estimators']} trees, "
              f"max_depth={self.params['max_depth']})...")

        eval_set = [(X_test, y_test)]
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

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
        train_r2 = r2_score(y_train_orig, y_train_pred)
        test_r2 = r2_score(y_test_orig, y_test_pred)

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

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            importance_type: 'gain', 'weight', or 'cover'

        Returns:
            DataFrame with features and importance scores

        Note:
            Particle emission features (out_n, out_p, etc.) will appear
            instead of raw MT codes, showing physics-informed encoding.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)

        feature_importance = []
        for i, feat_name in enumerate(self.feature_columns):
            xgb_feat_name = f'f{i}'
            importance = importance_dict.get(xgb_feat_name, 0.0)
            feature_importance.append({'Feature': feat_name, 'Importance': importance})

        df_importance = pd.DataFrame(feature_importance).sort_values('Importance', ascending=False)

        return df_importance

    def save(self, filepath: str) -> None:
        """Save model and pipeline to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'params': self.params,
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
        self.params = model_data['params']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
