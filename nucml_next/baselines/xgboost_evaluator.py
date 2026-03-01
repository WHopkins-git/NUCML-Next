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
        # Hyperparameter search space bounds
        n_estimators_range: tuple = (50, 500),
        max_depth_range: tuple = (3, 15),
        learning_rate_range: tuple = (0.01, 0.3),
        subsample_range: tuple = (0.6, 1.0),
        # Subsampling for memory efficiency
        subsample_fraction: Optional[float] = None,
        subsample_max_samples: Optional[int] = None,
        # Uncertainty-based sample filtering
        use_uncertainty_weights: Optional[str] = None,
        missing_uncertainty_handling: str = 'median',
        uncertainty_column: str = 'Uncertainty',
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
            n_estimators_range: (min, max) for n_estimators. Default: (50, 500)
            max_depth_range: (min, max) for max_depth. Default: (3, 15)
            learning_rate_range: (min, max) for learning_rate. Default: (0.01, 0.3)
            subsample_range: (min, max) for subsample. Default: (0.6, 1.0)
            subsample_fraction: Fraction of data to use for hyperparameter search
                (0.0-1.0). Final model is trained on full data. Default: None (no subsampling).
            subsample_max_samples: Hard cap on samples for hyperparameter search.
                If both this and subsample_fraction are given, the smaller result is used.
            use_uncertainty_weights: Weight mode (None, 'xs', or 'both').
                When set with missing_uncertainty_handling='exclude', samples without
                valid uncertainty are removed before hyperparameter search.
            missing_uncertainty_handling: How to handle missing uncertainties
                ('median', 'equal', or 'exclude'). Only 'exclude' affects
                the hyperparameter search by removing rows.
            uncertainty_column: Column name for cross-section uncertainty.

        Returns:
            Dictionary with best_params, best_cv_score, test_mse, trials
        """
        # Apply uncertainty-based row exclusion before optimization
        from nucml_next.baselines._weights import normalize_weight_mode
        weight_mode = normalize_weight_mode(use_uncertainty_weights)

        if weight_mode is not None and missing_uncertainty_handling == 'exclude':
            if uncertainty_column in df.columns:
                n_before = len(df)
                valid_mask = df[uncertainty_column].notna() & (df[uncertainty_column] > 0)
                df = df[valid_mask].reset_index(drop=True)
                n_after = len(df)
                if verbose:
                    print(f"  Uncertainty filter (missing_uncertainty_handling='exclude'):")
                    print(f"    {n_before:,} → {n_after:,} samples "
                          f"({100*n_after/n_before:.1f}% retained)")
                    print()
            else:
                if verbose:
                    print(f"  WARNING: uncertainty column '{uncertainty_column}' not found, "
                          f"skipping exclusion filter")
                    print()

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

        # Keep energy column in features - it's critical for cross-section prediction
        # The pipeline will log-transform it, but it should remain a feature

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
        # Strategy: Impute NaN with 0 (column mean in standardized space), then remove inf
        X_arr = X_transformed[feature_columns].values
        y_arr = y_transformed.values

        # Count NaN/inf before processing
        nan_mask = np.isnan(X_arr)
        inf_mask = np.isinf(X_arr)
        nan_cell_count = nan_mask.sum()
        inf_cell_count = inf_mask.sum()
        rows_with_nan = nan_mask.any(axis=1).sum()
        rows_with_inf = inf_mask.any(axis=1).sum()
        y_invalid = ~np.isfinite(y_arr)

        if missing_uncertainty_handling == 'exclude' and rows_with_nan > 0:
            # Drop rows with any NaN/inf in features
            valid_rows = ~nan_mask.any(axis=1) & ~inf_mask.any(axis=1)
            n_before = len(X_arr)
            X_arr = X_arr[valid_rows]
            y_arr = y_arr[valid_rows]
            if verbose:
                n_samples, n_features = nan_mask.shape
                print(f"  Feature matrix: {n_samples:,} samples x {n_features} features")
                print(f"  NaN cells: {nan_cell_count:,} ({nan_cell_count / nan_mask.size * 100:.2f}% of cells)")
                print(f"  Rows with any NaN: {rows_with_nan:,} ({rows_with_nan / n_samples * 100:.2f}% of samples)")
                print(f"  Dropping {n_before - len(X_arr):,} rows with NaN/inf features "
                      f"(missing_uncertainty_handling='exclude')")
        else:
            if verbose and (nan_cell_count > 0 or inf_cell_count > 0):
                n_samples, n_features = X_arr.shape
                print(f"  Feature matrix: {n_samples:,} samples x {n_features} features")
                print(f"  NaN cells: {nan_cell_count:,} ({nan_cell_count / X_arr.size * 100:.2f}% of cells)")
                print(f"  Rows with any NaN: {rows_with_nan:,} ({rows_with_nan / n_samples * 100:.2f}% of samples)")
                print(f"  Inf cells: {inf_cell_count:,}")
                print(f"  Target invalid: {y_invalid.sum():,} samples")
                print(f"  Note: NaN values imputed to 0 (standardized mean); no rows removed")
            # Impute NaN
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e10, neginf=-1e10)

        # Remove rows with invalid targets
        valid_target_mask = np.isfinite(y_arr)
        if not valid_target_mask.all():
            n_invalid = (~valid_target_mask).sum()
            if verbose:
                print(f"  Removing {n_invalid:,} rows with invalid target ({n_invalid/len(y_arr)*100:.2f}%)")
            X_arr = X_arr[valid_target_mask]
            y_arr = y_arr[valid_target_mask]

        if len(X_arr) == 0:
            raise ValueError(
                "No valid samples after handling NaN/inf. Check that:\n"
                "1. Target column (CrossSection) has valid positive values\n"
                "2. Energy column has valid positive values\n"
                "3. Feature columns don't have all-NaN values"
            )

        # Apply subsampling AFTER data preparation
        n_samples = len(X_arr)
        use_subsample = subsample_fraction is not None or subsample_max_samples is not None

        if use_subsample:
            target_size = n_samples
            if subsample_fraction is not None:
                target_size = min(target_size, int(n_samples * subsample_fraction))
            if subsample_max_samples is not None:
                target_size = min(target_size, subsample_max_samples)

            if target_size < n_samples:
                rng = np.random.default_rng(42)
                subsample_idx = rng.choice(n_samples, size=target_size, replace=False)
                X_arr_search = X_arr[subsample_idx]
                y_arr_search = y_arr[subsample_idx]
                if verbose:
                    print(f"\nSubsampled for search: {n_samples:,} → {target_size:,} samples "
                          f"({100*target_size/n_samples:.1f}%)")
                    print(f"  NOTE: Final model will be trained on FULL data after search")
            else:
                X_arr_search = X_arr
                y_arr_search = y_arr
        else:
            X_arr_search = X_arr
            y_arr_search = y_arr

        # Split data (use subsampled data for search)
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr_search, y_arr_search, test_size=test_size, random_state=42
        )

        # Hyperparameter search space (using provided ranges)
        space = {
            'n_estimators': hp.quniform('n_estimators',
                                        n_estimators_range[0],
                                        n_estimators_range[1], 50),
            'max_depth': hp.quniform('max_depth',
                                     max_depth_range[0],
                                     max_depth_range[1], 1),
            'learning_rate': hp.loguniform('learning_rate',
                                           np.log(learning_rate_range[0]),
                                           np.log(learning_rate_range[1])),
            'subsample': hp.uniform('subsample',
                                    subsample_range[0],
                                    subsample_range[1]),
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

        # Evaluate on test set (use full data when subsampling was active)
        if use_subsample and target_size < n_samples:
            X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
                X_arr, y_arr, test_size=test_size, random_state=42
            )
        else:
            X_train_full, X_test_full = X_train, X_test
            y_train_full, y_test_full = y_train, y_test

        final_model = xgb.XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            verbosity=0,
        )
        final_model.fit(X_train_full, y_train_full, verbose=False)
        y_test_pred = final_model.predict(X_test_full)

        # Inverse transform if using log
        if pipeline.config.log_target:
            y_test_pred = pipeline.inverse_transform_target(pd.Series(y_test_pred)).values
            y_test_orig = pipeline.inverse_transform_target(pd.Series(y_test_full)).values
        else:
            y_test_pred = y_test_pred
            y_test_orig = y_test_full

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
        uncertainty_column: str = 'Uncertainty',
        energy_uncertainty_column: str = 'Energy_Uncertainty',
        test_size: float = 0.2,
        exclude_columns: Optional[list] = None,
        pipeline: Optional[TransformationPipeline] = None,
        transformation_config: Optional[TransformationConfig] = None,
        early_stopping_rounds: Optional[int] = 10,
        use_uncertainty_weights: Optional[str] = None,
        missing_uncertainty_handling: str = 'median',
    ) -> Dict[str, float]:
        """
        Train the XGBoost model with full pipeline integration.

        Args:
            df: Training data with particle emission vectors (from to_tabular())
            target_column: Target column name
            energy_column: Energy column name
            uncertainty_column: Cross-section uncertainty column name
            energy_uncertainty_column: Energy uncertainty column name
            test_size: Test set fraction
            exclude_columns: Columns to exclude
            pipeline: Pre-configured pipeline (recommended)
            transformation_config: Config for new pipeline (if pipeline=None)
            early_stopping_rounds: Early stopping rounds (None to disable)
            use_uncertainty_weights: Uncertainty weighting mode:

                * ``None`` – no weighting (default)
                * ``'xs'`` – inverse-variance weighting using
                  cross-section uncertainty only (w = 1/sigma_xs^2)
                * ``'both'`` – combined weighting using cross-section AND
                  energy uncertainty (w = 1/(sigma_xs^2 * sigma_E^2))
            missing_uncertainty_handling: How to handle samples with missing
                uncertainties when weighting is enabled:

                - ``'median'``: Assign median weight (default, keeps all samples)
                - ``'equal'``: Assign weight 1.0 (keeps all samples)
                - ``'exclude'``: Drop samples without valid uncertainty

        Returns:
            Training metrics dictionary

        Note on Sample Weighting:
            When use_uncertainty_weights is enabled:
            - 'xs': w_i = 1 / sigma_xs_i^2
            - 'both': w_i = 1 / (sigma_xs_i^2 * sigma_E_i^2)
            - Weights are normalized to mean=1 to avoid numerical issues
            - This is statistically correct for least-squares regression
        """
        from nucml_next.baselines._weights import normalize_weight_mode, compute_sample_weights

        # Validate parameters
        valid_handling = {'median', 'equal', 'exclude'}
        if missing_uncertainty_handling not in valid_handling:
            raise ValueError(f"missing_uncertainty_handling must be one of {valid_handling}")

        weight_mode = normalize_weight_mode(use_uncertainty_weights)

        # Prepare features
        if exclude_columns is None:
            exclude_columns = [target_column, 'Uncertainty', 'Energy_Uncertainty', 'Entry', 'MT']

        working_df = df.copy()

        numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
        sparse_cols = [col for col in working_df.columns if isinstance(working_df[col].dtype, pd.SparseDtype)]
        all_numeric = list(set(numeric_cols + sparse_cols))
        self.feature_columns = [col for col in all_numeric if col not in exclude_columns]

        # Keep energy column in features - it's critical for cross-section prediction
        # The pipeline will log-transform it, but it should remain a feature

        X_features = working_df[self.feature_columns]
        y = working_df[target_column]
        energy = working_df[energy_column] if energy_column in working_df.columns else None

        # Resolve transformation config early so we can pass log flags to weights
        if pipeline is not None:
            _tc = pipeline.config
        elif transformation_config is not None:
            _tc = transformation_config
        else:
            _tc = TransformationConfig()
            transformation_config = _tc

        # Compute sample weights from uncertainty (inverse variance weighting)
        # When log_target is True, uncertainties are propagated into log-space
        # so that weights reflect relative precision (δσ/σ) rather than
        # absolute precision (δσ). This prevents extreme weight ratios.
        sample_weights = compute_sample_weights(
            working_df,
            mode=weight_mode,
            uncertainty_column=uncertainty_column,
            energy_uncertainty_column=energy_uncertainty_column,
            missing_handling=missing_uncertainty_handling,
            target_column=target_column,
            log_target=_tc.log_target,
            energy_column=energy_column,
            log_energy=_tc.log_energy,
        )

        # Create or use pipeline
        if pipeline is None:
            pipeline = TransformationPipeline(config=transformation_config)
            pipeline.fit(X_features, y, energy, feature_columns=self.feature_columns)

        self.pipeline = pipeline

        # Transform data
        X_transformed = pipeline.transform(X_features, energy)
        y_transformed = pipeline.transform_target(y)

        # Extract arrays
        X_arr = X_transformed[self.feature_columns].values
        y_arr = y_transformed.values

        # Handle inf/NaN (critical for AME-enriched data)
        nan_mask = np.isnan(X_arr)
        inf_mask = np.isinf(X_arr)
        nan_cell_count = nan_mask.sum()
        inf_cell_count = inf_mask.sum()
        rows_with_nan = nan_mask.any(axis=1).sum()
        y_invalid = ~np.isfinite(y_arr)

        if missing_uncertainty_handling == 'exclude' and rows_with_nan > 0:
            # Drop rows with any NaN/inf in features
            valid_rows = ~nan_mask.any(axis=1) & ~inf_mask.any(axis=1)
            n_before = len(X_arr)
            X_arr = X_arr[valid_rows]
            y_arr = y_arr[valid_rows]
            if sample_weights is not None:
                sample_weights = sample_weights[valid_rows]
            n_samples, n_features = nan_mask.shape
            print(f"  Feature matrix: {n_samples:,} samples x {n_features} features")
            print(f"  NaN cells: {nan_cell_count:,} ({nan_cell_count / nan_mask.size * 100:.2f}% of cells)")
            print(f"  Rows with any NaN: {rows_with_nan:,} ({rows_with_nan / n_samples * 100:.2f}% of samples)")
            print(f"  Dropping {n_before - len(X_arr):,} rows with NaN/inf features "
                  f"(missing_uncertainty_handling='exclude')")
        else:
            if nan_cell_count > 0 or inf_cell_count > 0:
                n_samples, n_features = X_arr.shape
                print(f"  Feature matrix: {n_samples:,} samples x {n_features} features")
                print(f"  NaN cells: {nan_cell_count:,} ({nan_cell_count / X_arr.size * 100:.2f}% of cells)")
                print(f"  Rows with any NaN: {rows_with_nan:,} ({rows_with_nan / n_samples * 100:.2f}% of samples)")
                print(f"  Inf cells: {inf_cell_count:,}")
                print(f"  Target invalid: {y_invalid.sum():,} samples")
                print(f"  Note: NaN values imputed to 0 (standardized mean); no rows removed")
            # Impute NaN
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e10, neginf=-1e10)

        # Remove rows with invalid targets (and corresponding sample weights)
        valid_target_mask = np.isfinite(y_arr)
        if not valid_target_mask.all():
            n_invalid = (~valid_target_mask).sum()
            print(f"  Removing {n_invalid:,} rows with invalid target ({n_invalid/len(y_arr)*100:.2f}%)")
            X_arr = X_arr[valid_target_mask]
            y_arr = y_arr[valid_target_mask]
            if sample_weights is not None:
                sample_weights = sample_weights[valid_target_mask]

        # Remove rows with NaN sample weights (from missing_uncertainty_handling='exclude')
        if sample_weights is not None:
            valid_weight_mask = np.isfinite(sample_weights)
            if not valid_weight_mask.all():
                n_excluded = (~valid_weight_mask).sum()
                print(f"  Excluding {n_excluded:,} rows without valid uncertainty "
                      f"(missing_uncertainty_handling='exclude')")
                X_arr = X_arr[valid_weight_mask]
                y_arr = y_arr[valid_weight_mask]
                sample_weights = sample_weights[valid_weight_mask]

        if len(X_arr) == 0:
            raise ValueError(
                "No valid samples after handling NaN/inf. Check that:\n"
                "1. Target column (CrossSection) has valid positive values\n"
                "2. Energy column has valid positive values\n"
                "3. Feature columns don't have all-NaN values"
            )

        # Split data (include sample weights if using uncertainty weighting)
        if sample_weights is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X_arr, y_arr, sample_weights,
                test_size=test_size, random_state=self.params['random_state']
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=test_size, random_state=self.params['random_state']
            )
            w_train = None

        # Train model
        print(f"Training XGBoost ({self.params['n_estimators']} trees, "
              f"max_depth={self.params['max_depth']})...")

        eval_set = [(X_test, y_test)]
        self.model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Predictions in transformed (log) space
        y_train_pred_log = self.model.predict(X_train)
        y_test_pred_log = self.model.predict(X_test)

        # ----------------------------------------------------------------
        # DUAL-SPACE METRIC CALCULATION
        # ----------------------------------------------------------------
        # 1) Feature space (log10): metrics on raw model output
        # 2) Physical space (barns): inverse-transform then re-calculate
        # ----------------------------------------------------------------

        # -- Log-space metrics --
        train_mse_log = mean_squared_error(y_train, y_train_pred_log)
        test_mse_log = mean_squared_error(y_test, y_test_pred_log)
        train_mae_log = mean_absolute_error(y_train, y_train_pred_log)
        test_mae_log = mean_absolute_error(y_test, y_test_pred_log)
        train_r2_log = r2_score(y_train, y_train_pred_log)
        test_r2_log = r2_score(y_test, y_test_pred_log)

        # -- Physical-space metrics (barns) --
        if self.pipeline.config.log_target:
            # Inverse transform: sigma = 10^y' - epsilon, clipped to >= 0
            y_train_pred_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_train_pred_log)).values
            y_test_pred_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_test_pred_log)).values
            y_train_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_train)).values
            y_test_barns = self.pipeline.inverse_transform_target(
                pd.Series(y_test)).values

            # Safety clip: ensure no negative values from floating-point noise
            y_train_pred_barns = np.clip(y_train_pred_barns, 0.0, None)
            y_test_pred_barns = np.clip(y_test_pred_barns, 0.0, None)
            y_train_barns = np.clip(y_train_barns, 0.0, None)
            y_test_barns = np.clip(y_test_barns, 0.0, None)
        else:
            y_train_pred_barns = y_train_pred_log
            y_test_pred_barns = y_test_pred_log
            y_train_barns = y_train
            y_test_barns = y_test

        train_mse_barns = mean_squared_error(y_train_barns, y_train_pred_barns)
        test_mse_barns = mean_squared_error(y_test_barns, y_test_pred_barns)
        train_mae_barns = mean_absolute_error(y_train_barns, y_train_pred_barns)
        test_mae_barns = mean_absolute_error(y_test_barns, y_test_pred_barns)
        train_r2_barns = r2_score(y_train_barns, y_train_pred_barns)
        test_r2_barns = r2_score(y_test_barns, y_test_pred_barns)

        # -- Ensemble structural diagnostics --
        n_train = len(X_train)
        n_test = len(X_test)
        best_iter = self.model.best_iteration if hasattr(self.model, 'best_iteration') else None

        # ----------------------------------------------------------------
        # Store all metrics
        # ----------------------------------------------------------------
        self.metrics = {
            # Log-space (feature space)
            'train_mse_log': train_mse_log,
            'test_mse_log': test_mse_log,
            'train_mae_log': train_mae_log,
            'test_mae_log': test_mae_log,
            'train_r2_log': train_r2_log,
            'test_r2_log': test_r2_log,
            # Physical space (barns)
            'train_mse_barns': train_mse_barns,
            'test_mse_barns': test_mse_barns,
            'train_mae_barns': train_mae_barns,
            'test_mae_barns': test_mae_barns,
            'train_r2_barns': train_r2_barns,
            'test_r2_barns': test_r2_barns,
            # Structural
            'n_train': n_train,
            'n_test': n_test,
            'best_iteration': best_iter,
            'n_estimators': self.params['n_estimators'],
            'max_depth': self.params['max_depth'],
        }

        self.is_trained = True

        # ----------------------------------------------------------------
        # FORMATTED PERFORMANCE DIAGNOSTICS TABLE
        # ----------------------------------------------------------------
        def _gap_pct(train_val, test_val):
            """Compute gap as percentage: (test - train) / |train| * 100."""
            if train_val == 0:
                return float('inf')
            return (test_val - train_val) / abs(train_val) * 100

        def _fmt_metric(val, is_r2=False):
            """Format a metric value for table display."""
            if is_r2:
                return f"{val:.4f}"
            elif abs(val) >= 1e5 or (abs(val) < 1e-2 and val != 0):
                return f"{val:.2e}"
            else:
                return f"{val:.4f}"

        print()
        print("=" * 70)
        print("FINAL PERFORMANCE DIAGNOSTICS")
        print("=" * 70)
        print(f"{'METRIC':<20s}| {'TRAIN SET':>16s} | {'TEST SET':>16s} | {'GAP (%)':>8s}")
        print("-" * 70)

        rows = [
            ("MSE  (Log10)",  train_mse_log,   test_mse_log,   False),
            ("MAE  (Log10)",  train_mae_log,    test_mae_log,   False),
            ("R^2  (Log10)",  train_r2_log,     test_r2_log,    True),
            ("MSE  (Barns)",  train_mse_barns,  test_mse_barns, False),
            ("MAE  (Barns)",  train_mae_barns,  test_mae_barns, False),
            ("R^2  (Barns)",  train_r2_barns,   test_r2_barns,  True),
        ]

        for label, train_val, test_val, is_r2 in rows:
            gap = _gap_pct(train_val, test_val)
            gap_str = f"{gap:+.1f}%" if abs(gap) < 1e6 else "   inf"
            print(f"{label:<20s}| {_fmt_metric(train_val, is_r2):>16s} "
                  f"| {_fmt_metric(test_val, is_r2):>16s} | {gap_str:>8s}")

        print("-" * 70)

        # ----------------------------------------------------------------
        # GENERALIZATION & OVERFITTING DIAGNOSTICS
        # ----------------------------------------------------------------
        mse_gap_log = _gap_pct(train_mse_log, test_mse_log)
        r2_gap_log = test_r2_log - train_r2_log

        print()
        print("GENERALIZATION & STRUCTURAL DIAGNOSTICS")
        print("-" * 70)
        print(f"  Log-space MSE gap (test-train)/train:  {mse_gap_log:+.1f}%")
        print(f"  Log-space R^2 gap (test - train):      {r2_gap_log:+.4f}")
        print(f"  n_estimators:        {self.params['n_estimators']}")
        print(f"  max_depth:           {self.params['max_depth']}")
        if best_iter is not None:
            print(f"  best_iteration:      {best_iter}")
        print(f"  Training samples:    {n_train:,}")
        print(f"  Test samples:        {n_test:,}")

        if mse_gap_log > 50:
            print(f"  [!] WARNING: Large generalization gap ({mse_gap_log:.0f}%) "
                  f"-- possible overfitting")
        elif mse_gap_log < 2:
            print(f"  [i] Tight generalization gap ({mse_gap_log:.1f}%) "
                  f"-- model may be underfitting")
        else:
            print(f"  [OK] Generalization gap within normal range")

        print("=" * 70)

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

        # Handle NaN with same imputation as training (0 = mean in standardized space)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e10, neginf=-1e10)

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
