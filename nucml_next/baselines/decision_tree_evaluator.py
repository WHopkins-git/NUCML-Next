"""
Decision Tree Evaluator - Research Baseline for Nuclear Cross-Section Prediction
=================================================================================

This module provides a production-grade Decision Tree baseline with multiple
hyperparameter optimization methods and full pipeline integration.

HYPERPARAMETER OPTIMIZATION METHODS
-----------------------------------
This module supports four hyperparameter optimization strategies:

1. **Grid Search** (`optimize_hyperparameters(method='grid')`)
   - Exhaustive search over all parameter combinations
   - Best for: Small search spaces, guaranteed to find optimum in space
   - Downside: Exponential time complexity O(n^k) for k parameters
   - sklearn: GridSearchCV

2. **Random Search** (`optimize_hyperparameters(method='random')`)
   - Randomly samples parameter combinations
   - Best for: Large search spaces, continuous parameters
   - Typically finds good solutions faster than grid search
   - sklearn: RandomizedSearchCV
   - Reference: Bergstra & Bengio (2012) "Random Search for Hyper-Parameter Optimization"

3. **Halving Random Search** (`optimize_hyperparameters(method='halving')`)
   - Successive halving with random sampling
   - Best for: Large datasets, many parameter combinations
   - Progressively eliminates poor candidates using subsets of data
   - sklearn: HalvingRandomSearchCV
   - Much faster than full random search on large datasets

4. **Bayesian Optimization** (`optimize_hyperparameters(method='bayesian')`)
   - Uses probabilistic model to guide search
   - Best for: Expensive evaluations, complex parameter interactions
   - Requires hyperopt library: `pip install hyperopt`
   - Most sample-efficient but more complex

QUICK START
-----------
```python
from nucml_next.data import NucmlDataset, DataSelection
from nucml_next.baselines import DecisionTreeEvaluator

# Load data
selection = DataSelection(tiers=['A', 'C'])
dataset = NucmlDataset('data.parquet', selection=selection)
df = dataset.to_tabular()

# Option 1: Grid Search (recommended for notebooks)
evaluator = DecisionTreeEvaluator()
result = evaluator.optimize_hyperparameters(
    df,
    method='grid',
    param_grid={
        'max_depth': [60, 80, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    cv_folds=3,
    scoring='neg_mean_squared_error'
)

# Option 2: Bayesian (for production)
result = evaluator.optimize_hyperparameters(
    df,
    method='bayesian',
    max_evals=50,
    cv_folds=5
)

# Train final model with best params
final_model = DecisionTreeEvaluator(**result['best_params'])
metrics = final_model.train(df)
```

FEATURES
--------
- TransformationPipeline integration with configurable scalers
- Automatic particle emission vector handling (MT codes → 9 features)
- Robust handling of AME-enriched data with NaN imputation
- Comprehensive feature importance analysis
- Model persistence (save/load)

EDUCATIONAL PURPOSE
-------------------
Demonstrates the "staircase effect" inherent to tree-based models:
- Decision trees partition feature space into rectangles
- Each rectangle maps to a constant prediction
- Real cross-sections are smooth functions → fundamental mismatch

This motivates the need for smooth, physics-informed models.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
import joblib

# Optional: Halving search (sklearn >= 0.24)
try:
    from sklearn.model_selection import HalvingRandomSearchCV
    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False

# Optional: Bayesian optimization
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform

from nucml_next.data.transformations import TransformationPipeline
from nucml_next.data.selection import TransformationConfig


class DecisionTreeEvaluator:
    """
    Decision Tree baseline for nuclear cross-section prediction.

    Integrates with NUCML-Next TransformationPipeline for:
    - Configurable log transforms (log10, ln, log2)
    - Multiple scaler types (standard, minmax, robust, none)
    - Automatic particle emission vector handling (MT → 9 features)
    - Reversible transformations for predictions

    Example:
        >>> from nucml_next.data import NucmlDataset, DataSelection
        >>>
        >>> # Load data with optimal configuration
        >>> selection = DataSelection(tiers=['A', 'B', 'C'])
        >>> dataset = NucmlDataset('data.parquet', selection=selection)
        >>> df = dataset.to_tabular()
        >>>
        >>> # Train with automatic transformation
        >>> evaluator = DecisionTreeEvaluator()
        >>> metrics = evaluator.train(df)
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
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_leaf: Minimum samples per leaf node
            min_samples_split: Minimum samples to split a node
            min_impurity_decrease: Minimum impurity decrease for split
            max_features: Features per split ('sqrt', 'log2', None=all)
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
        method: str = 'grid',
        target_column: str = 'CrossSection',
        energy_column: str = 'Energy',
        exclude_columns: Optional[list] = None,
        pipeline: Optional[TransformationPipeline] = None,
        transformation_config: Optional[TransformationConfig] = None,
        cv_folds: int = 3,
        test_size: float = 0.2,
        scoring: str = 'neg_mean_squared_error',
        verbose: bool = True,
        n_jobs: int = -1,
        # Grid/Random search parameters
        param_grid: Optional[Dict[str, List]] = None,
        n_iter: int = 50,  # For random search
        # Subsampling for memory efficiency
        subsample_fraction: Optional[float] = None,
        subsample_max_samples: Optional[int] = None,
        # Bayesian optimization parameters (method='bayesian')
        max_evals: int = 100,
        max_depth_options: Optional[list] = None,
        min_samples_split_range: tuple = (2, 20),
        min_samples_leaf_range: tuple = (1, 10),
        max_features_options: Optional[list] = None,
        # Uncertainty-based sample filtering
        use_uncertainty_weights: Optional[str] = None,
        missing_uncertainty_handling: str = 'median',
        uncertainty_column: str = 'Uncertainty',
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using specified method.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            method: Optimization method - 'grid', 'random', 'halving', or 'bayesian'
            target_column: Target column name
            energy_column: Energy column name
            exclude_columns: Columns to exclude from features
            pipeline: Pre-configured TransformationPipeline
            transformation_config: Config for new pipeline (if pipeline=None)
            cv_folds: Cross-validation folds
            test_size: Test set fraction (for final evaluation)
            scoring: Scoring metric for sklearn methods
                     Common options: 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'
            verbose: Print progress
            n_jobs: Parallel jobs (-1 = all cores)

            # Grid/Random search parameters:
            param_grid: Dictionary of parameters to search
                        Example: {'max_depth': [10, 50, 100], 'min_samples_leaf': [1, 5, 10]}
            n_iter: Number of iterations for random search

            # Subsampling for memory efficiency:
            subsample_fraction: Fraction of data to use for hyperparameter search (0.0-1.0)
                               Example: 0.1 uses 10% of data for search, then trains final
                               model on full data. Recommended for datasets > 1M samples.
            subsample_max_samples: Maximum number of samples for hyperparameter search.
                                   Alternative to subsample_fraction. If both provided,
                                   the smaller resulting sample size is used.

            # Bayesian optimization parameters (method='bayesian' only):
            max_evals: Maximum Bayesian optimization iterations
            max_depth_options: List of max_depth values [60, 80, 100, None]
            min_samples_split_range: (min, max) tuple for sampling
            min_samples_leaf_range: (min, max) tuple for sampling
            max_features_options: List of max_features values [None, 'sqrt']

            # Uncertainty-based sample filtering:
            use_uncertainty_weights: Weight mode (None, 'xs', or 'both').
                When set with missing_uncertainty_handling='exclude', samples without
                valid uncertainty are removed before hyperparameter search.
            missing_uncertainty_handling: How to handle missing uncertainties
                ('median', 'equal', or 'exclude'). Only 'exclude' affects
                the hyperparameter search by removing rows.
            uncertainty_column: Column name for cross-section uncertainty.

        Returns:
            Dictionary with:
            - best_params: Optimal hyperparameters
            - best_score: Best cross-validation score
            - test_mse: Test MSE in original scale
            - cv_results: Full CV results (for grid/random methods)
            - trials: Hyperopt trials object (for bayesian method)
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

        if method == 'bayesian':
            return self._optimize_bayesian(
                df=df,
                target_column=target_column,
                energy_column=energy_column,
                exclude_columns=exclude_columns,
                pipeline=pipeline,
                transformation_config=transformation_config,
                max_evals=max_evals,
                cv_folds=cv_folds,
                test_size=test_size,
                verbose=verbose,
                max_depth_options=max_depth_options,
                min_samples_split_range=min_samples_split_range,
                min_samples_leaf_range=min_samples_leaf_range,
                max_features_options=max_features_options,
            )
        elif method in ['grid', 'random', 'halving']:
            return self._optimize_sklearn(
                df=df,
                method=method,
                target_column=target_column,
                energy_column=energy_column,
                exclude_columns=exclude_columns,
                pipeline=pipeline,
                transformation_config=transformation_config,
                param_grid=param_grid,
                n_iter=n_iter,
                cv_folds=cv_folds,
                test_size=test_size,
                scoring=scoring,
                verbose=verbose,
                n_jobs=n_jobs,
                subsample_fraction=subsample_fraction,
                subsample_max_samples=subsample_max_samples,
                missing_uncertainty_handling=missing_uncertainty_handling,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'grid', 'random', 'halving', or 'bayesian'")

    def _optimize_sklearn(
        self,
        df: pd.DataFrame,
        method: str,
        target_column: str,
        energy_column: str,
        exclude_columns: Optional[list],
        pipeline: Optional[TransformationPipeline],
        transformation_config: Optional[TransformationConfig],
        param_grid: Optional[Dict],
        n_iter: int,
        cv_folds: int,
        test_size: float,
        scoring: str,
        verbose: bool,
        n_jobs: int,
        subsample_fraction: Optional[float] = None,
        subsample_max_samples: Optional[int] = None,
        missing_uncertainty_handling: str = 'median',
    ) -> Dict[str, Any]:
        """
        Hyperparameter optimization using sklearn methods (Grid, Random, Halving).

        Supports subsampling for memory efficiency on large datasets.
        """
        if method == 'halving' and not HALVING_AVAILABLE:
            raise ImportError(
                "HalvingRandomSearchCV requires sklearn >= 0.24. "
                "Install: pip install scikit-learn>=0.24"
            )

        method_names = {
            'grid': 'Grid Search',
            'random': 'Random Search',
            'halving': 'Halving Random Search'
        }

        # Determine if subsampling is needed
        n_original = len(df)
        use_subsample = subsample_fraction is not None or subsample_max_samples is not None

        if verbose:
            print("\n" + "=" * 80)
            print(f"HYPERPARAMETER OPTIMIZATION - Decision Tree ({method_names[method]})")
            print("=" * 80)
            print(f"Dataset size: {n_original:,} samples")
            print(f"Method: {method_names[method]}")
            print(f"Cross-validation folds: {cv_folds}")
            print(f"Scoring: {scoring}")

            if use_subsample:
                # Calculate target subsample size
                target_size = n_original
                if subsample_fraction is not None:
                    target_size = min(target_size, int(n_original * subsample_fraction))
                if subsample_max_samples is not None:
                    target_size = min(target_size, subsample_max_samples)
                print(f"\nSubsampling enabled for hyperparameter search:")
                print(f"  Original size: {n_original:,} samples")
                print(f"  Search size:   {target_size:,} samples ({100*target_size/n_original:.1f}%)")
                print(f"  NOTE: Final model will be trained on FULL data after search")
            print()

        # Prepare data (on full dataset - transformation pipeline needs full data for fitting)
        X_arr, y_arr, feature_columns, pipeline = self._prepare_data(
            df, target_column, energy_column, exclude_columns,
            pipeline, transformation_config, verbose,
            missing_uncertainty_handling=missing_uncertainty_handling,
        )

        # Apply subsampling AFTER data preparation
        n_samples = len(X_arr)
        if use_subsample:
            target_size = n_samples
            if subsample_fraction is not None:
                target_size = min(target_size, int(n_samples * subsample_fraction))
            if subsample_max_samples is not None:
                target_size = min(target_size, subsample_max_samples)

            if target_size < n_samples:
                # Stratified random sampling
                rng = np.random.default_rng(42)
                subsample_idx = rng.choice(n_samples, size=target_size, replace=False)
                X_arr_search = X_arr[subsample_idx]
                y_arr_search = y_arr[subsample_idx]
                if verbose:
                    print(f"  Subsampled: {n_samples:,} → {target_size:,} samples for search")
            else:
                X_arr_search = X_arr
                y_arr_search = y_arr
        else:
            X_arr_search = X_arr
            y_arr_search = y_arr

        # Split data for hyperparameter search (using subsampled data if applicable)
        X_train_search, X_test_search, y_train_search, y_test_search = train_test_split(
            X_arr_search, y_arr_search, test_size=test_size, random_state=42
        )

        # Also split full data for final evaluation (only if subsampling)
        if use_subsample and target_size < n_samples:
            X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
                X_arr, y_arr, test_size=test_size, random_state=42
            )
        else:
            X_train_full, X_test_full = X_train_search, X_test_search
            y_train_full, y_test_full = y_train_search, y_test_search

        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'max_depth': [20, 50, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': [None],
            }

        if verbose:
            print("Parameter grid:")
            for param, values in param_grid.items():
                print(f"  {param}: {values}")
            if method == 'grid':
                total_combinations = 1
                for values in param_grid.values():
                    total_combinations *= len(values)
                print(f"Total combinations: {total_combinations}")
            elif method == 'random':
                print(f"Random iterations: {n_iter}")
            print()

        # Create search object
        base_model = DecisionTreeRegressor(random_state=42)

        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=2 if verbose else 0,
                return_train_score=True,
            )
        elif method == 'random':
            # Convert lists to distributions for random search
            param_distributions = {}
            for param, values in param_grid.items():
                if all(isinstance(v, (int, type(None))) for v in values):
                    param_distributions[param] = values  # Use as-is for categorical
                else:
                    param_distributions[param] = values

            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=2 if verbose else 0,
                return_train_score=True,
                random_state=42,
            )
        elif method == 'halving':
            search = HalvingRandomSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=2 if verbose else 0,
                random_state=42,
                factor=3,  # Reduce candidates by factor of 3 each iteration
                min_resources='smallest',
            )

        # Run search on (sub)sampled data
        if verbose:
            print("Starting optimization...")
            print("-" * 80)

        search.fit(X_train_search, y_train_search)

        if verbose:
            print("-" * 80)

        # Extract best parameters
        best_params = search.best_params_.copy()
        best_params['random_state'] = 42

        # Evaluate on FULL test set (not subsampled)
        # If subsampling was used, retrain best model on full training data
        if use_subsample and target_size < n_samples:
            if verbose:
                print(f"\nRetraining best model on FULL training data ({len(X_train_full):,} samples)...")
            final_estimator = DecisionTreeRegressor(**best_params)
            final_estimator.fit(X_train_full, y_train_full)
            y_test_pred = final_estimator.predict(X_test_full)
        else:
            y_test_pred = search.best_estimator_.predict(X_test_full)
            final_estimator = search.best_estimator_

        # Inverse transform
        if pipeline.config.log_target:
            y_test_pred = pipeline.inverse_transform_target(pd.Series(y_test_pred)).values
            y_test_orig = pipeline.inverse_transform_target(pd.Series(y_test_full)).values
        else:
            y_test_orig = y_test_full

        test_mse = mean_squared_error(y_test_orig, y_test_pred)
        test_mae = mean_absolute_error(y_test_orig, y_test_pred)

        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE")
            print("=" * 80)
            print(f"Best CV Score ({scoring}): {search.best_score_:.6f}")
            print(f"Test MSE (original scale): {test_mse:.4e}")
            print(f"Test MAE (original scale): {test_mae:.4e}")
            print()
            print("Optimal Hyperparameters:")
            for key, value in best_params.items():
                if key != 'random_state':
                    print(f"  {key:25s}: {value}")
            print("=" * 80)

        return {
            'best_params': best_params,
            'best_score': search.best_score_,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'cv_results': pd.DataFrame(search.cv_results_),
            'best_estimator': final_estimator,
            'subsampled': use_subsample and target_size < n_samples,
        }

    def _optimize_bayesian(
        self,
        df: pd.DataFrame,
        target_column: str,
        energy_column: str,
        exclude_columns: Optional[list],
        pipeline: Optional[TransformationPipeline],
        transformation_config: Optional[TransformationConfig],
        max_evals: int,
        cv_folds: int,
        test_size: float,
        verbose: bool,
        max_depth_options: Optional[list],
        min_samples_split_range: tuple,
        min_samples_leaf_range: tuple,
        max_features_options: Optional[list],
    ) -> Dict[str, Any]:
        """
        Hyperparameter optimization using Bayesian optimization (hyperopt).
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError(
                "hyperopt required for Bayesian optimization. Install: pip install hyperopt"
            )

        if verbose:
            print("\n" + "=" * 80)
            print("HYPERPARAMETER OPTIMIZATION - Decision Tree (Bayesian)")
            print("=" * 80)
            print(f"Dataset size: {len(df):,} samples")
            print(f"Max evaluations: {max_evals}")
            print(f"Cross-validation folds: {cv_folds}")
            print()

        # Prepare data
        X_arr, y_arr, feature_columns, pipeline = self._prepare_data(
            df, target_column, energy_column, exclude_columns,
            pipeline, transformation_config, verbose
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=42
        )

        # Apply defaults for search space parameters
        if max_depth_options is None:
            max_depth_options = [20, 30, 50, 100, None]
        if max_features_options is None:
            max_features_options = [None, 'sqrt']

        if verbose:
            print("Search space:")
            print(f"  max_depth options: {max_depth_options}")
            print(f"  min_samples_split: {min_samples_split_range[0]} to {min_samples_split_range[1]}")
            print(f"  min_samples_leaf: {min_samples_leaf_range[0]} to {min_samples_leaf_range[1]}")
            print(f"  max_features options: {max_features_options}")
            print()

        # Hyperparameter search space
        space = {
            'max_depth': hp.choice('max_depth', max_depth_options),
            'min_samples_split': hp.quniform('min_samples_split',
                                             min_samples_split_range[0],
                                             min_samples_split_range[1], 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf',
                                            min_samples_leaf_range[0],
                                            min_samples_leaf_range[1], 1),
            'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-10), np.log(1e-4)),
            'max_features': hp.choice('max_features', max_features_options),
        }

        def objective(params):
            if params['max_depth'] is not None:
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
        if best_params['max_depth'] is not None:
            best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        best_params['random_state'] = 42

        # Evaluate on test set
        final_model = DecisionTreeRegressor(**best_params)
        final_model.fit(X_train, y_train)
        y_test_pred = final_model.predict(X_test)

        # Inverse transform
        if pipeline.config.log_target:
            y_test_pred = pipeline.inverse_transform_target(pd.Series(y_test_pred)).values
            y_test_orig = pipeline.inverse_transform_target(pd.Series(y_test)).values
        else:
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
            'best_score': best_cv_score,
            'test_mse': test_mse,
            'trials': trials,
        }

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        energy_column: str,
        exclude_columns: Optional[list],
        pipeline: Optional[TransformationPipeline],
        transformation_config: Optional[TransformationConfig],
        verbose: bool,
        missing_uncertainty_handling: str = 'median',
    ) -> tuple:
        """
        Prepare data for optimization: feature selection, transformation, NaN handling.

        When missing_uncertainty_handling='exclude', rows with any NaN in features
        are dropped instead of imputed to 0.

        Returns:
            Tuple of (X_arr, y_arr, feature_columns, pipeline)
        """
        # Prepare features
        if exclude_columns is None:
            exclude_columns = [target_column, 'Uncertainty', 'Entry', 'MT']

        # Auto-detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sparse_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)]
        all_numeric = list(set(numeric_cols + sparse_cols))
        feature_columns = [col for col in all_numeric if col not in exclude_columns]

        non_numeric = [col for col in df.columns if col not in all_numeric and col not in exclude_columns]
        if len(non_numeric) > 0 and verbose:
            print(f"Excluding {len(non_numeric)} non-numeric columns: {non_numeric[:5]}...")
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

        # Handle inf/NaN
        X_arr = X_transformed[feature_columns].values
        y_arr = y_transformed.values

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
            if verbose:
                n_samples, n_features = nan_mask.shape
                print(f"  Feature matrix: {n_samples:,} samples x {n_features} features")
                print(f"  NaN cells: {nan_cell_count:,} ({nan_cell_count / nan_mask.size * 100:.2f}% of cells)")
                print(f"  Rows with any NaN: {rows_with_nan:,} ({rows_with_nan / n_samples * 100:.2f}% of samples)")
                print(f"  Dropping {n_before - len(X_arr):,} rows with NaN/inf features "
                      f"(missing_uncertainty_handling='exclude')")
                print()
        else:
            if verbose and (nan_cell_count > 0 or inf_cell_count > 0):
                n_samples, n_features = X_arr.shape
                print(f"  Feature matrix: {n_samples:,} samples x {n_features} features")
                print(f"  NaN cells: {nan_cell_count:,} ({nan_cell_count / X_arr.size * 100:.2f}% of cells)")
                print(f"  Rows with any NaN: {rows_with_nan:,} ({rows_with_nan / n_samples * 100:.2f}% of samples)")
                print(f"  Inf cells: {inf_cell_count:,}")
                print(f"  Target invalid: {y_invalid.sum():,} samples")
                print(f"  Note: NaN values imputed to 0 (standardized mean); no rows removed")
                print()
            # Impute NaN
            X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e10, neginf=-1e10)

        # Remove rows with invalid targets
        valid_target_mask = np.isfinite(y_arr)
        if not valid_target_mask.all():
            n_invalid = (~valid_target_mask).sum()
            if verbose:
                print(f"  Removing {n_invalid:,} rows with invalid target")
            X_arr = X_arr[valid_target_mask]
            y_arr = y_arr[valid_target_mask]

        if len(X_arr) == 0:
            raise ValueError(
                "No valid samples after handling NaN/inf. Check that:\n"
                "1. Target column (CrossSection) has valid positive values\n"
                "2. Energy column has valid positive values\n"
                "3. Feature columns don't have all-NaN values"
            )

        return X_arr, y_arr, feature_columns, pipeline

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
        use_uncertainty_weights: Optional[str] = None,
        missing_uncertainty_handling: str = 'median',
    ) -> Dict[str, float]:
        """
        Train the Decision Tree model with full pipeline integration.

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

        X_features = working_df[self.feature_columns]
        y = working_df[target_column]
        energy = working_df[energy_column] if energy_column in working_df.columns else None

        # Compute sample weights from uncertainty (inverse variance weighting)
        sample_weights = compute_sample_weights(
            working_df,
            mode=weight_mode,
            uncertainty_column=uncertainty_column,
            energy_uncertainty_column=energy_uncertainty_column,
            missing_handling=missing_uncertainty_handling,
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
                test_size=test_size, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=test_size, random_state=self.random_state
            )
            w_train = None

        # Train model
        print(f"Training Decision Tree (max_depth={self.max_depth}, "
              f"min_samples_leaf={self.min_samples_leaf})...")
        self.model.fit(X_train, y_train, sample_weight=w_train)

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Inverse transform
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

        print(f"[OK] Training complete!")
        print(f"  Test MSE: {test_mse:.4e}")
        print(f"  Test MAE: {test_mae:.4e}")
        print(f"  Test R2: {test_r2:.4f}")
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

        # Handle NaN
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e10, neginf=-1e10)

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
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")

        importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

        return importance

    def save(self, filepath: str):
        """Save model and pipeline to disk."""
        model_data = {
            'model': self.model,
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'params': {
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'min_samples_split': self.min_samples_split,
                'min_impurity_decrease': self.min_impurity_decrease,
                'max_features': self.max_features,
                'random_state': self.random_state,
            }
        }
        joblib.dump(model_data, filepath)
        print(f"[OK] Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and pipeline from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.pipeline = model_data['pipeline']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']

        params = model_data['params']
        self.max_depth = params['max_depth']
        self.min_samples_leaf = params['min_samples_leaf']
        self.min_samples_split = params['min_samples_split']
        self.min_impurity_decrease = params['min_impurity_decrease']
        self.max_features = params['max_features']
        self.random_state = params['random_state']

        self.is_trained = True
        print(f"[OK] Model loaded from {filepath}")
