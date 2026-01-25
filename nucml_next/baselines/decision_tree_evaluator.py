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

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False


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
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize Decision Tree evaluator.

        Args:
            max_depth: Maximum tree depth (lower = more stairs)
            min_samples_leaf: Minimum samples per leaf (higher = coarser steps)
            min_samples_split: Minimum samples required to split a node
            min_impurity_decrease: Minimum impurity decrease for a split
            max_features: Number of features to consider ('sqrt', 'log2', or None for all)
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
        self.metrics = {}

    def optimize_hyperparameters(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        exclude_columns: Optional[list] = None,
        max_evals: int = 100,
        cv_folds: int = 3,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization (hyperopt).

        This finds the optimal max_depth, min_samples_leaf, and other parameters
        for the Decision Tree on the given dataset.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            max_evals: Maximum number of hyperparameter evaluations
            cv_folds: Number of cross-validation folds
            test_size: Fraction of data for final test set
            verbose: Print optimization progress

        Returns:
            Dictionary with:
                - best_params: Optimized hyperparameters
                - best_score: Best cross-validation score
                - trials: Hyperopt trials object for analysis

        Example:
            >>> dt = DecisionTreeEvaluator()
            >>> result = dt.optimize_hyperparameters(df_train, max_evals=50)
            >>> print(f"Optimal depth: {result['best_params']['max_depth']}")
            >>> # Now create new model with optimal params
            >>> dt_optimal = DecisionTreeEvaluator(**result['best_params'])
            >>> dt_optimal.train(df_train)
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError(
                "hyperopt is required for hyperparameter optimization.\n"
                "Install with: pip install hyperopt"
            )

        if verbose:
            print("\n" + "=" * 80)
            print("HYPERPARAMETER OPTIMIZATION - Decision Tree")
            print("=" * 80)
            print(f"Dataset size: {len(df):,} samples")
            print(f"Max evaluations: {max_evals}")
            print(f"Cross-validation folds: {cv_folds}")
            print()

        # Prepare features and target
        if exclude_columns is None:
            exclude_columns = [target_column, 'Isotope', 'Reaction']

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # CRITICAL: Filter out non-numeric columns automatically
        # This prevents "could not convert string to float" errors
        # Includes pandas sparse arrays (common for one-hot encoded MT codes)

        # Get numeric columns (standard numeric types)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Also include sparse columns (pandas sparse arrays for one-hot encoded data)
        sparse_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)]

        # Combine and remove duplicates
        all_numeric_cols = list(set(numeric_cols + sparse_cols))
        feature_columns = [col for col in all_numeric_cols if col not in exclude_columns]

        # Report excluded non-numeric columns if any
        all_candidate_columns = [col for col in df.columns if col not in exclude_columns]
        non_numeric_columns = [col for col in all_candidate_columns if col not in feature_columns]

        if len(non_numeric_columns) > 0 and verbose:
            print(f"⚠️  Excluding {len(non_numeric_columns)} non-numeric columns:")
            for col in non_numeric_columns[:5]:  # Show first 5
                print(f"    - {col} (dtype: {df[col].dtype})")
            if len(non_numeric_columns) > 5:
                print(f"    ... and {len(non_numeric_columns) - 5} more")
            print()

        # Get the feature DataFrame
        X_df = df[feature_columns]

        # Handle sparse DataFrames
        sparse_columns = [col for col in X_df.columns if isinstance(X_df[col].dtype, pd.SparseDtype)]

        if len(sparse_columns) > 0:
            import scipy.sparse as sp
            try:
                if len(sparse_columns) == len(X_df.columns):
                    X = sp.csr_matrix(X_df.sparse.to_coo())
                else:
                    X = X_df.values
            except (AttributeError, ValueError):
                X = X_df.values
        else:
            X = X_df.values

        y = df[target_column].values

        # Log-transform target
        y_log = np.log10(y + 1e-10)

        # CRITICAL: Remove inf/NaN values that cause sklearn to fail
        # This can happen with AME enrichment data for unstable isotopes
        if isinstance(X, np.ndarray):
            # For dense arrays, check for inf/NaN
            finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y_log)
            if not finite_mask.all():
                n_invalid = (~finite_mask).sum()
                if verbose:
                    print(f"⚠️  Removing {n_invalid:,} rows with inf/NaN values ({n_invalid/len(X)*100:.2f}%)")
                X = X[finite_mask]
                y_log = y_log[finite_mask]
        # Note: scipy sparse matrices handle this differently, typically don't contain inf

        # Split into train/test (hyperopt will use train for CV)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=42
        )

        # Define hyperparameter search space
        space = {
            'max_depth': hp.quniform('max_depth', 5, 30, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 50, 1),
            'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-8), np.log(1e-2)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        }

        # Define objective function for hyperopt
        def objective(params):
            # Convert float hyperparameters to int where needed
            params['max_depth'] = int(params['max_depth'])
            params['min_samples_split'] = int(params['min_samples_split'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])

            # Create model with these hyperparameters
            model = DecisionTreeRegressor(
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                min_impurity_decrease=params['min_impurity_decrease'],
                max_features=params['max_features'],
                random_state=42,
            )

            # Cross-validation score (negative MSE, since hyperopt minimizes)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            # Return mean CV score (hyperopt minimizes, so negate)
            mean_cv_score = -cv_scores.mean()

            return {
                'loss': mean_cv_score,
                'status': STATUS_OK,
                'params': params,
            }

        # Run Bayesian optimization
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

        # Convert best parameters to correct types
        best_params = space_eval(space, best)
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        best_params['random_state'] = 42

        # Get best score from trials
        best_trial = trials.best_trial
        best_cv_score = -best_trial['result']['loss']

        # Train final model with best params and evaluate on test set
        final_model = DecisionTreeRegressor(**best_params)
        final_model.fit(X_train, y_train)
        y_test_pred = final_model.predict(X_test)

        # Convert from log space
        y_test_pred = 10 ** y_test_pred
        y_test_orig = 10 ** y_test
        test_mse = mean_squared_error(y_test_orig, y_test_pred)

        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE")
            print("=" * 80)
            print(f"Best cross-validation MSE (log): {best_cv_score:.6f}")
            print(f"Test set MSE (original): {test_mse:.4e}")
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

        # CRITICAL: Filter out non-numeric columns automatically
        # This prevents "could not convert string to float" errors
        # Includes pandas sparse arrays (common for one-hot encoded MT codes)

        # Get numeric columns (standard numeric types)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Also include sparse columns (pandas sparse arrays for one-hot encoded data)
        sparse_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)]

        # Combine and remove duplicates
        all_numeric_cols = list(set(numeric_cols + sparse_cols))
        self.feature_columns = [col for col in all_numeric_cols if col not in exclude_columns]

        # Report excluded non-numeric columns if any
        all_candidate_columns = [col for col in df.columns if col not in exclude_columns]
        non_numeric_columns = [col for col in all_candidate_columns if col not in self.feature_columns]

        if len(non_numeric_columns) > 0:
            print(f"  ⚠️  Excluding {len(non_numeric_columns)} non-numeric columns:")
            for col in non_numeric_columns[:5]:  # Show first 5
                print(f"      - {col} (dtype: {df[col].dtype})")
            if len(non_numeric_columns) > 5:
                print(f"      ... and {len(non_numeric_columns) - 5} more")
            print()

        # Handle sparse DataFrames efficiently (avoid memory explosion)
        # Pandas sparse arrays are common with one-hot encoded MT codes
        X_df = df[self.feature_columns]

        # Check if DataFrame contains sparse arrays (all MT_* columns should be sparse)
        sparse_columns = [col for col in X_df.columns if isinstance(X_df[col].dtype, pd.SparseDtype)]

        if len(sparse_columns) > 0:
            # Convert to scipy sparse matrix (memory efficient)
            # We need to handle mixed sparse/dense DataFrames
            import scipy.sparse as sp

            try:
                # Try to convert sparse columns to scipy sparse matrix
                if len(sparse_columns) == len(X_df.columns):
                    # All columns are sparse - use sparse accessor
                    X = sp.csr_matrix(X_df.sparse.to_coo())
                    print(f"  → Using sparse matrix format (memory efficient)")
                else:
                    # Mixed sparse/dense - convert sparse columns, densify, then combine
                    # This is safer but uses more memory
                    print(f"  → Converting {len(sparse_columns)} sparse columns to dense (mixed DataFrame)")
                    X = X_df.values
            except (AttributeError, ValueError) as e:
                # Fallback to dense if sparse conversion fails
                print(f"  → Sparse conversion failed, using dense format")
                X = X_df.values
        else:
            # Dense data - use numpy array
            X = X_df.values

        y = df[target_column].values

        # Log-transform target for better numerical stability
        y_log = np.log10(y + 1e-10)

        # CRITICAL: Remove inf/NaN values that cause sklearn to fail
        # This can happen with AME enrichment data for unstable isotopes
        if isinstance(X, np.ndarray):
            # For dense arrays, check for inf/NaN
            finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y_log)
            if not finite_mask.all():
                n_invalid = (~finite_mask).sum()
                print(f"  ⚠️  Removing {n_invalid:,} rows with inf/NaN values ({n_invalid/len(X)*100:.2f}%)")
                X = X[finite_mask]
                y_log = y_log[finite_mask]
        # Note: scipy sparse matrices handle this differently, typically don't contain inf

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

        # Build feature matrix matching training features
        if mode == 'naive':
            # Create DataFrame with all feature columns from training
            features = []
            for energy in energies:
                feat_dict = {'Z': Z, 'A': A, 'Energy': energy}

                # Add all MT one-hot columns that exist in trained model
                for col in self.feature_columns:
                    if col.startswith('MT_'):
                        # Extract MT code from column name (e.g., 'MT_18' -> 18)
                        try:
                            mt_value = int(col.split('_')[1])
                            feat_dict[col] = 1.0 if mt_value == mt_code else 0.0
                        except (IndexError, ValueError):
                            feat_dict[col] = 0.0
                    elif col not in feat_dict:
                        # Handle any other feature columns
                        feat_dict[col] = 0.0

                features.append(feat_dict)
            df = pd.DataFrame(features)

            # Ensure all feature columns exist (fill missing with 0)
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0

            # Reorder columns to match training order
            df = df[self.feature_columns]

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

            # Ensure all feature columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0

            # Reorder columns to match training order
            df = df[self.feature_columns]

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
