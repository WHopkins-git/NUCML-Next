"""
Reversible Transformation Pipeline for Nuclear Cross-Section ML
================================================================

Implements standardized transformations for nuclear cross-section data:
1. Log-scaling for cross-sections and energies (stabilizes gradients)
2. StandardScaler for nuclear features (centers data at zero)
3. Inverse transformations for predictions (converts back to physical units)

Mathematical Transformations:
-----------------------------
1. Cross-section log-transform:
   Forward:  σ' = log₁₀(σ + 10⁻¹⁰)
   Inverse:  σ = 10^(σ') - 10⁻¹⁰

2. Energy log-transform:
   Forward:  E' = log₁₀(E)
   Inverse:  E = 10^(E')

3. Feature standardization (Z-score normalization):
   Forward:  X' = (X - μ) / σ
   Inverse:  X = X' * σ + μ

Pipeline Hygiene:
-----------------
- All transformations are reversible (fit/transform/inverse_transform)
- Scaler parameters (μ, σ) stored for inference time
- Prevents data leakage: fit only on training set, transform train/val/test
- Thread-safe: Can be pickled and loaded for production deployment

Usage:
------
    from nucml_next.data.transformations import TransformationPipeline

    # Create pipeline
    pipeline = TransformationPipeline()

    # Fit on training data
    X_train_transformed = pipeline.fit_transform(
        X_train,
        y_train,
        energy=energy_train,
        feature_columns=['Z', 'A', 'N', 'R_fm', 'kR', 'Mass_Excess_MeV']
    )

    # Transform validation/test data (using fitted parameters)
    X_val_transformed = pipeline.transform(X_val, energy_val)
    y_val_transformed = pipeline.transform_target(y_val)

    # Make predictions and convert back to physical units
    y_pred_log = model.predict(X_val_transformed)
    y_pred_physical = pipeline.inverse_transform_target(y_pred_log)

References:
-----------
- Valdez 2021 PhD Thesis (feature engineering best practices)
- sklearn.preprocessing.StandardScaler (Z-score normalization)
- Log-transform for positivity constraints in nuclear physics
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

# Small constant for numerical stability (avoids log(0))
EPSILON = 1e-10


class TransformationPipeline:
    """
    Reversible transformation pipeline for nuclear cross-section ML.

    Implements log-scaling, standardization, and inverse transformations
    with proper handling of training/inference time parameter reuse.
    """

    def __init__(self):
        """Initialize transformation pipeline."""
        # Standardization parameters (fitted on training data)
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.feature_columns_: Optional[List[str]] = None

        # Track whether pipeline has been fitted
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        energy: Optional[pd.Series] = None,
        feature_columns: Optional[List[str]] = None
    ) -> 'TransformationPipeline':
        """
        Fit transformation parameters on training data.

        Computes mean and standard deviation for each feature to enable
        Z-score normalization. These parameters are stored and reused
        for transform() calls on validation/test data.

        Args:
            X: Feature matrix (DataFrame)
            y: Target cross-sections (Series) - optional, not used for fitting
            energy: Incident energies (Series) - optional, not used for fitting
            feature_columns: List of columns to standardize
                           If None, standardizes all numeric columns

        Returns:
            self (fitted pipeline)

        Example:
            >>> pipeline = TransformationPipeline()
            >>> pipeline.fit(X_train, y_train, energy_train,
            ...              feature_columns=['Z', 'A', 'N', 'R_fm', 'Mass_Excess_MeV'])
        """
        # Determine which columns to standardize
        if feature_columns is None:
            # Default: standardize all numeric columns
            feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_columns_ = feature_columns

        # Compute mean and std for standardization
        X_features = X[feature_columns].values
        self.feature_mean_ = np.mean(X_features, axis=0)
        self.feature_std_ = np.std(X_features, axis=0)

        # Prevent division by zero for constant features
        self.feature_std_[self.feature_std_ == 0] = 1.0

        self.is_fitted_ = True

        logger.info(f"Fitted transformation pipeline on {len(X)} samples")
        logger.info(f"  Standardizing {len(feature_columns)} features: {feature_columns[:5]}...")

        return self

    def transform(
        self,
        X: pd.DataFrame,
        energy: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply transformations to features and energy.

        Applies standardization to features and log-scaling to energy
        using parameters fitted during fit().

        Args:
            X: Feature matrix (DataFrame)
            energy: Incident energies (Series) - if provided, log-transformed

        Returns:
            Transformed DataFrame with standardized features

        Raises:
            RuntimeError: If pipeline not fitted yet

        Example:
            >>> X_test_transformed = pipeline.transform(X_test, energy_test)
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_transformed = X.copy()

        # 1. Standardize features (Z-score normalization)
        X_features = X[self.feature_columns_].values
        X_standardized = (X_features - self.feature_mean_) / self.feature_std_
        X_transformed[self.feature_columns_] = X_standardized

        # 2. Log-transform energy if provided
        if energy is not None:
            if 'Energy' in X_transformed.columns:
                # Replace Energy column with log-transformed version
                X_transformed['Energy'] = np.log10(energy.values)
            else:
                # Add log-transformed energy as new column
                X_transformed['Energy_log'] = np.log10(energy.values)

        return X_transformed

    def transform_target(self, y: pd.Series) -> pd.Series:
        """
        Apply log-transformation to target cross-sections.

        Formula: σ' = log₁₀(σ + ε) where ε = 10⁻¹⁰

        The epsilon term prevents log(0) and stabilizes gradients near zero.

        Args:
            y: Cross-section values (barns)

        Returns:
            Log-transformed cross-sections

        Example:
            >>> y_train_log = pipeline.transform_target(y_train)
        """
        return pd.Series(
            np.log10(y.values + EPSILON),
            index=y.index,
            name='CrossSection_log'
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        energy: Optional[pd.Series] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit pipeline and transform data in one step.

        Convenience method that calls fit() followed by transform().

        Args:
            X: Feature matrix
            y: Target cross-sections (optional)
            energy: Incident energies (optional)
            feature_columns: Columns to standardize (optional)

        Returns:
            Tuple of (X_transformed, y_transformed)

        Example:
            >>> X_train_t, y_train_t = pipeline.fit_transform(
            ...     X_train, y_train, energy_train,
            ...     feature_columns=['Z', 'A', 'N', 'R_fm']
            ... )
        """
        self.fit(X, y, energy, feature_columns)
        X_transformed = self.transform(X, energy)
        y_transformed = self.transform_target(y) if y is not None else None

        return X_transformed, y_transformed

    def inverse_transform(
        self,
        X: pd.DataFrame,
        energy: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Reverse standardization and log-transforms on features.

        Converts transformed features back to original scale:
        - Standardized features: X = X' * σ + μ
        - Log energy: E = 10^(E')

        Args:
            X: Transformed feature matrix
            energy: Log-transformed energies (optional)

        Returns:
            Features in original scale

        Raises:
            RuntimeError: If pipeline not fitted

        Example:
            >>> X_original = pipeline.inverse_transform(X_transformed)
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_original = X.copy()

        # 1. Reverse standardization
        X_standardized = X[self.feature_columns_].values
        X_features = X_standardized * self.feature_std_ + self.feature_mean_
        X_original[self.feature_columns_] = X_features

        # 2. Reverse log-transform on energy if provided
        if energy is not None:
            if 'Energy' in X_original.columns:
                X_original['Energy'] = 10 ** energy.values
            elif 'Energy_log' in X_original.columns:
                X_original['Energy'] = 10 ** X_original['Energy_log'].values
                X_original = X_original.drop(columns=['Energy_log'])

        return X_original

    def inverse_transform_target(self, y_log: pd.Series) -> pd.Series:
        """
        Reverse log-transformation on cross-sections.

        Formula: σ = 10^(σ') - ε where ε = 10⁻¹⁰

        Converts log-space predictions back to physical cross-sections (barns).

        Args:
            y_log: Log-transformed cross-sections

        Returns:
            Cross-sections in original units (barns)

        Example:
            >>> y_pred = model.predict(X_test_transformed)
            >>> y_pred_physical = pipeline.inverse_transform_target(pd.Series(y_pred))
        """
        y_physical = 10 ** y_log.values - EPSILON

        # Ensure non-negative cross-sections (clip numerical artifacts)
        y_physical = np.maximum(y_physical, 0.0)

        return pd.Series(
            y_physical,
            index=y_log.index,
            name='CrossSection'
        )

    def save(self, filepath: str) -> None:
        """
        Save fitted pipeline parameters to disk.

        Serializes mean, std, and feature columns for deployment.

        Args:
            filepath: Path to save pickle file

        Example:
            >>> pipeline.save('models/transformation_pipeline.pkl')
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted pipeline. Call fit() first.")

        state = {
            'feature_mean': self.feature_mean_,
            'feature_std': self.feature_std_,
            'feature_columns': self.feature_columns_,
            'is_fitted': self.is_fitted_,
            'epsilon': EPSILON,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved transformation pipeline to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TransformationPipeline':
        """
        Load fitted pipeline from disk.

        Restores mean, std, and feature columns from saved file.

        Args:
            filepath: Path to pickle file

        Returns:
            Loaded TransformationPipeline

        Example:
            >>> pipeline = TransformationPipeline.load('models/transformation_pipeline.pkl')
            >>> X_test_transformed = pipeline.transform(X_test)
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        pipeline = cls()
        pipeline.feature_mean_ = state['feature_mean']
        pipeline.feature_std_ = state['feature_std']
        pipeline.feature_columns_ = state['feature_columns']
        pipeline.is_fitted_ = state['is_fitted']

        logger.info(f"Loaded transformation pipeline from {filepath}")
        logger.info(f"  Features: {pipeline.feature_columns_[:5]}...")

        return pipeline

    def get_params(self) -> Dict[str, Any]:
        """
        Get transformation parameters.

        Returns:
            Dictionary with fitted parameters (mean, std, feature names)

        Example:
            >>> params = pipeline.get_params()
            >>> print(f"Standardizing {len(params['feature_columns'])} features")
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        return {
            'feature_mean': self.feature_mean_,
            'feature_std': self.feature_std_,
            'feature_columns': self.feature_columns_,
            'n_features': len(self.feature_columns_),
            'epsilon': EPSILON,
        }

    def __repr__(self) -> str:
        """String representation of pipeline."""
        if self.is_fitted_:
            return (
                f"TransformationPipeline(fitted=True, "
                f"n_features={len(self.feature_columns_)}, "
                f"features={self.feature_columns_[:3]}...)"
            )
        else:
            return "TransformationPipeline(fitted=False)"
