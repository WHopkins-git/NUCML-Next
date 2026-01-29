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
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

if TYPE_CHECKING:
    from nucml_next.data.selection import TransformationConfig

logger = logging.getLogger(__name__)

# Small constant for numerical stability (avoids log(0))
# NOTE: This is kept for backward compatibility. New code should use config.target_epsilon
EPSILON = 1e-10


class TransformationPipeline:
    """
    Reversible transformation pipeline for nuclear cross-section ML.

    Implements log-scaling, standardization, and inverse transformations
    with proper handling of training/inference time parameter reuse.

    Supports multiple scaling strategies:
    - 'standard': Z-score normalization (X-μ)/σ
    - 'minmax': Min-max scaling to [0, 1]
    - 'robust': Robust scaling using median and IQR
    - 'none': No scaling
    """

    def __init__(self, config: Optional['TransformationConfig'] = None):
        """
        Initialize transformation pipeline.

        Args:
            config: Transformation configuration. If None, uses default settings
                   (log₁₀ transforms enabled, Z-score standardization)
        """
        # Import here to avoid circular dependency
        if config is None:
            from nucml_next.data.selection import TransformationConfig
            config = TransformationConfig()

        self.config = config

        # Standardization parameters (fitted on training data)
        # For 'standard' scaler
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None

        # For 'minmax' scaler
        self.feature_min_: Optional[np.ndarray] = None
        self.feature_max_: Optional[np.ndarray] = None

        # For 'robust' scaler
        self.feature_median_: Optional[np.ndarray] = None
        self.feature_iqr_: Optional[np.ndarray] = None

        # Feature columns to scale
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
        # Determine which columns to scale
        if feature_columns is None:
            # Use config if available, otherwise all numeric columns
            if self.config.scale_features is not None:
                feature_columns = self.config.scale_features
            else:
                # Default: standardize all numeric columns
                feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_columns_ = feature_columns

        # Compute scaler parameters based on scaler_type
        if self.config.scaler_type != 'none':
            X_features = X[feature_columns].values

            if self.config.scaler_type == 'standard':
                # Z-score normalization: (X - μ) / σ
                # Use nanmean/nanstd to ignore NaN values (e.g., missing AME separation energies)
                self.feature_mean_ = np.nanmean(X_features, axis=0)
                self.feature_std_ = np.nanstd(X_features, axis=0)
                # Handle all-NaN columns: nanmean/nanstd return NaN for all-NaN input
                # Replace NaN with 0 (mean) and 1 (std) so these columns become zeros after scaling
                self.feature_mean_[np.isnan(self.feature_mean_)] = 0.0
                self.feature_std_[np.isnan(self.feature_std_)] = 1.0
                # Prevent division by zero for constant features
                self.feature_std_[self.feature_std_ == 0] = 1.0

            elif self.config.scaler_type == 'minmax':
                # Min-max scaling: (X - min) / (max - min)
                # Use nanmin/nanmax to ignore NaN values
                self.feature_min_ = np.nanmin(X_features, axis=0)
                self.feature_max_ = np.nanmax(X_features, axis=0)
                # Handle all-NaN columns: nanmin/nanmax return NaN for all-NaN input
                self.feature_min_[np.isnan(self.feature_min_)] = 0.0
                self.feature_max_[np.isnan(self.feature_max_)] = 1.0
                # Prevent division by zero for constant features
                range_ = self.feature_max_ - self.feature_min_
                self.feature_max_[range_ == 0] = self.feature_min_[range_ == 0] + 1.0

            elif self.config.scaler_type == 'robust':
                # Robust scaling: (X - median) / IQR
                # Use nanmedian/nanpercentile to ignore NaN values
                self.feature_median_ = np.nanmedian(X_features, axis=0)
                q75 = np.nanpercentile(X_features, 75, axis=0)
                q25 = np.nanpercentile(X_features, 25, axis=0)
                self.feature_iqr_ = q75 - q25
                # Handle all-NaN columns: nanmedian/nanpercentile return NaN for all-NaN input
                self.feature_median_[np.isnan(self.feature_median_)] = 0.0
                self.feature_iqr_[np.isnan(self.feature_iqr_)] = 1.0
                # Prevent division by zero for constant features
                self.feature_iqr_[self.feature_iqr_ == 0] = 1.0

        self.is_fitted_ = True

        logger.info(f"Fitted transformation pipeline on {len(X)} samples")
        logger.info(f"  Scaler type: {self.config.scaler_type}")
        if self.config.scaler_type != 'none':
            logger.info(f"  Scaling {len(feature_columns)} features: {feature_columns[:5]}...")
        logger.info(f"  Log target: {self.config.log_target} (base={self.config.log_base})")
        logger.info(f"  Log energy: {self.config.log_energy} (base={self.config.energy_log_base})")

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

        # 1. Scale features based on scaler_type
        if self.config.scaler_type != 'none':
            X_features = X[self.feature_columns_].values.astype(np.float64)

            if self.config.scaler_type == 'standard':
                # Z-score normalization: (X - μ) / σ
                X_scaled = (X_features - self.feature_mean_) / self.feature_std_

            elif self.config.scaler_type == 'minmax':
                # Min-max scaling: (X - min) / (max - min)
                X_scaled = (X_features - self.feature_min_) / (self.feature_max_ - self.feature_min_)

            elif self.config.scaler_type == 'robust':
                # Robust scaling: (X - median) / IQR
                X_scaled = (X_features - self.feature_median_) / self.feature_iqr_

            # Handle inf values that can arise from:
            # - Division when std/IQR/range is very small (despite our std==0 check)
            # - Input data containing extreme values
            # Replace inf with large finite values to allow downstream processing
            X_scaled = np.nan_to_num(X_scaled, nan=np.nan, posinf=1e10, neginf=-1e10)

            X_transformed[self.feature_columns_] = X_scaled

        # 2. Log-transform energy if enabled
        if self.config.log_energy and energy is not None:
            energy_transformed = self._apply_log(energy.values, self.config.energy_log_base)

            if 'Energy' in X_transformed.columns:
                # Replace Energy column with log-transformed version
                X_transformed['Energy'] = energy_transformed
            else:
                # Add log-transformed energy as new column
                X_transformed['Energy_log'] = energy_transformed

        return X_transformed

    def _apply_log(self, values: np.ndarray, base: int) -> np.ndarray:
        """
        Apply logarithm with specified base.

        Args:
            values: Input values
            base: Logarithm base (10, 'e', or 2)

        Returns:
            Log-transformed values
        """
        if base == 10:
            return np.log10(values)
        elif base == 'e':
            return np.log(values)
        elif base == 2:
            return np.log2(values)
        else:
            raise ValueError(f"Invalid log base: {base}")

    def _inverse_log(self, values: np.ndarray, base: int) -> np.ndarray:
        """
        Apply inverse logarithm with specified base.

        Args:
            values: Log-transformed values
            base: Logarithm base (10, 'e', or 2)

        Returns:
            Original scale values
        """
        if base == 10:
            return 10 ** values
        elif base == 'e':
            return np.exp(values)
        elif base == 2:
            return 2 ** values
        else:
            raise ValueError(f"Invalid log base: {base}")

    def transform_target(self, y: pd.Series) -> pd.Series:
        """
        Apply log-transformation to target cross-sections.

        Formula: σ' = log(σ + ε) where ε is config.target_epsilon

        The epsilon term prevents log(0) and stabilizes gradients near zero.

        Args:
            y: Cross-section values (barns)

        Returns:
            Log-transformed cross-sections (or original if log_target=False)

        Example:
            >>> y_train_log = pipeline.transform_target(y_train)
        """
        if not self.config.log_target:
            return y.copy()

        y_transformed = self._apply_log(y.values + self.config.target_epsilon, self.config.log_base)

        return pd.Series(
            y_transformed,
            index=y.index,
            name='CrossSection_log' if self.config.log_target else 'CrossSection'
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

        # 1. Reverse scaling based on scaler_type
        if self.config.scaler_type != 'none':
            X_scaled = X[self.feature_columns_].values

            if self.config.scaler_type == 'standard':
                # Reverse Z-score: X = X' * σ + μ
                X_features = X_scaled * self.feature_std_ + self.feature_mean_

            elif self.config.scaler_type == 'minmax':
                # Reverse min-max: X = X' * (max - min) + min
                X_features = X_scaled * (self.feature_max_ - self.feature_min_) + self.feature_min_

            elif self.config.scaler_type == 'robust':
                # Reverse robust: X = X' * IQR + median
                X_features = X_scaled * self.feature_iqr_ + self.feature_median_

            X_original[self.feature_columns_] = X_features

        # 2. Reverse log-transform on energy if enabled
        if self.config.log_energy and energy is not None:
            energy_original = self._inverse_log(energy.values, self.config.energy_log_base)

            if 'Energy' in X_original.columns:
                X_original['Energy'] = energy_original
            elif 'Energy_log' in X_original.columns:
                X_original['Energy'] = self._inverse_log(X_original['Energy_log'].values, self.config.energy_log_base)
                X_original = X_original.drop(columns=['Energy_log'])

        return X_original

    def inverse_transform_target(self, y_log: pd.Series) -> pd.Series:
        """
        Reverse log-transformation on cross-sections.

        Formula: σ = base^(σ') - ε where ε is config.target_epsilon

        Converts log-space predictions back to physical cross-sections (barns).

        Args:
            y_log: Log-transformed cross-sections (or original if log_target=False)

        Returns:
            Cross-sections in original units (barns)

        Example:
            >>> y_pred = model.predict(X_test_transformed)
            >>> y_pred_physical = pipeline.inverse_transform_target(pd.Series(y_pred))
        """
        if not self.config.log_target:
            return y_log.copy()

        y_physical = self._inverse_log(y_log.values, self.config.log_base) - self.config.target_epsilon

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
            'config': self.config,
            'feature_columns': self.feature_columns_,
            'is_fitted': self.is_fitted_,
            # Standard scaler parameters
            'feature_mean': self.feature_mean_,
            'feature_std': self.feature_std_,
            # MinMax scaler parameters
            'feature_min': self.feature_min_,
            'feature_max': self.feature_max_,
            # Robust scaler parameters
            'feature_median': self.feature_median_,
            'feature_iqr': self.feature_iqr_,
            # Legacy epsilon for backward compatibility
            'epsilon': EPSILON,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved transformation pipeline to {filepath}")
        logger.info(f"  Scaler type: {self.config.scaler_type}")

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

        # Handle legacy files without config
        if 'config' in state:
            config = state['config']
        else:
            # Create default config for backward compatibility
            from nucml_next.data.selection import TransformationConfig
            config = TransformationConfig()
            logger.warning("Loading legacy pipeline without config. Using default TransformationConfig.")

        pipeline = cls(config=config)
        pipeline.feature_columns_ = state['feature_columns']
        pipeline.is_fitted_ = state['is_fitted']

        # Load scaler parameters based on type
        pipeline.feature_mean_ = state.get('feature_mean')
        pipeline.feature_std_ = state.get('feature_std')
        pipeline.feature_min_ = state.get('feature_min')
        pipeline.feature_max_ = state.get('feature_max')
        pipeline.feature_median_ = state.get('feature_median')
        pipeline.feature_iqr_ = state.get('feature_iqr')

        logger.info(f"Loaded transformation pipeline from {filepath}")
        logger.info(f"  Scaler type: {pipeline.config.scaler_type}")
        if pipeline.feature_columns_:
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

        params = {
            'config': self.config,
            'scaler_type': self.config.scaler_type,
            'feature_columns': self.feature_columns_,
            'n_features': len(self.feature_columns_) if self.feature_columns_ else 0,
            'log_target': self.config.log_target,
            'log_energy': self.config.log_energy,
            'target_epsilon': self.config.target_epsilon,
            'log_base': self.config.log_base,
        }

        # Add scaler-specific parameters
        if self.config.scaler_type == 'standard':
            params.update({
                'feature_mean': self.feature_mean_,
                'feature_std': self.feature_std_,
            })
        elif self.config.scaler_type == 'minmax':
            params.update({
                'feature_min': self.feature_min_,
                'feature_max': self.feature_max_,
            })
        elif self.config.scaler_type == 'robust':
            params.update({
                'feature_median': self.feature_median_,
                'feature_iqr': self.feature_iqr_,
            })

        return params

    def __repr__(self) -> str:
        """String representation of pipeline."""
        if self.is_fitted_:
            n_features = len(self.feature_columns_) if self.feature_columns_ else 0
            features_preview = self.feature_columns_[:3] if self.feature_columns_ else []
            return (
                f"TransformationPipeline(fitted=True, "
                f"scaler={self.config.scaler_type}, "
                f"log_target={self.config.log_target}, "
                f"log_energy={self.config.log_energy}, "
                f"n_features={n_features}, "
                f"features={features_preview}...)"
            )
        else:
            return (
                f"TransformationPipeline(fitted=False, "
                f"scaler={self.config.scaler_type}, "
                f"log_target={self.config.log_target})"
            )
