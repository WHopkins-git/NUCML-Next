"""
Example: Toggleable Transformations in NUCML-Next
==================================================

Demonstrates how to configure and use toggleable transformations:
- Different scaler types (standard, minmax, robust, none)
- Toggle log transformations on/off
- Custom epsilon values
- Different log bases

Author: NUCML-Next Team
Date: 2026-01-25
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from nucml_next.data.selection import DataSelection, TransformationConfig
from nucml_next.data.transformations import TransformationPipeline

# Create sample nuclear data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Z': np.random.randint(1, 100, n_samples),
    'A': np.random.randint(10, 250, n_samples),
    'N': np.random.randint(5, 150, n_samples),
    'Energy': np.random.uniform(1e3, 1e7, n_samples),
    'CrossSection': np.random.uniform(0.01, 1000, n_samples),
})

print("=" * 80)
print("NUCML-Next Toggleable Transformations Example")
print("=" * 80)
print()

# ============================================================================
# Example 1: Default Configuration (Z-score standardization, log₁₀ transforms)
# ============================================================================
print("Example 1: Default Configuration")
print("-" * 80)

config_default = TransformationConfig()
print(f"Config: {config_default}")
print()

pipeline_default = TransformationPipeline(config=config_default)
pipeline_default.fit(data, feature_columns=['Z', 'A', 'N'])

X_transformed = pipeline_default.transform(data[['Z', 'A', 'N']], energy=data['Energy'])
y_transformed = pipeline_default.transform_target(data['CrossSection'])

print(f"Original Z range: [{data['Z'].min()}, {data['Z'].max()}]")
print(f"Transformed Z range: [{X_transformed['Z'].min():.2f}, {X_transformed['Z'].max():.2f}]")
print(f"Transformed Z mean: {X_transformed['Z'].mean():.2e}, std: {X_transformed['Z'].std():.2f}")
print()
print(f"Original CrossSection range: [{data['CrossSection'].min():.2f}, {data['CrossSection'].max():.2f}]")
print(f"Transformed CrossSection range: [{y_transformed.min():.2f}, {y_transformed.max():.2f}]")
print()

# ============================================================================
# Example 2: MinMax Scaler (scales features to [0, 1])
# ============================================================================
print("Example 2: MinMax Scaler")
print("-" * 80)

config_minmax = TransformationConfig(scaler_type='minmax')
pipeline_minmax = TransformationPipeline(config=config_minmax)
pipeline_minmax.fit(data, feature_columns=['Z', 'A', 'N'])

X_minmax = pipeline_minmax.transform(data[['Z', 'A', 'N']])

print(f"Scaler type: {config_minmax.scaler_type}")
print(f"Transformed Z range: [{X_minmax['Z'].min():.2f}, {X_minmax['Z'].max():.2f}]")
print(f"Transformed A range: [{X_minmax['A'].min():.2f}, {X_minmax['A'].max():.2f}]")
print(f"Transformed N range: [{X_minmax['N'].min():.2f}, {X_minmax['N'].max():.2f}]")
print()

# ============================================================================
# Example 3: Robust Scaler (uses median and IQR)
# ============================================================================
print("Example 3: Robust Scaler (Outlier-Resistant)")
print("-" * 80)

config_robust = TransformationConfig(scaler_type='robust')
pipeline_robust = TransformationPipeline(config=config_robust)
pipeline_robust.fit(data, feature_columns=['Z', 'A', 'N'])

X_robust = pipeline_robust.transform(data[['Z', 'A', 'N']])

print(f"Scaler type: {config_robust.scaler_type}")
print(f"Transformed Z median: {X_robust['Z'].median():.2e}")
print(f"Transformed A median: {X_robust['A'].median():.2e}")
print(f"Transformed N median: {X_robust['N'].median():.2e}")
print()

# ============================================================================
# Example 4: No Scaling (raw features)
# ============================================================================
print("Example 4: No Scaling (Raw Features)")
print("-" * 80)

config_none = TransformationConfig(scaler_type='none')
pipeline_none = TransformationPipeline(config=config_none)
pipeline_none.fit(data, feature_columns=['Z', 'A', 'N'])

X_none = pipeline_none.transform(data[['Z', 'A', 'N']])

print(f"Scaler type: {config_none.scaler_type}")
print(f"Features unchanged: {np.allclose(X_none['Z'].values, data['Z'].values)}")
print()

# ============================================================================
# Example 5: Toggle Log Transformations
# ============================================================================
print("Example 5: Toggle Log Transformations")
print("-" * 80)

# Without log transform
config_no_log = TransformationConfig(
    log_target=False,
    log_energy=False,
    scaler_type='none'
)
pipeline_no_log = TransformationPipeline(config=config_no_log)

y_no_log = pipeline_no_log.transform_target(data['CrossSection'])

print(f"Log target: {config_no_log.log_target}")
print(f"CrossSection unchanged: {np.allclose(y_no_log.values, data['CrossSection'].values)}")
print()

# ============================================================================
# Example 6: Custom Epsilon Value
# ============================================================================
print("Example 6: Custom Epsilon Value")
print("-" * 80)

config_custom_epsilon = TransformationConfig(
    target_epsilon=1e-8,  # Larger epsilon for numerical stability
    log_base=10
)
pipeline_custom = TransformationPipeline(config=config_custom_epsilon)

# Test with very small cross-sections
y_small = pd.Series([1e-12, 1e-10, 1e-8, 1e-6])
y_log_custom = pipeline_custom.transform_target(y_small)
y_reconstructed = pipeline_custom.inverse_transform_target(y_log_custom)

print(f"Epsilon: {config_custom_epsilon.target_epsilon:.1e}")
print(f"Original small values: {y_small.values}")
print(f"Log-transformed: {y_log_custom.values}")
print(f"Reconstructed: {y_reconstructed.values}")
print(f"Reconstruction error: {np.abs(y_reconstructed.values - y_small.values).max():.2e}")
print()

# ============================================================================
# Example 7: Different Log Bases
# ============================================================================
print("Example 7: Different Log Bases")
print("-" * 80)

test_value = pd.Series([100.0])

config_log10 = TransformationConfig(log_base=10)
pipeline_log10 = TransformationPipeline(config=config_log10)
y_log10 = pipeline_log10.transform_target(test_value)

config_loge = TransformationConfig(log_base='e')
pipeline_loge = TransformationPipeline(config=config_loge)
y_loge = pipeline_loge.transform_target(test_value)

config_log2 = TransformationConfig(log_base=2)
pipeline_log2 = TransformationPipeline(config=config_log2)
y_log2 = pipeline_log2.transform_target(test_value)

print(f"Original value: {test_value.values[0]}")
print(f"log₁₀(100): {y_log10.values[0]:.4f} (expected: ~2.0)")
print(f"ln(100): {y_loge.values[0]:.4f} (expected: ~4.6)")
print(f"log₂(100): {y_log2.values[0]:.4f} (expected: ~6.6)")
print()

# ============================================================================
# Example 8: Using with DataSelection and NucmlDataset
# ============================================================================
print("Example 8: Integration with DataSelection")
print("-" * 80)

# Create custom transformation config
transform_config = TransformationConfig(
    scaler_type='minmax',
    log_target=True,
    log_energy=True,
    target_epsilon=1e-9,
    log_base='e'
)

# Create data selection with custom transformations
selection = DataSelection(
    projectile='neutron',
    energy_min=1e-5,
    energy_max=2e7,
    mt_mode='reactor_core',
    tiers=['A', 'B', 'C'],
    transformation_config=transform_config
)

print(f"DataSelection summary:")
print(selection)
print()

print("To use with NucmlDataset:")
print(">>> from nucml_next.data import NucmlDataset")
print(">>> dataset = NucmlDataset(")
print("...     data_path='data/exfor_processed.parquet',")
print("...     selection=selection")
print("... )")
print(">>> pipeline = dataset.get_transformation_pipeline()")
print(">>> # Pipeline will use the transformation_config from selection")
print()

# ============================================================================
# Example 9: Comparing Different Scalers
# ============================================================================
print("Example 9: Comparing Different Scalers")
print("-" * 80)

scalers = ['standard', 'minmax', 'robust', 'none']
results = {}

for scaler_type in scalers:
    config = TransformationConfig(scaler_type=scaler_type)
    pipeline = TransformationPipeline(config=config)
    pipeline.fit(data, feature_columns=['Z'])
    X_scaled = pipeline.transform(data[['Z']])
    results[scaler_type] = X_scaled['Z']

print(f"{'Scaler':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
print("-" * 63)
print(f"{'Original':<15} {data['Z'].min():<12.2f} {data['Z'].max():<12.2f} "
      f"{data['Z'].mean():<12.2f} {data['Z'].std():<12.2f}")
for scaler_type, values in results.items():
    print(f"{scaler_type:<15} {values.min():<12.2f} {values.max():<12.2f} "
          f"{values.mean():<12.2f} {values.std():<12.2f}")
print()

print("=" * 80)
print("Examples Complete!")
print("=" * 80)
print()
print("Summary:")
print("- Use 'standard' (Z-score) for most ML algorithms (default)")
print("- Use 'minmax' for neural networks or algorithms sensitive to scale")
print("- Use 'robust' for data with outliers")
print("- Use 'none' to disable scaling (for algorithms that don't require it)")
print("- Toggle log transforms based on your data distribution")
print("- Adjust epsilon for numerical stability with very small cross-sections")
