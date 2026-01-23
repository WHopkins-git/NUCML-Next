"""Debug script to investigate horizontal line predictions."""
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from nucml_next.data import NucmlDataset, DataSelection
from nucml_next.baselines import DecisionTreeEvaluator

# Load small subset for debugging
print("="*80)
print("DEBUGGING HORIZONTAL LINE PREDICTIONS")
print("="*80)

# Create training selection
training_selection = DataSelection(
    projectile='neutron',
    energy_min=1e-5,
    energy_max=2e7,
    mt_mode='all_physical',
    exclude_bookkeeping=True,
    drop_invalid=True,
)

# Load dataset
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    selection=training_selection
)

# Get naive tabular projection
df_naive = dataset.to_tabular(mode='naive')

print(f"\nTraining data shape: {df_naive.shape}")
print(f"Features: {df_naive.columns.tolist()[:10]}...")  # Show first 10

# Check Energy values in training data
print(f"\nEnergy statistics in training data:")
print(f"  Min: {df_naive['Energy'].min():.2e}")
print(f"  Max: {df_naive['Energy'].max():.2e}")
print(f"  Mean: {df_naive['Energy'].mean():.2e}")
print(f"  Median: {df_naive['Energy'].median():.2e}")

# Check if Energy is log-transformed in training data
print(f"\nFirst 10 Energy values in training data:")
print(df_naive['Energy'].head(10).values)

# Train a simple model
print("\n" + "="*80)
print("Training Decision Tree...")
print("="*80)
dt_model = DecisionTreeEvaluator(max_depth=8, min_samples_leaf=10)
dt_metrics = dt_model.train(df_naive.head(100000))  # Use subset for speed

print(f"\nModel feature columns: {dt_model.feature_columns[:10]}...")  # Show first 10

# Now test prediction for U-235
print("\n" + "="*80)
print("Testing predictions for U-235...")
print("="*80)

Z, A, mt_code = 92, 235, 18
energy_range = (1.0, 100.0)

# Manually build features like predict_resonance_region does
energies = np.linspace(energy_range[0], energy_range[1], 10)  # Just 10 points for debug

print(f"\nBuilding features for {len(energies)} energy points...")
features = []
for i, energy in enumerate(energies):
    feat_dict = {'Z': Z, 'A': A, 'Energy': energy}

    # Add MT one-hot columns
    for col in dt_model.feature_columns:
        if col.startswith('MT_'):
            try:
                mt_value = int(col.split('_')[1])
                feat_dict[col] = 1.0 if mt_value == mt_code else 0.0
            except (IndexError, ValueError):
                feat_dict[col] = 0.0
        elif col not in feat_dict:
            feat_dict[col] = 0.0

    features.append(feat_dict)

    if i < 3:  # Print first 3 feature dictionaries
        print(f"\nFeature dict for energy {energy:.2f}:")
        print(f"  Z={feat_dict['Z']}, A={feat_dict['A']}, Energy={feat_dict['Energy']}")
        # Check which MT columns are 1
        mt_active = [k for k, v in feat_dict.items() if k.startswith('MT_') and v == 1.0]
        print(f"  Active MT columns: {mt_active}")

df_predict = pd.DataFrame(features)

# Ensure column order matches training
df_predict = df_predict[dt_model.feature_columns]

print(f"\nPrediction DataFrame shape: {df_predict.shape}")
print(f"First row:\n{df_predict.iloc[0]}")

# Get predictions
predictions = dt_model.predict(df_predict)

print(f"\n" + "="*80)
print("PREDICTIONS:")
print("="*80)
for i, (e, p) in enumerate(zip(energies, predictions)):
    print(f"  Energy {e:8.2f} eV → CrossSection {p:.4e} barns")

# Check if predictions are all the same
if len(set(predictions.round(10))) == 1:
    print("\n❌ ALL PREDICTIONS ARE IDENTICAL (horizontal line!)")
else:
    print(f"\n✓ Predictions vary (found {len(set(predictions.round(10)))} unique values)")

# Check feature importance
print(f"\n" + "="*80)
print("FEATURE IMPORTANCE:")
print("="*80)
importance = dt_model.get_feature_importance()
print(importance.head(15))

# Check if Energy is important
energy_importance = importance[importance['Feature'] == 'Energy']
if len(energy_importance) > 0:
    print(f"\n✓ Energy importance: {energy_importance['Importance'].values[0]:.6f}")
else:
    print("\n❌ Energy not found in feature importance!")
