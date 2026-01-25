# Data Loading Fixes Needed

## Issue Summary

The notebook `00_Baselines_and_Limitations.ipynb` has three data quality issues:

### 1. ❌ Missing Particle Vector Transformation
**Problem**: Using `mode='naive'` which does MT one-hot encoding instead of particle vector
**Current**: 117 MT columns (MT_0, MT_1, MT_2, ...)
**Expected**: 9 particle features (out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met)

**Fix**: Change from `mode='naive'` to `mode='tier'`

```python
# OLD (wrong):
df_naive = dataset_full.to_tabular(mode='naive')

# NEW (correct):
df_tier = dataset_full.to_tabular(mode='tier', tiers=['A', 'C'])
# This uses FeatureGenerator which transforms MT → particle vector
```

### 2. ❌ Bookkeeping Codes Not Filtered
**Problem**: MT 0, 1, and 9000+ are included despite `exclude_bookkeeping=True`
**Data shows**:
- MT 0: 6,019,310 measurements (should be excluded)
- MT 1: 4,068,479 measurements (should be excluded)
- MT 9000: 586,482 measurements (should be excluded)

**Fix**: Add manual filter after loading

```python
# Manual bookkeeping filter (workaround for PyArrow bug)
dataset_full.df = dataset_full.df[
    (dataset_full.df['MT'] != 0) &
    (dataset_full.df['MT'] != 1) &
    (dataset_full.df['MT'] < 9000)
].copy()
```

### 3. ❌ Energy Filter Not Working
**Problem**: PyArrow predicate pushdown doesn't filter energy correctly
**Expected**: Max 2e7 eV (20 MeV)
**Actual**: Up to 1e9 eV (1 GeV) - 50x too high!

**Fix**: Already has manual workaround, but should be applied FIRST

## Recommended Code Changes

Replace the data loading cell with:

```python
# Load FULL dataset for training with physics-aware filtering
print("=" * 80)
print("Loading training dataset with predicate pushdown...")
print("=" * 80)
dataset_full = NucmlDataset(
    data_path='../data/exfor_processed.parquet',
    mode='tabular',
    selection=training_selection
)

# ============================================================================
# WORKAROUND: PyArrow Filter Bugs
# ============================================================================
# PyArrow's predicate pushdown is not correctly applying filters
# Apply manual filters to ensure data quality

print("\n⚠️  Applying manual filters (PyArrow bug workarounds)...")
original_size = len(dataset_full.df)

# Filter 1: Energy range
dataset_full.df = dataset_full.df[
    (dataset_full.df['Energy'] >= training_selection.energy_min) &
    (dataset_full.df['Energy'] <= training_selection.energy_max)
].copy()
energy_filtered = len(dataset_full.df)

# Filter 2: Exclude bookkeeping codes (MT 0, 1, >= 9000)
dataset_full.df = dataset_full.df[
    (dataset_full.df['MT'] != 0) &
    (dataset_full.df['MT'] != 1) &
    (dataset_full.df['MT'] < 9000)
].copy()
final_size = len(dataset_full.df)

print(f"  Energy filter: {original_size:,} → {energy_filtered:,} ({energy_filtered/original_size*100:.1f}%)")
print(f"  MT filter: {energy_filtered:,} → {final_size:,} ({final_size/energy_filtered*100:.1f}%)")
print(f"  Total: {original_size:,} → {final_size:,} ({final_size/original_size*100:.1f}%)")

# ============================================================================
# Project to tabular format with TIER-BASED features (particle vector)
# ============================================================================
print("\n" + "=" * 80)
print("Projecting to tabular format (tier mode with particle vector)...")
print("=" * 80)
df_tier = dataset_full.to_tabular(mode='tier', tiers=['A', 'C'])

print(f"\n✓ Training dataset: {df_tier.shape}")
print(f"  Energy range: {df_tier['Energy'].min():.2e} to {df_tier['Energy'].max():.2e} eV")
print(f"\n📋 Features Available:")
print(f"  Tier A features: Z, A, N, Energy + 9-feature particle vector")
print(f"  Tier C features: Mass excess, binding, separation energies")
print(f"\nFeature names:")
print(df_tier.columns.tolist())
```

## PyArrow Predicate Pushdown Investigation

The `_build_selection_filter()` method in `dataset.py` needs debugging:
- Energy filters are not being applied correctly
- MT filters for bookkeeping codes are not working
- Need to verify PyArrow filter expressions are correct

**File**: `nucml_next/data/dataset.py:388-440`
**Method**: `_build_selection_filter()`

Test with:
```python
import pyarrow as pa
import pyarrow.parquet as pq

# Check what filters are actually being generated
filter_expr = dataset._build_selection_filter(selection)
print(f"Generated filter: {filter_expr}")

# Test if filter is applied
dataset = pq.ParquetDataset('data/exfor_processed.parquet', filters=filter_expr)
print(f"Fragments after filter: {len(list(dataset.fragments))}")
```
