# Data Loading Issues and Fixes Summary

## Executive Summary

Your data loading achieved **excellent performance** (700-1000x faster!), but has **three data quality issues** that need fixing:

1. ❌ **Missing particle vector transformation** (using wrong mode)
2. ❌ **Bookkeeping codes not filtered** (PyArrow bug)
3. ⚠️ **Energy range not fully filtered** (PyArrow bug, partially fixed)

---

## Performance Results ✅

| Metric | Result | Status |
|--------|--------|--------|
| Load Time | 1.0s (was 700-1000s) | ✅ **EXCELLENT** |
| Speedup | 700-1000x faster | ✅ **OUTSTANDING** |
| AME Enrichment | 79.4% coverage, all files loaded | ✅ **WORKING** |
| On-Demand Loading | Auto-detects data/*.mas20.txt | ✅ **WORKING** |

**Verdict**: The lean ingestion + on-demand enrichment architecture is working brilliantly!

---

## Data Quality Issues ❌

### Issue #1: Missing Particle Vector Transformation

**Problem**: You're using `mode='naive'` which does MT one-hot encoding instead of particle features.

**Evidence from output**:
```
Feature names (Naive Mode):
['Z', 'A', 'Energy', 'MT_0', 'MT_1', 'MT_2', ... 'MT_9001', 'CrossSection']
                     ^^^^^^^^^^^^^^^^^^^^^^^^ 117 MT columns (wrong!)
```

**Expected output** (with particle vector):
```
Feature names (Tier Mode):
['Z', 'A', 'N', 'Energy', 'out_n', 'out_p', 'out_a', 'out_g', 'out_f',
 'out_t', 'out_h', 'out_d', 'is_met', 'Mass_Excess_MeV', ...]
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Particle vector (correct!)
```

**Root Cause**: Three projection modes available:
- `mode='naive'` → MT one-hot encoding (intentionally bad for educational comparison)
- `mode='physics'` → Graph features (Q-value, Threshold, etc.)
- `mode='tier'` → Uses FeatureGenerator with **particle vector** ✅

**Fix**:
```python
# OLD (wrong):
df_naive = dataset_full.to_tabular(mode='naive')

# NEW (correct):
df_tier = dataset_full.to_tabular(mode='tier', tiers=['A', 'C'])
```

**Impact**: Without particle features, the model treats reactions as independent categories instead of understanding physics (e.g., (n,2n) and (n,3n) both emit neutrons).

---

### Issue #2: Bookkeeping Codes Not Filtered

**Problem**: MT 0, 1, and 9000+ are included despite `exclude_bookkeeping=True`

**Evidence from output**:
```
Top 10 Reaction Types (MT codes):
  MT   0 MT-0           : 6,019,310 measurements  ← Should be excluded!
  MT   1 MT-1           : 4,068,479 measurements  ← Should be excluded!
  MT 9000 MT-9000        :   586,482 measurements  ← Should be excluded!
```

**Why this is bad**:
- **MT 0**: Undefined reactions
- **MT 1**: Total cross-section = sum of all partial reactions (arithmetic identity)
- **MT 9000+**: Lumped covariance data (not independent physics)

**Root Cause**: PyArrow predicate pushdown doesn't correctly apply MT filters

**Fix**: Add manual filter after loading
```python
# Manual MT filter (after energy filter)
dataset_full.df = dataset_full.df[
    (dataset_full.df['MT'] != 0) &      # Exclude undefined
    (dataset_full.df['MT'] != 1) &      # Exclude total XS
    (dataset_full.df['MT'] < 9000)      # Exclude covariance
].copy()
```

**Impact**: Training on bookkeeping codes confuses the model because they're arithmetic identities, not independent physics measurements.

---

### Issue #3: Energy Range Not Fully Filtered

**Problem**: Data includes energies up to 1 GeV (should be max 20 MeV)

**Evidence from output**:
```
Selection: energy_max=2e7  (20 MeV)
Loaded:    Energy range: 1.00e-05 to 1.00e+09 eV  ← 1 GeV! 50x too high!

Manual filter removed: 1.8M measurements (10.9% of data > 20 MeV)
```

**Root Cause**: PyArrow predicate pushdown doesn't correctly apply energy filters

**Fix**: Already in place (manual energy filter), just needs to be combined with MT filter

**Impact**: Model trains on high-energy physics data (up to 1 GeV) which is not relevant for reactor physics (max 20 MeV).

---

## Recommended Code Changes

Replace cell-3 in `00_Baselines_and_Limitations.ipynb` with the code in:
**`CORRECTED_NOTEBOOK_CELL.py`**

Key changes:
1. ✅ Use `mode='tier'` instead of `mode='naive'`
2. ✅ Add manual MT filter after energy filter
3. ✅ Update comments to explain what's actually happening

---

## PyArrow Predicate Pushdown Bug Investigation

**Status**: Pending investigation

**Symptoms**:
- Energy filters not applied (`energy_max=2e7` ignored, loads up to 1e9)
- MT filters not applied (`exclude_bookkeeping=True` ignored)
- Projectile filter WORKS correctly (removes non-neutron data)

**Hypothesis**: `_build_selection_filter()` in `dataset.py:388-440` may be generating incorrect PyArrow filter expressions

**Test approach**:
```python
# Check generated filter
filter_expr = dataset._build_selection_filter(selection)
print(f"Generated filter: {filter_expr}")

# Verify fragments
import pyarrow.parquet as pq
dataset = pq.ParquetDataset('data/exfor_processed.parquet', filters=filter_expr)
print(f"Fragments after filter: {len(list(dataset.fragments))}")
```

---

## Migration Checklist

- [ ] Replace cell-3 in notebook with corrected code
- [ ] Verify output shows particle vector features (out_n, out_p, etc.)
- [ ] Verify MT 0, 1, 9000+ are excluded
- [ ] Verify energy range is 1e-5 to 2e7 (not 1e9)
- [ ] Update subsequent cells to use `df_tier` instead of `df_naive`
- [ ] Re-run notebook and verify training works correctly

---

## Files Created

- **FIXES_NEEDED.md**: Detailed technical analysis
- **CORRECTED_NOTEBOOK_CELL.py**: Drop-in replacement for cell-3
- **DATA_LOADING_SUMMARY.md**: This file (executive summary)

---

## Bottom Line

✅ **Performance**: Outstanding (700-1000x faster!)
⚠️ **Data Quality**: Needs fixes before training
📋 **Action**: Use CORRECTED_NOTEBOOK_CELL.py to replace cell-3
