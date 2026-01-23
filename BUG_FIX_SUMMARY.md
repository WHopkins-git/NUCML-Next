# Bug Fix: Horizontal Line Predictions

## Problem
ML models (Decision Tree and XGBoost) were producing horizontal line predictions instead of proper cross-section curves.

## Root Cause
The EXFOR processed dataset contained only **13 measurements** (after ingestion) instead of the expected thousands:
- No U-235 fission data (MT=18) at all
- Only 4 Cl-35 measurements
- Insufficient training data for ML models to learn energy-dependent patterns

### Why Horizontal Lines?
With such limited data, the Decision Tree had:
- **Tree depth: 1** (only one split)
- **Energy feature importance: 0.0** (completely ignored)
- **MT feature importance: 1.0** (only used reaction type)

The model was predicting constant values based solely on (Z, A, MT) combination, completely ignoring Energy!

## Solution
Created a synthetic demonstration dataset (`scripts/create_demo_data.py`) with:

### Dataset Contents:
- **12,100 total measurements** (11,897 after validity filtering)
- **U-235 Fission**: 3,000 points with realistic Breit-Wigner resonances
  - Energy range: 0.01 eV to 10 MeV
  - Includes famous resonances at 0.29 eV, 1.14 eV, 2.03 eV, etc.
  - Fast neutron region behavior
- **Cl-35 (n,p)**: 397 points with threshold behavior
  - Energy range: 0.6 MeV (threshold) to 20 MeV
  - Realistic cross-section values (peak ~0.15 barns)
- **Additional isotopes** for training diversity:
  - Fe-56 elastic (1,500 points)
  - C-12 elastic (1,500 points)
  - O-16 elastic (1,500 points)
  - Pu-239 fission (2,000 points)

### Physics Realism:
1. **U-235 Fission Resonances**: Modeled with Breit-Wigner formula
   ```python
   σ(E) = σ₀ * Γ² / ((E - E_r)² + Γ²/4)
   ```
2. **1/v Behavior**: Capture cross sections at thermal energies
3. **Threshold Behavior**: Cl-35 (n,p) has Q-value requiring ~0.6 MeV minimum energy
4. **Realistic Noise**: 5-10% uncertainties matching experimental data

## How to Use

### Option 1: Use Demonstration Dataset (Already Created)
The demonstration dataset is ready to use:
```bash
# Already created - just run the notebook
jupyter notebook notebooks/00_Baselines_and_Limitations.ipynb
```

### Option 2: Re-create Demonstration Dataset
```bash
python scripts/create_demo_data.py
```

### Option 3: Use Full EXFOR Database (Production)
For production use with real EXFOR data:
```bash
# Download x4sqlite1.db from https://www-nds.iaea.org/x4/
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db
```

## Expected Results (After Fix)
With the new demonstration dataset:
- Models will learn energy-dependent patterns
- U-235 fission plots will show resonance structure (jagged but following peaks)
- Cl-35 (n,p) plots will show threshold behavior
- Energy will have non-zero feature importance
- The notebook will demonstrate why classical ML fails (staircase effect, not horizontal lines)

## Files Changed
- `scripts/create_demo_data.py` (NEW): Synthetic data generator
- `data/exfor_processed.parquet/*` (UPDATED): New demonstration dataset

## Verification
```python
from nucml_next.data import NucmlDataset

dataset = NucmlDataset(data_path='data/exfor_processed.parquet', mode='tabular')
print(f"Dataset size: {len(dataset.df):,} measurements")

# Verify U-235 fission exists
u235_fis = dataset.df[(dataset.df['Z'] == 92) &
                      (dataset.df['A'] == 235) &
                      (dataset.df['MT'] == 18)]
print(f"U-235 Fission: {len(u235_fis):,} measurements")
# Expected: ~3,000 measurements
```

## For Full IAEA EXFOR Data
The sample database (`x4sqlite1_sample.db`) contains limited data. For production work:
1. Download full database: https://www-nds.iaea.org/x4/
2. Run ingestion: `python scripts/ingest_exfor.py --x4-db x4sqlite1.db`
3. Expected size: ~3-10 million measurements

## Summary
The bug was **insufficient training data**, not a code issue. The new demonstration dataset provides realistic nuclear cross-section data with proper physics behavior, allowing the notebooks to demonstrate classical ML limitations correctly.
