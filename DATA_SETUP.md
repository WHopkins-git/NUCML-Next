# Data Setup Guide for NUCML-Next

This guide explains how to set up all required data files for NUCML-Next.

## Quick Summary

**Minimum Required:**
- X4Pro SQLite database (EXFOR experimental data)

**Recommended for Full Features:**
- AME2020/NUBASE2020 data files (5 files for tier-based features)

---

## Step 1: EXFOR Database (Required)

NUCML-Next uses IAEA EXFOR experimental nuclear cross-section data in X4Pro SQLite format.

### Option A: Sample Database (Testing)

A small sample database is included in the repository:

```bash
ls data/x4sqlite1_sample.db
# File is already present, ~40 MB
```

### Option B: Full Database (Production)

Download the complete EXFOR database for production use:

1. Visit: https://www-nds.iaea.org/x4/
2. Download: `x4sqlite1.db` (~2-4 GB)
3. Place in project root or `data/` directory

**Note:** The full database is NOT committed to GitHub due to size constraints.

---

## Step 2: AME2020/NUBASE2020 Data (Recommended)

NUCML-Next implements a **tier-based feature hierarchy** requiring multiple AME2020/NUBASE2020 files.

### Why Multiple Files?

Different nuclear properties come from different evaluation files:
- **mass_1.mas20.txt**: Basic nuclear masses and binding energies
- **rct1.mas20.txt**: Two-particle separation energies and decay Q-values
- **rct2_1.mas20.txt**: One-particle separation energies and reaction Q-values
- **nubase_4.mas20.txt**: Nuclear structure (spin, parity, half-life)
- **covariance.mas20.txt**: Uncertainty correlations (optional)

### Download All Files

```bash
# Navigate to data directory
cd data/

# Download AME2020 files (required for Tiers B, C, E)
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct2_1.mas20.txt

# Download NUBASE2020 file (required for Tier D)
wget https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt

# Optional: Download covariance data
wget https://www-nds.iaea.org/amdc/ame2020/covariance.mas20.txt
```

### Verify Downloads

```bash
ls -lh data/*.mas20.txt
```

Expected output:
```
-rw-r--r-- 1 user user  24M covariance.mas20.txt  (optional)
-rw-r--r-- 1 user user 462K mass_1.mas20.txt
-rw-r--r-- 1 user user 500K rct1.mas20.txt
-rw-r--r-- 1 user user 499K rct2_1.mas20.txt
-rw-r--r-- 1 user user 5.8K nubase_4.mas20.txt
```

---

## Step 3: Ingest EXFOR Data

Convert X4Pro SQLite to Parquet format with full AME2020/NUBASE2020 enrichment.

### Pre-Enrichment Architecture (Recommended)

**Key Insight:** Load ALL AME2020/NUBASE2020 files once during ingestion, not repeatedly during feature generation.

```bash
# Full enrichment - adds ALL tier columns to Parquet
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1_sample.db \
    --output data/exfor_enriched.parquet \
    --ame2020-dir data/
```

**What This Does:**
1. Loads ALL 5 AME2020/NUBASE2020 files from `data/` directory
2. Merges all enrichment columns into EXFOR dataframe
3. Writes complete enrichment schema to Parquet
4. Feature generation becomes simple column selection (no file I/O, no joins)

**Files Loaded:**
- `mass_1.mas20.txt` → Mass_Excess_keV, Binding_Energy_keV, Binding_Per_Nucleon_keV
- `rct1.mas20.txt` → S_2n, S_2p, Q_alpha, Q_2beta_minus, Q_ep, Q_beta_n
- `rct2_1.mas20.txt` → S_1n, S_1p, Q_4beta_minus, Q_d_alpha, Q_p_alpha, Q_n_alpha
- `nubase_4.mas20.txt` → Spin, Parity, Isomer_Level, Half_Life_s
- `covariance.mas20.txt` → (optional, not yet implemented)

**Benefits:**
- ✅ Faster feature generation (no file parsing, no joins)
- ✅ Consistent preprocessing (all users get same enrichment)
- ✅ Production-ready (single data source)
- ✅ Parquet columnar format (only loads needed columns)

### Basic Ingestion (No Enrichment)

If you don't need AME2020/NUBASE2020 features:

```bash
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1_sample.db \
    --output data/exfor_processed.parquet
```

This creates a minimal Parquet with only EXFOR data (Z, A, MT, Energy, CrossSection, Uncertainty).

---

## Step 4: Verify Setup

Test that pre-enrichment worked correctly:

```python
import pandas as pd

# Load pre-enriched Parquet
df = pd.read_parquet('data/exfor_enriched.parquet')

print(f"✓ Loaded {len(df):,} EXFOR measurements")
print(f"✓ Columns: {len(df.columns)} total")

# Check which tier columns are present
tier_columns = {
    'Tier A': ['Z', 'A', 'N', 'Energy', 'MT'],
    'Tier B/C': ['Mass_Excess_keV', 'Binding_Energy_keV', 'S_1n', 'S_2n'],
    'Tier D': ['Spin', 'Parity', 'Half_Life_s'],
    'Tier E': ['Q_alpha', 'Q_n_alpha']
}

for tier, cols in tier_columns.items():
    present = [col for col in cols if col in df.columns]
    print(f"✓ {tier}: {len(present)}/{len(cols)} columns present")

# Check enrichment coverage
if 'Mass_Excess_keV' in df.columns:
    coverage = df['Mass_Excess_keV'].notna().sum() / len(df) * 100
    print(f"✓ AME2020 coverage: {coverage:.1f}% of data points")

# Feature generation is now just column selection (no file I/O!)
from nucml_next.data.features import FeatureGenerator

gen = FeatureGenerator()  # No enricher needed - data is pre-enriched!
features = gen.generate_features(df, tiers=['A', 'C', 'D'])
print(f"✓ Generated {features.shape[1]} features from Tier A+C+D")
```

**Expected Output (if all files present):**
```
✓ Loaded 123,456 EXFOR measurements
✓ Columns: 25 total
✓ Tier A: 5/5 columns present
✓ Tier B/C: 4/4 columns present
✓ Tier D: 3/3 columns present
✓ Tier E: 2/2 columns present
✓ AME2020 coverage: 94.3% of data points
✓ Generated 32 features from Tier A+C+D
```

---

## Tier-Based Features

With all AME2020/NUBASE2020 files in place, you can use the full tier system:

### Feature Counts by Tier

| Tier | Description | Features | Required Files |
|------|-------------|----------|----------------|
| **A** | Core | 14 | None (always available) |
| **B** | + Geometric | 16 | mass_1.mas20.txt |
| **C** | + Energetics | 23 | mass_1, rct1, rct2_1 |
| **D** | + Topological | 32 | nubase_4.mas20.txt |
| **E** | + Complete Q-values | 40 | rct1, rct2_1 |

### Example Usage (Pre-Enriched Data)

```python
import pandas as pd
from nucml_next.data.features import FeatureGenerator

# Load pre-enriched Parquet (already has ALL tier columns)
df = pd.read_parquet('data/exfor_enriched.parquet')

# Filter to neutron reactions in reactor energy range
df_filtered = df[
    (df['Energy'] >= 1e-5) &
    (df['Energy'] <= 2e7)
].copy()

# Generate tier-based features (just column selection, no file I/O!)
gen = FeatureGenerator()  # No enricher needed!
features = gen.generate_features(df_filtered, tiers=['A', 'C', 'D'])

print(f"Generated {features.shape[1]} features including Tier D topological features")
print(f"Available columns: {list(features.columns)}")

# Feature generation is fast because:
# - No AME2020 file parsing
# - No dataframe joins
# - Just column selection from pre-enriched Parquet
```

**Key Differences from Legacy Approach:**
- ✅ **Old:** `NucmlDataset` loads AME2020 files every time → slow, redundant I/O
- ✅ **New:** AME2020 columns already in Parquet → fast column selection only

---

## Troubleshooting

### Missing AME2020 Files

If AME2020/NUBASE2020 files are missing, you'll see warnings:

```
WARNING: nubase_4.mas20.txt not found - Tier D features unavailable
```

**Solution:** Download the missing file(s) from https://www-nds.iaea.org/amdc/

### Available Tiers Less Than Expected

```python
enricher.get_available_tiers()  # Returns ['A', 'B', 'C', 'E']
# Missing 'D' because nubase_4.mas20.txt not found
```

**Solution:** Check which files are present:
```bash
ls -lh data/*.mas20.txt
```

### Feature Generation Errors

If you request a tier that's unavailable:

```python
# Error if nubase_4.mas20.txt is missing
df = dataset.to_tabular(mode='tier', tiers=['D'])
```

**Solution:** Only request tiers that are available, or download the required files.

---

## File Format Details

### AME2020 Files

All AME2020 files use **fixed-width Fortran-style format**:
- Headers start with `1` (page break) or `0` (line feed)
- Data lines contain isotope information
- `#` indicates estimated (non-experimental) values
- `*` indicates non-calculable values

### NUBASE2020 File

NUBASE uses a different fixed-width format:
- Columns 1-3: Mass number (A)
- Columns 5-8: Atomic number + isomer state (ZZZi)
- Columns 89-102: Spin and parity (J^π)
- Columns 70-80: Half-life with units (ys, zs, as, ..., My, Gy, Ey)

---

## Citations

If you use AME2020 or NUBASE2020 data in your work, please cite:

**AME2020:**
```
W.J. Huang, M. Wang, F.G. Kondev, G. Audi, and S. Naimi,
"The AME 2020 atomic mass evaluation (I),"
Chinese Phys. C 45, 030002 (2021).

M. Wang, W.J. Huang, F.G. Kondev, G. Audi, and S. Naimi,
"The AME 2020 atomic mass evaluation (II),"
Chinese Phys. C 45, 030003 (2021).
```

**NUBASE2020:**
```
F.G. Kondev, M. Wang, W.J. Huang, S. Naimi, and G. Audi,
"The NUBASE2020 evaluation of nuclear physics properties,"
Chinese Phys. C 45, 030001 (2021).
```

---

## Complete Setup Checklist

- [ ] X4Pro SQLite database obtained (sample or full)
- [ ] `mass_1.mas20.txt` downloaded
- [ ] `rct1.mas20.txt` downloaded
- [ ] `rct2_1.mas20.txt` downloaded
- [ ] `nubase_4.mas20.txt` downloaded
- [ ] EXFOR data ingested to Parquet
- [ ] Verified tier system with `enricher.get_available_tiers()`
- [ ] Tested feature generation with `dataset.to_tabular(mode='tier')`

**Status: Ready for tier-based feature engineering and ML training!** ✓
