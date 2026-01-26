# NUCML-Next: Next-Generation Nuclear Data Evaluation

**Physics-Informed Deep Learning for Nuclear Cross-Section Prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

NUCML-Next is a **production-ready framework** for nuclear data evaluation using physics-informed machine learning. It implements the evolution from classical ML (XGBoost, Decision Trees) to physics-informed deep learning (GNNs + Transformers) using real experimental nuclear cross-section data from IAEA EXFOR.

### The Problem We Solve

**The Validation Paradox:**
> Low MSE on test data ≠ Safe reactor predictions!

Classical ML models can achieve low error on geometric metrics while producing unphysical and unsafe reactor predictions.

**The Solution:**
- **Graph Neural Networks** → Learn nuclear topology
- **Transformers** → Smooth cross-section curves
- **Physics-Informed Loss** → Enforce constraints
- **Sensitivity Weighting** → Prioritize reactor-critical reactions

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/WHopkins-git/NUCML-Next.git
cd NUCML-Next

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

**Step 1: Obtain X4Pro SQLite Database**

NUCML-Next uses the X4Pro SQLite format for EXFOR data ingestion.

**Option A: Use Sample Database (Quick Start)**
```bash
# Sample database included in repository
ls data/x4sqlite1_sample.db  # Contains subset for testing
```

**Option B: Download Full Database (Production)**
- Visit: https://www-nds.iaea.org/x4/
- Download: x4sqlite1.db (~2-4 GB)
- Place in project directory or specify custom path

> **Note:** The full X4Pro database is NOT committed to GitHub due to size.
> Only a small sample database (`data/x4sqlite1_sample.db`) is included.

**Step 2: Ingest X4 to Parquet (with Full AME2020/NUBASE2020 Enrichment)**

```bash
# Basic ingestion (no enrichment)
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1_sample.db \
    --output data/exfor_processed.parquet

# Recommended: Full enrichment (all tier columns added to Parquet)
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1_sample.db \
    --output data/exfor_enriched.parquet \
    --ame2020-dir data/
```

**Pre-Enrichment Architecture:**
- The `--ame2020-dir` option loads ALL 5 AME2020/NUBASE2020 files during ingestion
- All enrichment columns (Tier B-E) are written to the Parquet file
- Feature selection becomes simple column selection (no file I/O, no joins)
- Parquet columnar format only loads needed columns anyway → fast and efficient

**Step 3: Load in notebooks**
```python
from nucml_next.data import NucmlDataset

# Load EXFOR data (optimized for large files)
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    # Automatic optimizations: column pruning, memory mapping, lazy graph building
    filters={'Z': [92], 'MT': [18, 102]}  # Optional: Filter for faster loading
)
```

**Step 4: Run training notebooks**
```bash
jupyter notebook notebooks/00_Baselines_and_Limitations.ipynb
```

---

## Performance Optimization

### Loading Large EXFOR Databases (4.7GB+)

The full EXFOR database contains ~18M measurements and can be slow to load. NUCML-Next includes automatic optimizations:

**Automatic Optimizations (v1.1.0+):**
- ✅ **Column pruning**: Only reads essential columns (50-60% faster)
- ✅ **Memory mapping**: Reduced RAM usage, faster I/O
- ✅ **Multi-threaded read**: Parallel decompression
- ✅ **Lazy graph building**: Graph built on first access

**Expected Load Times:**
| Method | Load Time | RAM Usage |
|--------|-----------|-----------|
| Full database (optimized) | 60-120s | 4-6 GB |
| Filtered (e.g., U-235 only) | 2-10s | 200-800 MB |
| Lazy load (prototyping) | <1s | <100 MB |

**Recommended: Use Filters for Faster Loading**

```python
# Load only uranium fission/capture data (10x faster)
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    filters={
        'Z': [92],           # Uranium only
        'MT': [18, 102]      # Fission and capture
    }
)
# Load time: ~5 seconds instead of 60-120 seconds
```

**For Full Training (All Data):**

```python
# Load entire database with automatic optimizations
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular'
)
# Load time: 60-120 seconds (optimized from 600+ seconds)
# Shows timing breakdown: Parquet read + Arrow→Pandas conversion
```

📖 **See [PERFORMANCE.md](PERFORMANCE.md) for detailed optimization guide**

---

## Data Sources

### EXFOR Database (Required)

NUCML-Next uses real experimental nuclear cross-section data from the IAEA EXFOR database via the X4Pro SQLite format.

**Why X4Pro SQLite?**
- Single-file database (no directory recursion)
- Efficient querying with SQL
- Standardized schema across EXFOR releases
- Faster ingestion than JSON formats

**Ingestion Process:**

```python
from nucml_next.ingest import ingest_x4

# Ingest X4 database to Parquet
df = ingest_x4(
    x4_db_path='data/x4sqlite1.db',
    output_path='data/exfor_processed.parquet',
    ame2020_path='data/ame2020.txt',  # Optional: for enhanced isotope features
)
```

**Or use the convenience helper:**

```python
from nucml_next.examples import quick_ingest

# Automatically uses sample database if no path specified
df = quick_ingest()
```

**Output:**
- Partitioned Parquet dataset by Z/A/MT
- Optional AME2020-enriched isotope features
- Preserves experimental uncertainties
- Standard schema: [Entry, Z, A, MT, Energy, CrossSection, Uncertainty]

### AME2020/NUBASE2020 Integration (Recommended)

NUCML-Next implements a **tier-based feature hierarchy** (Valdez 2021) using the complete AME2020 and NUBASE2020 nuclear data suite.

**Required Data Files:**

All files are available from https://www-nds.iaea.org/amdc/

```bash
cd data/

# Core AME2020 files (required for Tiers B, C, E)
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct2_1.mas20.txt

# NUBASE2020 file (required for Tier D)
wget https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt

# Optional: Covariance data
wget https://www-nds.iaea.org/amdc/ame2020/covariance.mas20.txt
```

**File-to-Feature Mapping:**

| File | Features Provided | Tiers |
|------|------------------|-------|
| `mass_1.mas20.txt` | Mass excess, binding energy, nuclear radius | B, C |
| `rct1.mas20.txt` | S(2n), S(2p), Q(α), Q(2β⁻) | C |
| `rct2_1.mas20.txt` | S(1n), S(1p), all reaction Q-values | C, E |
| `nubase_4.mas20.txt` | Spin, parity, half-life, isomeric states | D |
| `covariance.mas20.txt` | Mass uncertainty correlations | (optional) |

**Tier-Based Feature System:**

- **Tier A** (Core): Z, A, N, Energy + **9-feature Numerical Particle Vector** → **13 features**
- **Tier B** (Geometric): + Nuclear radius (R_fm), kR parameter → **15 features**
- **Tier C** (Energetics): + Mass excess, binding energy, separation energies → **22 features**
- **Tier D** (Topological): + Spin, parity, valence, pairing, magic numbers → **30 features**
- **Tier E** (Complete): + All reaction Q-values (8 Q-values) → **38 features**

**Numerical Particle Vector (Tier A):**

Instead of categorical MT codes, NUCML-Next uses a **9-feature physics-based coordinate system**:

```
[out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met]
```

- `out_n`: Neutrons emitted (integer multiplicity)
- `out_p`: Protons emitted (integer)
- `out_a`: Alpha particles emitted (integer)
- `out_g`: Gamma emission indicator (0/1)
- `out_f`: Fission indicator (0/1)
- `out_t`: Tritons (³H) emitted (integer)
- `out_h`: Helions (³He) emitted (integer)
- `out_d`: Deuterons emitted (integer)
- `is_met`: Isomeric state indicator (0/1)

**Benefits for ML:**
- ✅ **Decision Trees**: Split on physical observables (`if out_n > 2`)
- ✅ **Neural Networks**: Learn mass/charge conservation patterns
- ✅ **Interpretability**: Feature importance maps to nuclear physics
- ✅ **MT Preserved**: Original MT column kept for user queries (`df[df['MT']==102]`)

**Example Mappings:**
```python
MT=2   (Elastic):   [1,0,0,0,0,0,0,0,0]  # 1 neutron out
MT=18  (Fission):   [0,0,0,0,1,0,0,0,0]  # Fission indicator
MT=102 (Capture):   [0,0,0,1,0,0,0,0,0]  # Gamma emission
MT=103 (n,p):       [0,1,0,0,0,0,0,0,0]  # 1 proton out
MT=16  (n,2n):      [2,0,0,0,0,0,0,0,0]  # 2 neutrons out
MT=600+ (n,p iso):  [0,1,0,0,0,0,0,0,1]  # Proton + isomer
```

**Usage in Feature Generation:**

```python
from nucml_next.data import NucmlDataset
from nucml_next.data.selection import DataSelection

# Select features by tier (automatic AME2020 loading)
selection = DataSelection(
    projectile='neutron',
    energy_min=1e-5,
    energy_max=2e7,
    mt_mode='all_physical',
    tiers=['A', 'B', 'C']  # Core + Geometric + Energetics
)

dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    selection=selection
)

# Generate tier-based features
df = dataset.to_tabular(mode='tier')
# Returns DataFrame with 23 features including energetics from AME2020
```

**Recommended: Pre-Enrichment During Ingestion**

For production use, enrich data during ingestion with all AME2020/NUBASE2020 files:

```bash
# Download all required files first
cd data/
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/rct2_1.mas20.txt
wget https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt

# Ingest with full enrichment
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1.db \
    --output data/exfor_enriched.parquet \
    --ame2020-dir data/
```

**Architecture Benefits:**
- ✅ Load AME2020 files once during ingestion (not every feature generation call)
- ✅ All enrichment columns in Parquet → consistent preprocessing for all users
- ✅ Feature generation = column selection → no file I/O, no joins, faster
- ✅ Parquet columnar format → only loads needed columns anyway

**Legacy Approach (Not Recommended):**
If AME2020 files are not provided during ingestion, feature generation will load files on-demand (slower, redundant I/O). The recommended approach is pre-enrichment during ingestion.

**Citations:**

If you use AME2020 or NUBASE2020 data, please cite:

- **AME2020:** W.J. Huang et al., "The AME 2020 atomic mass evaluation," Chinese Phys. C **45**, 030002 (2021)
- **NUBASE2020:** F.G. Kondev et al., "The NUBASE2020 evaluation of nuclear physics properties," Chinese Phys. C **45**, 030001 (2021)

### Technical: X4Pro Schema Support

The X4Ingestor automatically detects and handles multiple X4Pro database schemas:

1. **Official X4Pro Schema** (x4pro_ds + x4pro_x5z):
   - Metadata in `x4pro_ds` table (DatasetID, Z, Target, MT, reaction codes)
   - Cross-section data in JSON format (`x4pro_x5z.jx5z` column)
   - Joins on `DatasetID` (string field, e.g., "30649005S")
   - Parses `c5data` JSON: `x1` (energy), `y` (cross-section), `dy` (uncertainty)
   - Extracts Z/A from target strings (e.g., "U-235" → Z=92, A=235)

2. **Alternative X4Pro Schema** (x4pro_ds + x4pro_c5dat):
   - Fallback for databases where c5dat table is populated instead of JSON

3. **Legacy Schemas**:
   - Simple `data_points` table
   - Joined tables (reactions + energies + cross_sections)

**Column Mapping:**
```
X4Pro → NUCML-Next
─────────────────────
DatasetID → Entry
En        → Energy
Data      → CrossSection
dData     → Uncertainty
```

**Note:** DatasetID values are strings and must be quoted properly in SQL queries to avoid tokenization errors.

---

## Features

### v1.1.0-alpha (Production-Ready)
✓ **X4Pro SQLite ingestor** with AME2020 enrichment
✓ **Partitioned Parquet** data fabric for large-scale datasets
✓ **Real experimental data** from IAEA EXFOR database
✓ **No simulation or synthetic data** - production-grade only
✓ **Minimal examples helper** for notebooks and documentation

### Core Framework
✓ **Dual-view data architecture** (Graph + Tabular)
✓ **Baseline models** (Decision Trees, XGBoost)
✓ **GNN-Transformer** architecture
✓ **Physics-informed loss** functions
✓ **OpenMC integration** for validation
✓ **Sensitivity analysis** for reactor safety

---

## Architecture

### Package Structure

```
nucml_next/
├── ingest/                    # Data ingestion
│   └── x4.py                  # X4Pro SQLite ingestor
├── data/                      # Data handling
│   ├── dataset.py             # NucmlDataset with dual-view
│   ├── graph_builder.py       # Chart of Nuclides graph
│   └── tabular_projector.py   # Graph → Tabular projection
├── examples/                  # Convenience helpers for notebooks
│   └── helpers.py             # Quick-start functions
├── baselines/                 # Classical ML baselines
│   ├── decision_tree_evaluator.py
│   └── xgboost_evaluator.py
├── model/                     # Deep learning models
│   ├── nuclide_gnn.py         # Graph Neural Network
│   ├── energy_transformer.py  # Transformer for σ(E)
│   └── gnn_transformer_evaluator.py
├── physics/                   # Physics-informed constraints
│   ├── physics_informed_loss.py
│   └── sensitivity_weighted_loss.py
├── validation/                # OpenMC integration
│   ├── openmc_validator.py
│   ├── sensitivity_analyzer.py
│   └── reactor_benchmark.py
└── utils/                     # Utilities
```

### Data Flow

```
X4Pro SQLite Database (x4sqlite1.db)
        ↓
X4Ingestor (with AME2020)
        ↓
Partitioned Parquet (by Z/A/MT)
        ↓
NucmlDataset (Dual-View)
    ├─→ Graph View (PyG) → GNN-Transformer
    └─→ Tabular View (DataFrame) → XGBoost/Decision Trees
        ↓
Predictions → OpenMC Validation → Sensitivity Analysis
```

---

## Usage Example

```python
from nucml_next.examples import quick_ingest, load_dataset, print_dataset_summary
from nucml_next.baselines import XGBoostEvaluator
from nucml_next.model import GNNTransformerEvaluator

# Quick start: Ingest sample database
df = quick_ingest()  # Uses data/x4sqlite1_sample.db

# Load dataset with filters
dataset = load_dataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    filters={'Z': [92, 17], 'A': [235, 35], 'MT': [18, 103]}
)

# Print summary
print_dataset_summary(dataset)

# Baseline: XGBoost with tier-based features
df = dataset.to_tabular()  # Tier-based features with particle vectors
xgb = XGBoostEvaluator()
xgb.train(df)

# Advanced: GNN-Transformer
dataset_graph = load_dataset(
    data_path='data/exfor_processed.parquet',
    mode='graph'
)
model = GNNTransformerEvaluator()
# ... training loop (see notebooks)
```

---

## Notebooks

Progressive learning pathway:

1. **00_Production_EXFOR_Data_Loading.ipynb**
   Load and verify EXFOR experimental data

2. **01_Data_Fabric_and_Graph.ipynb**
   Build Chart of Nuclides graph representation

3. **02_GNN_Transformer_Training.ipynb**
   Train physics-informed deep learning models

4. **03_OpenMC_Loop_and_Inference.ipynb**
   Reactor validation and sensitivity analysis

---

## Migration Notes (X5json → X4Pro SQLite)

**What Changed:**
- ❌ **Removed:** EXFOR-X5json ingestion (directory recursion, JSON parsing)
- ✅ **Added:** X4Pro SQLite ingestion (single database file, SQL queries)
- ✅ **Added:** `nucml_next.examples` helper module for notebooks
- ✅ **Simplified:** Single clean ingestion path, no legacy abstractions

**Required Actions:**
1. **Obtain X4 database:**
   - Sample: Use `data/x4sqlite1_sample.db` (in repository)
   - Full: Download from https://www-nds.iaea.org/x4/

2. **Update ingestion command:**
   ```bash
   # Old (X5json)
   python scripts/ingest_exfor.py --exfor-root ~/data/EXFOR-X5json/ --output data/exfor.parquet

   # New (X4)
   python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --output data/exfor.parquet
   ```

3. **Update code imports:**
   ```python
   # Old
   from nucml_next.data import ingest_exfor

   # New
   from nucml_next.ingest import ingest_x4
   # Or use convenience helper
   from nucml_next.examples import quick_ingest
   ```

**Output Compatibility:**
- ✅ Parquet schema unchanged (Z, A, MT, Energy, CrossSection, Uncertainty)
- ✅ NucmlDataset API unchanged
- ✅ Downstream models work without modification

**Known Limitations:**
- X4 schema variations may require manual inspection for non-standard databases
- AME2020 enrichment is optional (falls back to SEMF approximation)

---

## Citation

If you use NUCML-Next in your research, please cite:

```bibtex
@software{nucml_next2025,
  author = {NUCML-Next Team},
  title = {NUCML-Next: Physics-Informed Deep Learning for Nuclear Data Evaluation},
  year = {2025},
  version = {1.1.0-alpha},
  url = {https://github.com/WHopkins-git/NUCML-Next}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Documentation

See the [Wiki](https://github.com/WHopkins-git/NUCML-Next/wiki) for:
- Detailed installation instructions
- EXFOR ingestion tutorials
- Model training guides
- OpenMC integration examples
- API reference

---

## Support

- **Issues:** [GitHub Issues](https://github.com/WHopkins-git/NUCML-Next/issues)
- **Discussions:** [GitHub Discussions](https://github.com/WHopkins-git/NUCML-Next/discussions)

---

**Production-ready nuclear data evaluation with real experimental data** ✓
