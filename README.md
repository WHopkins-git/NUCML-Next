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

**Step 1: Download EXFOR-X5json database**
- Visit: https://www-nds.iaea.org/exfor/
- Download: EXFOR-X5json bulk ZIP (~500 MB compressed)
- Unzip to: `~/data/EXFOR-X5json/`

**Step 2: Ingest EXFOR to Parquet**
```bash
python scripts/ingest_exfor.py \
    --exfor-root ~/data/EXFOR-X5json/ \
    --output data/exfor_processed.parquet
```

**Step 3: Load in notebooks**
```python
from nucml_next.data import NucmlDataset

# Load EXFOR data
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='graph',
    filters={'Z': [92], 'MT': [18, 102]}  # Optional: U reactions only
)
```

**Step 4: Run training notebooks**
```bash
jupyter notebook notebooks/00_Production_EXFOR_Data_Loading.ipynb
```

---

## Data Sources

### EXFOR Database (Required)

NUCML-Next uses real experimental nuclear cross-section data from the IAEA EXFOR database.

**Ingestion Process:**

```python
from nucml_next.data import ingest_exfor

# Ingest EXFOR database to Parquet
df = ingest_exfor(
    exfor_root='~/data/EXFOR-X5json/',
    output_path='data/exfor_processed.parquet',
    ame2020_path='data/ame2020.txt',  # Optional: for enhanced isotope features
    max_files=None  # Process all files
)
```

**Output:**
- Partitioned Parquet dataset by Z/A/MT
- AME2020-enriched isotope features
- Preserves experimental uncertainties
- Flags natural targets

### AME2020 Integration (Optional)

For enhanced isotope features with real mass excess and binding energy data:

```bash
# Download AME2020
wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt -O data/ame2020.txt

# Use during ingestion
python scripts/ingest_exfor.py \
    --exfor-root ~/data/EXFOR-X5json/ \
    --output data/exfor_processed.parquet \
    --ame2020 data/ame2020.txt
```

---

## Features

### v1.1.0-alpha (Production-Ready)
✓ **EXFOR-X5json bulk ingestor** with AME2020 enrichment
✓ **Partitioned Parquet** data fabric for large-scale datasets
✓ **Real experimental data** from IAEA EXFOR database
✓ **No simulation or synthetic data** - production-grade only

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
├── data/                      # Data ingestion and handling
│   ├── exfor_ingestor.py      # EXFOR-X5json bulk ingestor
│   ├── dataset.py             # NucmlDataset with dual-view
│   ├── graph_builder.py       # Chart of Nuclides graph
│   └── tabular_projector.py   # Graph → Tabular projection
├── baselines/                 # Classical ML baselines
│   ├── decision_tree_evaluator.py
│   └── xgboost_evaluator.py
├── model/                     # Deep learning models
│   ├── nuclide_gnn.py         # Graph Neural Network
│   ├── energy_transformer.py  # Transformer for σ(E)
│   └── gnn_transformer_evaluator.py
├── physics/                   # Physics-informed constraints
│   ├── physics_informed_loss.py
│   ├── unitarity_constraint.py
│   ├── threshold_constraint.py
│   └── sensitivity_weighted_loss.py
├── validation/                # OpenMC integration
│   ├── openmc_validator.py
│   ├── sensitivity_analyzer.py
│   └── reactor_benchmark.py
└── utils/                     # Utilities
```

### Data Flow

```
EXFOR-X5json Database
        ↓
EXFORIngestor (with AME2020)
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
from nucml_next.data import NucmlDataset
from nucml_next.baselines import XGBoostEvaluator
from nucml_next.model import GNNTransformerEvaluator

# Load EXFOR data
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    filters={'Z': [92], 'A': [235], 'MT': [18, 102]}
)

# Baseline: XGBoost with physics features
df = dataset.to_tabular(mode='physics')
xgb = XGBoostEvaluator()
xgb.train(df)

# Advanced: GNN-Transformer
dataset_graph = NucmlDataset(
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
