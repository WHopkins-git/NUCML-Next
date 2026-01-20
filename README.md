# NUCML-Next: Next-Generation Nuclear Data Evaluation

**Physics-Informed Deep Learning for Nuclear Cross-Section Prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

NUCML-Next is an **educational platform** for teaching physics-informed machine learning in nuclear engineering. It demonstrates the evolution from classical ML (XGBoost, Decision Trees) to physics-informed deep learning (GNNs + Transformers) for nuclear data evaluation.

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

```bash
# Install
git clone https://github.com/WHopkins-git/NUCML-Next.git
cd NUCML-Next
pip install -r requirements.txt

# Run educational notebooks
jupyter notebook notebooks/00_Baselines_and_Limitations.ipynb
```

---

## Features

✓ **Dual-view data architecture** (Graph + Tabular)
✓ **Baseline models** (Decision Trees, XGBoost)
✓ **GNN-Transformer** architecture
✓ **Physics-informed loss** functions
✓ **OpenMC integration** for validation
✓ **Sensitivity analysis** for reactor safety

---

## Documentation

See the [Wiki](https://github.com/WHopkins-git/NUCML-Next/wiki) for full documentation and tutorials.

---

## License

MIT License - See LICENSE file for details.