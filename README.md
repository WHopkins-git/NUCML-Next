# NUCML-Next

Machine learning for nuclear cross-section prediction using EXFOR
experimental data and AME2020/NUBASE2020 enrichment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/WHopkins-git/NUCML-Next.git
cd NUCML-Next
pip install -r requirements.txt
```

### 2. Download data files

Place all files in the `data/` directory.

**EXFOR database** (required) -- X4Pro SQLite format:

| File | Source |
|------|--------|
| `x4sqlite1.db` | Download the **Full-DB** from https://nds.iaea.org/cdroms/#x4pro1 |

A small sample database (`data/x4sqlite1_sample.db`) is included in the
repository for testing.

**AME2020 / NUBASE2020** (required for Tier B--E features):

These files are not used during ingestion. They are read at runtime when
the notebook loads data and generates features via `NucmlDataset`.

Download the `*.mas20.txt` files from https://www-nds.iaea.org/amdc/

| File | Description |
|------|-------------|
| `mass_1.mas20.txt` | Mass excess, binding energy |
| `rct1.mas20.txt` | S(2n), S(2p), Q(α), Q(2β⁻) |
| `rct2_1.mas20.txt` | S(1n), S(1p), reaction Q-values |
| `nubase_4.mas20.txt` | Spin, parity, half-life, isomeric states |

### 3. Run ingestion

```bash
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1.db \
    --output data/exfor_processed.parquet
```

This reads the X4Pro SQLite database and writes a Parquet dataset with
schema `[Entry, Z, A, MT, Energy, CrossSection, Uncertainty]`.
AME2020 enrichment is applied later during feature generation, not at
ingestion time.

### 4. Run notebooks

```bash
jupyter notebook notebooks/00_Baselines_and_Limitations.ipynb
```

---

## Feature tiers

Features are organised into additive tiers selected at runtime via the
`tiers` parameter on `DataSelection`:

| Tier | Name | Count | Features |
|------|------|------:|----------|
| A | Core + particle vector | 13 | Z, A, N, Energy, out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met |
| B | Geometric | +2 | R_fm, kR |
| C | Energetics | +7 | Mass_Excess_MeV, Binding_Energy_MeV, Binding_Per_Nucleon_MeV, S_1n, S_2n, S_1p, S_2p |
| D | Topological | +9 | Spin, Parity, Isomer_Level, Half_Life_log10_s, Valence_N, Valence_P, P_Factor, Shell_Closure_N, Shell_Closure_P |
| E | Q-values | +8 | Q_alpha, Q_2beta_minus, Q_ep, Q_beta_n, Q_4beta_minus, Q_d_alpha, Q_p_alpha, Q_n_alpha |

Tier A is always included. Reaction channels (MT codes) are encoded as a
9-component particle-emission vector rather than one-hot indicators.

---

## Package structure

```
nucml_next/
  ingest/          X4Pro SQLite -> Parquet ingestion
  data/            NucmlDataset, DataSelection, feature generation, transformations
  baselines/       DecisionTreeEvaluator, XGBoostEvaluator
  model/           GNN-Transformer architecture
  physics/         Physics-informed and sensitivity-weighted loss
  validation/      OpenMC reactor benchmarking
  visualization/   CrossSectionFigure (EXFOR overlay plots)
  utils/           Helpers

scripts/
  ingest_exfor.py  CLI for X4Pro -> Parquet
  clean_ame_files.py  Replace '#' estimated-value markers in AME files

notebooks/
  00_Baselines_and_Limitations.ipynb   Decision Tree and XGBoost baselines
  00_Production_EXFOR_Data_Loading.ipynb
  01_Data_Fabric_and_Graph.ipynb
  01_Database_Statistical_Audit.ipynb
  02_GNN_Transformer_Training.ipynb
  03_OpenMC_Loop_and_Inference.ipynb
```

---

## Citations

- **AME2020:** W.J. Huang et al., Chinese Phys. C **45**, 030002 (2021)
- **NUBASE2020:** F.G. Kondev et al., Chinese Phys. C **45**, 030001 (2021)

## License

MIT -- see LICENSE.
