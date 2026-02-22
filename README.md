# NUCML-Next

Machine learning for nuclear cross-section prediction using EXFOR experimental data and AME2020/NUBASE2020 enrichment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

```bash
git clone https://github.com/WHopkins-git/NUCML-Next.git
cd NUCML-Next
pip install -r requirements.txt

# Download EXFOR database (see Data Files below) into data/
# Then run ingestion:
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

# Open a notebook:
jupyter notebook notebooks/00_Baselines_and_Limitations.ipynb
```

---

## Data Files

Place all files in the `data/` directory.

### EXFOR Database (required)

| File | Source | Notes |
|------|--------|-------|
| `x4sqlite1.db` | [IAEA X4Pro Full-DB](https://nds.iaea.org/cdroms/#x4pro1) | ~2 GB, X4Pro SQLite format |
| `x4sqlite1_sample.db` | Included in repo | Small sample for testing |

### AME2020 / NUBASE2020 (required for Tier B--E features)

Download from https://www-nds.iaea.org/amdc/

| File | Content |
|------|---------|
| `mass_1.mas20.txt` | Mass excess, binding energy, binding per nucleon |
| `rct1.mas20.txt` | S(2n), S(2p), Q(alpha), Q(2beta-), Q(ep), Q(beta-n) |
| `rct2_1.mas20.txt` | S(1n), S(1p), Q(4beta-), Q(d,alpha), Q(p,alpha), Q(n,alpha) |
| `nubase_4.mas20.txt` | Spin, parity, half-life, isomeric states, decay modes |
| `covariance.mas20.txt` | Mass uncertainty correlations (optional, 24 MB) |

**Note:** AME files are NOT used at ingestion time. They are loaded at runtime by `NucmlDataset` during feature generation.

---

## Ingestion Pipeline

```bash
# Basic (no outlier detection, metadata filtering ON by default)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

# Test subset (Uranium + Chlorine only)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

# With per-experiment GP outlier detection (recommended)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method experiment

# Include non-pure data for analysis
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --include-non-pure

# Full pipeline with GPU and checkpointing
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method experiment --svgp-device cuda --svgp-checkpoint-dir data/checkpoints/
```

See `docs/INGESTION_PIPELINE.md` for full details.

### Output Schema

```
Entry, Z, A, N, Projectile, MT, Energy, CrossSection, Uncertainty,
Energy_Uncertainty, log_E, log_sigma, sf5, sf6, sf8, sf9, is_pure, data_type,
gp_mean, gp_std, z_score, experiment_outlier, point_outlier, calibration_metric, experiment_id
```

The last 7 columns (`gp_mean` through `experiment_id`) are only present when `--outlier-method experiment` is used.

---

## Metadata Filtering

By default, ingestion excludes non-pure EXFOR data. This is **type filtering**, not evaluator bias -- it removes entries that are not directly comparable cross-section measurements in barns vs. energy.

| Excluded Category | Reason |
|-------------------|--------|
| Relative measurements (SF8=REL) | Different units, not absolute barns |
| Spectrum-averaged data (SF8=MXW/SPA/FIS/AV) | Different x-axis meaning |
| Ratio data | Dimensionless, not cross sections |
| Non-cross-section quantities | Fission yields, angular distributions |
| Calculated/derived/evaluated (SF9=CALC/DERIV/EVAL/RECOM) | Not experimental measurements |
| Superseded entries (SPSDD flag) | Replaced by newer data |

### CLI Flags

| Flag | Effect |
|------|--------|
| `--include-non-pure` | Keep all data types (marks them via `is_pure` column) |
| `--include-superseded` | Keep superseded entries |

The columns `sf5`, `sf6`, `sf8`, `sf9`, `is_pure`, and `data_type` are always preserved in the output Parquet for downstream analysis.

---

## Feature Tiers

Features are organised into additive tiers selected at runtime via the `tiers` parameter on `DataSelection`:

| Tier | Name | Count | Features |
|------|------|------:|----------|
| A | Core + particle vector | 13 | Z, A, N, Energy, out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met |
| B | Geometric | +2 | R_fm, kR |
| C | Energetics | +7 | Mass_Excess_MeV, Binding_Energy_MeV, Binding_Per_Nucleon_MeV, S_1n, S_2n, S_1p, S_2p |
| D | Topological | +9 | Spin, Parity, Isomer_Level, Half_Life_log10_s, Valence_N, Valence_P, P_Factor, Shell_Closure_N, Shell_Closure_P |
| E | Q-values | +8 | Q_alpha, Q_2beta_minus, Q_ep, Q_beta_n, Q_4beta_minus, Q_d_alpha, Q_p_alpha, Q_n_alpha |

Tier A is always included. Reaction channels (MT codes) are encoded as a 9-component particle-emission vector rather than one-hot indicators.

---

## Outlier Detection

Two methods are available via `--outlier-method`:

| Method | Flag | Approach |
|--------|------|----------|
| Per-experiment GP | `experiment` | Fits independent GPs per experiment, builds consensus, flags discrepant experiments. **Recommended.** |
| Legacy SVGP | `svgp` | Pools all experiments per (Z, A, MT) group into a single Sparse Variational GP. Point-level only. |

### Output Columns (per-experiment GP)

| Column | Type | Description |
|--------|------|-------------|
| `experiment_outlier` | bool | Entire experiment flagged as discrepant |
| `point_outlier` | bool | Individual point is anomalous |
| `z_score` | float | Continuous anomaly score |
| `calibration_metric` | float | Per-experiment Wasserstein distance |
| `experiment_id` | str | EXFOR Entry identifier |

See `docs/INGESTION_PIPELINE.md` for algorithm details and edge-case handling.

### Interactive Threshold Explorer

```python
from nucml_next.visualization.threshold_explorer import ThresholdExplorer

explorer = ThresholdExplorer('data/exfor_processed.parquet')
explorer.show()  # cascading dropdowns + probability surface + z-score bands
```

---

## Package Structure

```
nucml_next/
  ingest/          X4Pro SQLite -> Parquet ingestion
  data/            NucmlDataset, DataSelection, TransformationPipeline,
                   MetadataFilter (metadata_filter.py),
                   ExperimentOutlierDetector (experiment_outlier.py),
                   SVGPOutlierDetector (outlier_detection.py, legacy)
  baselines/       DecisionTreeEvaluator, XGBoostEvaluator
  model/           GNN-Transformer architecture
  physics/         Physics-informed and sensitivity-weighted loss
  validation/      OpenMC reactor benchmarking
  visualization/   CrossSectionFigure, IsotopePlotter, ThresholdExplorer
  utils/           Helpers

scripts/
  ingest_exfor.py  CLI for X4Pro -> Parquet

docs/
  INGESTION_PIPELINE.md  Full ingestion pipeline reference
  ML_TRAINING.md         ML training guide

notebooks/
  00_Baselines_and_Limitations.ipynb
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
