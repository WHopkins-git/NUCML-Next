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

# With smooth mean + local MAD outlier detection (recommended)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method local_mad

# Custom experiment discrepancy thresholds
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method local_mad \
    --z-threshold 5 --exp-z-threshold 3 --exp-fraction-threshold 0.25

# With diagnostic metadata for interactive inspection
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset --diagnostics

# Include non-pure data for analysis
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --include-non-pure

# Legacy SVGP outlier detection
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp
```

See `docs/INGESTION_PIPELINE.md` for algorithm details and edge-case handling.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--x4-db` | required | Path to X4Pro SQLite database |
| `--output` | `data/exfor_processed.parquet` | Output Parquet path |
| `--test-subset` | OFF | Use test subset: U (Z=92) + Cl (Z=17) |
| `--z-filter` | None | Comma-separated Z values (e.g., `79,92,26`) |
| `--include-non-pure` | OFF | Include non-pure data (relative, ratio, averaged, etc.) |
| `--include-superseded` | OFF | Include superseded entries |
| `--diagnostics` | OFF | Add Author, Year, ReactionType, FullCode, NDataPoints columns for interactive inspection |
| `--outlier-method` | None | `local_mad` (recommended) or `svgp` (legacy) |
| `--z-threshold` | 3.0 | Z-score threshold for point outliers |
| `--exp-z-threshold` | 3.0 | Z-score threshold for counting bad points in experiment discrepancy (local_mad only) |
| `--exp-fraction-threshold` | 0.30 | Fraction of bad points to flag experiment as discrepant (local_mad only) |
| `--svgp-device` | `cpu` | `cpu` or `cuda` (SVGP only) |
| `--svgp-likelihood` | `student_t` | Likelihood: `student_t`, `heteroscedastic`, `gaussian` (SVGP only) |
| `--num-threads` | 50% of cores | CPU threads for NumPy/SciPy linear algebra |
| `--ame2020-dir` | None | DEPRECATED -- ignored (AME loaded at feature-generation time) |
| `--run-svgp` | OFF | DEPRECATED -- use `--outlier-method svgp` instead |
| `--no-svgp` | OFF | DEPRECATED -- outlier detection is off by default |

### Output Schema

```
Entry, Z, A, N, Projectile, MT, Energy, CrossSection, Uncertainty,
Energy_Uncertainty, log_E, log_sigma, sf5, sf6, sf8, sf9, is_pure, data_type,
gp_mean, gp_std, z_score, experiment_outlier, point_outlier, calibration_metric, outlier_probability, experiment_id,
Year, Author, ReactionType, FullCode, NDataPoints
```

The scoring columns (`gp_mean` through `experiment_id`) are present when `--outlier-method local_mad` is used. `gp_mean` contains the smooth mean, `gp_std` contains the effective sigma (local MAD combined with measurement uncertainty in quadrature). The diagnostic columns (`Year` through `NDataPoints`) are only present when `--diagnostics` is used.

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

Use `--include-non-pure` and `--include-superseded` to override filtering (see CLI Flags table above). The columns `sf5`, `sf6`, `sf8`, `sf9`, `is_pure`, and `data_type` are always preserved in the output Parquet for downstream analysis.

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
| Smooth mean + local MAD | `local_mad` | Fits pooled smooth mean, computes energy-local MAD, incorporates EXFOR measurement uncertainties in quadrature. Parallel across CPU cores. **Recommended.** |
| Legacy SVGP | `svgp` | Pools all experiments per (Z, A, MT) group into a single Sparse Variational GP. Point-level only. |

### Output Columns (local_mad)

| Column | Type | Description |
|--------|------|-------------|
| `experiment_outlier` | bool | Experiment has > 30% of points with z > 3 (configurable) |
| `point_outlier` | bool | Individual point exceeds z-threshold |
| `z_score` | float | \|residual from smooth mean\| / effective_sigma |
| `gp_mean` | float | Smooth mean value (consensus trend) |
| `gp_std` | float | Effective sigma = sqrt(local_MAD² + σ_measurement²) |
| `experiment_id` | str | EXFOR Entry identifier |
| `calibration_metric` | float | NaN (not used by local_mad) |
| `outlier_probability` | float | NaN (not used by local_mad) |

### Legacy GP options

- **Smooth mean** (`SmoothMeanConfig`): Data-driven consensus trend from pooled EXFOR data. Opt-in via `smooth_mean_type='spline'`. Removes gross energy dependence before GP fitting.
- **Gibbs kernel** (`KernelConfig`): Physics-informed nonstationary kernel using RIPL-3 level density. Opt-in via `kernel_type='gibbs'`. Adapts lengthscale to nuclear resonance structure.
- **Robust likelihood** (`LikelihoodConfig`): Contaminated normal mixture for principled outlier identification. Opt-in via `likelihood_type='contaminated'`. Assigns continuous outlier probability per point.
- **Hierarchical refitting** (`hierarchical_refitting=True`): Two-pass fitting that constrains per-experiment hyperparameters to a group-informed feasible region. Produces more consistent GP fits across experiments within a reaction group.

See `docs/INGESTION_PIPELINE.md` for algorithm details, edge-case handling, and scientific motivation.

### Interactive Threshold Explorer

```python
from nucml_next.visualization.threshold_explorer import ThresholdExplorer

explorer = ThresholdExplorer('data/exfor_processed.parquet')
explorer.show()  # cascading dropdowns + probability surface + z-score bands
```

### Diagnostics Interactive Inspector

For investigating suspicious data clusters, use the Plotly-based diagnostic notebook. Hover over individual points to see Entry ID, author, year, full REACTION string, and more.

```bash
# Ingest with diagnostic metadata
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset --diagnostics

# Open the notebook
jupyter notebook notebooks/Diagnostics_Interactive_Inspector.ipynb
```

The notebook provides cascading Z/A/MT dropdowns with three coloring modes (Uniform, Color by Entry, Color by data_type) and WebGL-accelerated rendering for large datasets.

---

## Package Structure

```
nucml_next/
  ingest/          X4Pro SQLite -> Parquet ingestion
  data/            NucmlDataset, DataSelection, TransformationPipeline,
                   MetadataFilter (metadata_filter.py),
                   ExperimentOutlierDetector (experiment_outlier.py),
                   SmoothMeanConfig (smooth_mean.py),
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
  Diagnostics_Interactive_Inspector.ipynb
```

---

## Citations

- **AME2020:** W.J. Huang et al., Chinese Phys. C **45**, 030002 (2021)
- **NUBASE2020:** F.G. Kondev et al., Chinese Phys. C **45**, 030001 (2021)

## License

MIT -- see LICENSE.
