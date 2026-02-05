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

**Or with conda:**

```bash
git clone https://github.com/WHopkins-git/NUCML-Next.git
cd NUCML-Next
conda create -n nucml python=3.10
conda activate nucml
conda install -c conda-forge numpy pandas scipy matplotlib seaborn scikit-learn xgboost ipywidgets jupyter pyarrow
conda install -c pytorch -c gpytorch gpytorch pytorch
pip install -r requirements.txt  # for any remaining packages
```

Key dependencies:

- `gpytorch` -- required for GP outlier detection (`--outlier-method`)
- `ipywidgets` -- required for interactive threshold explorer in notebooks
- `xgboost`, `scikit-learn` -- baseline models

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
# Basic ingestion (no outlier detection)
python scripts/ingest_exfor.py \
    --x4-db data/x4sqlite1.db \
    --output data/exfor_processed.parquet

# Test subset: Uranium + Chlorine only (~300K points, minutes instead of hours)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

# Custom element subset (Gold, Uranium, Iron)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --z-filter 79,92,26

# With per-experiment GP outlier detection (RECOMMENDED)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method experiment

# With legacy SVGP outlier detection
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp

# With GPU acceleration
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment --svgp-device cuda

# With checkpointing (resume interrupted runs)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment --svgp-device cuda \
    --svgp-checkpoint-dir data/checkpoints/

# Full pipeline: test subset + per-experiment outlier detection
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --test-subset --outlier-method experiment --z-threshold 3.0
```

The ingestion pipeline:

1. **Extract** -- reads cross-section measurements from the X4Pro SQLite
   database (auto-detects schema variants).
2. **Normalise** -- maps to standard schema, applies data-quality filters
   (removes A=0, non-positive values, unrealistic ranges).
3. **Outlier scoring** (if `--outlier-method`) -- fits Gaussian Processes
   to detect anomalous measurements and discrepant experiments.
4. **Write Parquet** -- saves lean dataset for downstream use.

Output schema:

```
[Entry, Z, A, N, Projectile, MT, Energy, CrossSection, Uncertainty,
 Energy_Uncertainty, log_E, log_sigma, gp_mean, gp_std, z_score,
 experiment_outlier, point_outlier, calibration_metric, experiment_id]
                ^^^^^ added by --outlier-method experiment ^^^^^
```

AME2020 enrichment is applied later during feature generation, not at
ingestion time.

<details>
<summary>Expected console output (with --outlier-method experiment)</summary>

```
======================================================================
NUCML-Next: X4Pro EXFOR Data Ingestion (Lean Mode)
======================================================================
X4 Database:  data/x4sqlite1.db
Output:       data/exfor_processed.parquet
Mode:         Lean extraction (EXFOR data only)
Outlier:      Per-experiment GP (recommended) - z_threshold=3.0
======================================================================

... (Experiment scoring progress bar) ...

Ingestion complete!
Processed 13,419,082 data points
Saved to: data/exfor_processed.parquet
Per-experiment outlier detection:
    Point outliers: 180,000 (1.34%)
    Experiment outliers: 450 experiments flagged
```
</details>

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

## Transformation pipeline

The `TransformationPipeline` applies transforms in a fixed order before
training. The order matters: log-transforms run **first**, so that the
scaler sees compressed log-space values rather than raw multi-order-of-
magnitude physical values.

**Order of operations (forward transform):**

1. Log-transform cross-section: `sigma' = log10(sigma + epsilon)`
   - `target_epsilon = 1e-10` prevents `log(0)` for very small cross-sections
2. Log-transform energy: `E' = log10(E)`
   - No epsilon for energy (measurements are always > 0)
3. Scale all features: `X' = (X - min) / (max - min)`
   - Applied to already-log-transformed values

**Defaults:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_target` | `True` | Log-transform cross-sections |
| `target_epsilon` | `1e-10` | Epsilon for `log(sigma + eps)` |
| `log_energy` | `True` | Log-transform energies |
| `scaler_type` | `'minmax'` | Feature scaling method (`minmax`, `standard`, `robust`, `none`) |
| `scale_features` | `'all'` | Scale every numeric column |

For tree-based models (Decision Trees, XGBoost), feature scaling is not
mathematically necessary because trees only use value ordering. MinMax
scaling is cheap and prepares the pipeline for neural networks where
scaling is required.

```python
from nucml_next.data.selection import TransformationConfig

config = TransformationConfig(
    log_target=True,
    target_epsilon=1e-10,
    log_energy=True,
    scaler_type='minmax',
    scale_features='all',
)
```

---

## Outlier detection (optional)

NUCML-Next includes GP-based outlier detection that flags suspicious EXFOR
measurements and identifies discrepant experiments. Two methods are available:

### Per-experiment GP (recommended)

The `--outlier-method experiment` approach fits independent Exact GPs to each
EXFOR experiment (Entry) within a (Z, A, MT) group, builds consensus from
multiple experiments, and flags entire experiments that deviate systematically.

**Key advantages:**
- Uses heteroscedastic noise from measurement uncertainties
- Flags discrepant *experiments*, not just individual points
- Better handles resonance structure (no over-smoothing)
- Calibrates lengthscale via Wasserstein distance

**Output columns:**
- `experiment_outlier` -- bool: entire experiment flagged as discrepant
- `point_outlier` -- bool: individual point is anomalous
- `z_score` -- float: continuous anomaly score
- `calibration_metric` -- float: per-experiment Wasserstein distance
- `experiment_id` -- str: EXFOR Entry identifier

### Legacy SVGP

The `--outlier-method svgp` approach pools all experiments per (Z, A, MT)
group and fits a single Sparse Variational GP. This produces point-level
z-scores but cannot identify systematically discrepant experiments.

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--outlier-method` | none | `experiment` (recommended) or `svgp` (legacy) |
| `--z-threshold` | 3.0 | Z-score threshold for point outliers |
| `--svgp-device` | `cpu` | PyTorch device: `cpu` or `cuda` |
| `--svgp-checkpoint-dir` | none | Directory for checkpoint files |
| `--svgp-likelihood` | `student_t` | Likelihood type for SVGP method |

**Performance:** The full EXFOR database (~13.4M points, ~30K experiments)
takes approximately 4 hours single-threaded or under 1 hour with 8 workers.
Checkpointing allows interrupted runs to resume.

### Run ingestion with outlier scoring

```bash
# Per-experiment GP (recommended)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method experiment

# Per-experiment GP with GPU and checkpointing
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment --svgp-device cuda \
    --svgp-checkpoint-dir data/checkpoints/

# Legacy SVGP
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp
```

### Filter outliers at load time

```python
from nucml_next.data import DataSelection, NucmlDataset

selection = DataSelection(
    z_threshold=3.0,          # Exclude points with z_score > 3
    include_outliers=False,   # Remove (not just flag) outliers
)
dataset = NucmlDataset('data/exfor_processed.parquet', selection=selection)
# -> [OK] Outlier filter: Removed 180,000 points with z_score > 3.0

# View outlier statistics at different thresholds
dataset.outlier_summary()
#    threshold  outliers     pct  retained
# 0        2.0    380000    2.83  13039082
# 1        3.0    180000    1.34  13239082
# 2        4.0     45000    0.34  13374082
# 3        5.0     12000    0.09  13407082
```

### Programmatic API

```python
from nucml_next.data import ExperimentOutlierDetector, ExperimentOutlierConfig

# Per-experiment GP (recommended)
config = ExperimentOutlierConfig(point_z_threshold=3.0)
detector = ExperimentOutlierDetector(config)
df_scored = detector.score_dataframe(df)
# df_scored has: experiment_outlier, point_outlier, z_score, experiment_id

# Filter to non-discrepant experiments
df_clean = df_scored[~df_scored['experiment_outlier']]
```

### Interactive threshold explorer

In Jupyter notebooks, use the interactive widget to browse any (Z, A, MT)
group, visualise the GP predictive distribution, and adjust the threshold:

```python
from nucml_next.visualization.threshold_explorer import ThresholdExplorer

explorer = ThresholdExplorer('data/exfor_processed.parquet')
explorer.show()  # cascading dropdowns + probability surface + z-score bands
```

### Edge cases

- Groups with **< 10 points**: MAD (Median Absolute Deviation) fallback
  with consistency constant 1.4826
- Experiments with **< 5 points**: evaluated against consensus from larger
  experiments, or MAD if no consensus available
- Groups with **1 experiment**: GP fit possible, but `experiment_outlier`
  cannot be assessed (no comparison)
- **No `--outlier-method`**: outlier columns are absent; `z_threshold`
  emits a warning and has no effect
- **GP numerical failure**: falls back to MAD automatically, logged at
  WARNING level

---

## Package structure

```
nucml_next/
  ingest/          X4Pro SQLite -> Parquet ingestion
  data/            NucmlDataset, DataSelection, TransformationPipeline,
                   ExperimentOutlierDetector (experiment_outlier.py),
                   SVGPOutlierDetector (outlier_detection.py, legacy)
  baselines/       DecisionTreeEvaluator, XGBoostEvaluator
  model/           GNN-Transformer architecture
  physics/         Physics-informed and sensitivity-weighted loss
  validation/      OpenMC reactor benchmarking
  visualization/   CrossSectionFigure, IsotopePlotter (one-line plotting),
                   ThresholdExplorer (interactive notebook widget)
  utils/           Helpers

scripts/
  ingest_exfor.py  CLI for X4Pro -> Parquet (with --outlier-method support)
  validate_experiment_outlier.py  Validation on benchmark reactions
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
