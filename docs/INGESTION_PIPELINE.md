# Ingestion Pipeline Reference

Technical reference for the NUCML-Next data ingestion pipeline:
X4Pro SQLite -> metadata filter -> normalize -> optional outlier detection -> Parquet.

---

## 1. Overview

The ingestion pipeline converts IAEA EXFOR cross-section measurements from an X4Pro SQLite
database into a clean, ML-ready Parquet file. The pipeline stages are:

| Stage | Description | Key Module |
|-------|-------------|------------|
| Extraction | Read X4Pro SQLite (JSON c5data) | `nucml_next.ingest.x4.X4Ingestor` |
| Metadata Filtering | Exclude non-pure and superseded data | `nucml_next.data.metadata_filter.MetadataFilter` |
| Normalization | Column mapping, physical range filters | `X4Ingestor._normalize()` |
| Outlier Detection | Optional GP-based z-scoring | `experiment_outlier.py` or `outlier_detection.py` |
| Output | Write Parquet with all columns | PyArrow `write_to_dataset` |

Entry point: `python scripts/ingest_exfor.py --x4-db <path>`

---

## 2. Data Files

### EXFOR Database (Required)

| Item | Detail |
|------|--------|
| File | `x4sqlite1.db` |
| Source | https://nds.iaea.org/cdroms/#x4pro1 |
| Size | ~2-4 GB (full), ~40 MB (sample) |
| Sample | `data/x4sqlite1_sample.db` (included in repo) |
| Format | SQLite with X4Pro schema |

### AME2020/NUBASE2020 (Recommended)

These files are NOT used during ingestion. They are loaded by `NucmlDataset` during
feature generation, joined on `(Z, A)` at runtime.

| File | Content | Download |
|------|---------|----------|
| `mass_1.mas20.txt` | Mass excess, binding energy | https://www-nds.iaea.org/amdc/ame2020/ |
| `rct1.mas20.txt` | S_2n, S_2p, Q_alpha, Q_2beta_minus | https://www-nds.iaea.org/amdc/ame2020/ |
| `rct2_1.mas20.txt` | S_1n, S_1p, Q_4beta_minus, Q_d_alpha | https://www-nds.iaea.org/amdc/ame2020/ |
| `nubase_4.mas20.txt` | Spin, parity, isomer level, half-life | https://www-nds.iaea.org/amdc/ame2020/ |
| `covariance.mas20.txt` | Uncertainty correlations (optional) | https://www-nds.iaea.org/amdc/ame2020/ |

Place all files in the `data/` directory.

---

## 3. Extraction

### X4Pro Schema

The primary extraction path uses two tables joined on `DatasetID`:

| Table | Role |
|-------|------|
| `x4pro_ds` | Dataset metadata: `DatasetID`, `zTarg1` (Z), `Targ1` (target string), `Proj` (projectile), `MT` |
| `x4pro_x5z` | JSON `jx5z` column containing c5data with energy/cross-section arrays |

Alternative schemas (`x4pro_c5dat`, legacy `data_points`, joined `reactions` tables) are
auto-detected and handled by the ingestor.

### Unit Handling

The C5 computational format is pre-normalized by X4Pro:

| Quantity | Unit | JSON Path |
|----------|------|-----------|
| Energy | eV (electronvolts) | `c5data.x1.x1` |
| Energy uncertainty | eV | `c5data.x1.dx1` |
| Cross section | barns (b) | `c5data.y.y` |
| Cross-section uncertainty | barns (b) | `c5data.y.dy` |

No unit conversion is performed during ingestion.

### Columns Extracted

```
DatasetID, Z, A, Projectile, MT, En, dEn, Data, dData
```

---

## 4. Metadata Filtering

### Background

EXFOR REACTION strings contain subfields SF1-SF9. Non-pure entries have fundamentally
different units or meaning than point-wise absolute cross sections:

| Category | Example Codes | Why Excluded |
|----------|--------------|--------------|
| Relative data | SF8=REL | Arbitrary units, not absolute barns |
| Spectrum-averaged | SF8=MXW, SPA, FIS, AV | Integral averages, not pointwise |
| Ratio data | compound `reacCombi` | Dimensionless, not in barns |
| Non-XS quantities | SF6 != SIG | Fission yields, angular distributions, etc. |
| Calculated/derived | SF9=CALC, DERIV, EVAL, RECOM | Not experimental measurements |
| Superseded entries | SUBENT.SPSDD = '1' or 'D' | Replaced or deprecated |

### Implementation

- `MetadataFilter` class discovers the X4Pro schema at runtime via `PRAGMA table_info`.
- Joins `REACODE` table (via `ReacodeID` = `DatasetID`) to extract `fullCode`.
- Parses SF5, SF6, SF8, SF9 from `fullCode` using regex.
- Handles compound/ratio reactions via `reacCombi` column: skips `fullCode` parsing for
  these entries and marks them as ratio.
- Vectorized classification using `pandas.isin()` and `np.select()` for 13M+ rows (no `.apply()`).

### Filtering Rules

| Subfield | Strategy | Codes |
|----------|----------|-------|
| SF6 | Whitelist | `SIG`, `WID` |
| SF8 Tier 1 | Exclude | `REL`, `RAT`, `RTE`, `FCT` |
| SF8 Tier 2 | Exclude | `MXW`, `SPA`, `FIS`, `AV`, `BRA`, `BRS` |
| SF8 Tier 3 | Exclude | `ETA`, `ALF`, `RES`, `G` |
| SF5 | Exclude | `PRE`, `SEC`, `TER`, `QTR`, `DL`, `PAR` |
| SF9 | Exclude | `CALC`, `DERIV`, `EVAL`, `RECOM` |
| SPSDD | Exclude | `'1'` (superseded), `'D'` (deprecated) |

Default filtering applies SF8 Tiers 1+2 plus all SF5, SF6, SF9, and SPSDD rules.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| (default) | ON | Exclude non-pure and superseded data |
| `--include-non-pure` | OFF | Keep all data types including non-pure |
| `--include-superseded` | OFF | Keep superseded entries |

---

## 5. Normalization

### Column Mapping

| Source (X4Pro) | Target (Parquet) |
|----------------|------------------|
| `DatasetID` | `Entry` |
| `zTarg1` / `Z` | `Z` |
| `A` | `A` |
| `Proj` / `Projectile` | `Projectile` |
| `MT` | `MT` |
| `En` | `Energy` |
| `dEn` | `Energy_Uncertainty` |
| `Data` | `CrossSection` |
| `dData` | `Uncertainty` |
| (computed) | `N` (= A - Z) |

### Physical Range Filters

Applied sequentially after column mapping:

| Filter | Condition | Reason |
|--------|-----------|--------|
| Natural targets | `A = 0` | Cannot enrich with AME2020; invalid N |
| Non-positive energy | `Energy <= 0` | Unphysical |
| Non-positive XS | `CrossSection <= 0` | Unphysical (log-space operations require > 0) |
| Infinite values | `isinf(Energy)` or `isinf(CrossSection)` | Corrupt data |
| Energy range | `1e-5 eV` to `1e9 eV` | Thermal to 1 GeV |
| Cross-section range | `1e-10 b` to `1e6 b` | Nuclear-scale bounds |

### Unit Verification

Sanity checks are logged as warnings (not enforced as filters):

- Median energy > 1e12 suggests keV/MeV instead of eV.
- Median cross section < 1e-15 suggests mb/ub instead of barns.

---

## 6. Outlier Detection

### Per-Experiment GP (Recommended)

Selected with `--outlier-method experiment`.

| Aspect | Detail |
|--------|--------|
| Approach | Fits independent Exact GPs to each experiment (Entry) within a (Z, A, MT) group |
| Kernel | Matern 5/2 with heteroscedastic noise from measurement uncertainties |
| Calibration | Wasserstein distance between LOO |z|-scores and half-normal distribution |
| Consensus | Weighted median of per-experiment GP posteriors at common energy grid |
| Experiment flagging | Flagged if > 20% of grid points have |z| > 2 vs consensus |
| Implementation | `nucml_next.data.experiment_outlier.ExperimentOutlierDetector` |

**Output columns:**

| Column | Type | Description |
|--------|------|-------------|
| `experiment_outlier` | bool | Entire experiment deviates from consensus |
| `point_outlier` | bool | Individual point exceeds z-threshold |
| `z_score` | float | Standardized residual from GP posterior |
| `calibration_metric` | float | Wasserstein distance (lower = better calibrated) |
| `experiment_id` | str | EXFOR Entry identifier |

**Edge cases:**

| Condition | Handling |
|-----------|----------|
| Group < 10 points | MAD fallback (Median Absolute Deviation) |
| Experiment < 5 points | Evaluated against consensus if available; else MAD |
| Single-experiment group | Cannot assess `experiment_outlier`; point-level only |
| Single-point group | z_score = 0, gp_std = 1.0 |
| Cholesky failure | Falls back to MAD; logged at WARNING |

**Processing logic per (Z, A, MT) group:**

```
IF n_total_points < 10:
    -> MAD fallback

ELSE IF n_experiments == 1:
    -> Fit ExactGP if n >= 5, else MAD
    -> Cannot flag experiment_outlier (no comparison)

ELSE (n_experiments >= 2):
    large_exps = [e for e in experiments if len(e) >= 5]
    small_exps = [e for e in experiments if len(e) < 5]

    IF len(large_exps) >= 2:
        -> Fit ExactGP to each large experiment
        -> Build consensus from posteriors
        -> Flag discrepant experiments
        -> Evaluate small experiments against consensus

    ELIF len(large_exps) == 1:
        -> Use single GP as reference for small experiments

    ELSE:
        -> All experiments small - MAD within each
```

### Legacy SVGP

Selected with `--outlier-method svgp`.

| Aspect | Detail |
|--------|--------|
| Approach | Pools all experiments per (Z, A, MT) group into single Sparse Variational GP |
| Inducing points | 50, evenly spaced in log-energy |
| Likelihood | Student-t (default), heteroscedastic, or Gaussian |
| Training | Adam optimizer (lr=0.05), early stopping (patience=10, tol=1e-3), max 300 epochs |
| Output | Point-level z-scores only; no experiment-level flagging |
| Implementation | `nucml_next.data.outlier_detection.SVGPOutlierDetector` |

**Likelihood comparison:**

| Likelihood | Max/Min Uncertainty Ratio | Notes |
|------------|--------------------------|-------|
| Gaussian (legacy) | 1.47x | Constant-width bands |
| Student-t (default) | 2.65x | Learns nu ~ 2-4; robust to outliers |
| Heteroscedastic (strict) | 17.55x | Uses measurement uncertainties; best calibration |
| Heteroscedastic (relaxed) | 1.28x | `learn_additional_noise=True`; additional noise dominates |

---

## 7. Output Schema

Full output with all optional columns enabled:

```
Entry, Z, A, N, Projectile, MT,
Energy, Energy_Uncertainty, CrossSection, Uncertainty,
sf5, sf6, sf8, sf9, is_pure, data_type,
log_E, log_sigma, gp_mean, gp_std, z_score,
experiment_outlier, point_outlier, calibration_metric, experiment_id
```

| Column Group | Present When |
|--------------|-------------|
| Core (Entry through Uncertainty) | Always |
| Metadata (sf5 through data_type) | Metadata filtering is ON (default) |
| GP columns (log_E through z_score) | `--outlier-method` is set |
| Experiment columns (experiment_outlier through experiment_id) | `--outlier-method experiment` |

---

## 8. CLI Reference

```
python scripts/ingest_exfor.py [FLAGS]
```

### Full Flag Table

| Flag | Default | Description |
|------|---------|-------------|
| `--x4-db` | required | Path to X4Pro SQLite database |
| `--output` | `data/exfor_processed.parquet` | Output Parquet path |
| `--test-subset` | OFF | Use test subset: U (Z=92) + Cl (Z=17) |
| `--z-filter` | None | Comma-separated Z values (e.g., `79,92,26`) |
| `--outlier-method` | None | `experiment` (recommended) or `svgp` (legacy) |
| `--z-threshold` | 3.0 | Z-score threshold for point outliers |
| `--svgp-device` | `cpu` | `cpu` or `cuda` |
| `--max-gpu-points` | 40000 | Max points per experiment on GPU; larger auto-route to CPU |
| `--max-subsample-points` | 15000 | Subsample large experiments for GP fitting |
| `--svgp-checkpoint-dir` | None | Enable checkpointing for resume on interruption |
| `--svgp-likelihood` | `student_t` | Likelihood: `student_t`, `heteroscedastic`, `gaussian` |
| `--include-non-pure` | OFF | Include non-pure data (relative, ratio, averaged, etc.) |
| `--include-superseded` | OFF | Include superseded entries |
| `--num-threads` | 50% of cores | CPU threads for NumPy/PyTorch linear algebra |

### Example Commands

```bash
# Standard ingestion (full database, ~13M points)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

# Test subset (~300K points, minutes not hours)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

# Per-experiment GP outlier detection (recommended)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment

# GPU-accelerated with checkpointing
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment --svgp-device cuda \
    --svgp-checkpoint-dir data/checkpoints/

# Custom element subset (Gold, Uranium, Iron)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --z-filter 79,92,26

# Include all data (no metadata filtering)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --include-non-pure --include-superseded
```

---

## 9. Resource Management

| Resource | Policy |
|----------|--------|
| CPU threads | 50% of available cores by default (shared machine etiquette); override with `--num-threads` |
| GPU | Unrestricted; experiments > `--max-gpu-points` auto-route to CPU |
| Memory per experiment | n^2 * 8 bytes for Exact GP covariance matrix (40k pts = 12.8 GB) |
| CUDA OOM | Automatic retry on CPU |
| Checkpointing | Reduces peak memory from ~100 GB to ~20-30 GB; saves `.pt` files every 1000 groups |
| Resume | Pipeline detects existing checkpoints and resumes from next unprocessed group |

Thread environment variables set before NumPy/PyTorch import:

```
OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, NUMEXPR_NUM_THREADS
```

---

## 10. Performance Tips

### Data Loading (Downstream)

| Method | Load Time | RAM | Use Case |
|--------|-----------|-----|----------|
| Filtered load (PyArrow pushdown) | 2-10 s | ~200 MB | Development, prototyping |
| Optimized full load (column pruning) | 60-120 s | 4-6 GB | Production training |
| Lazy load (1000 rows) | < 1 s | < 100 MB | Schema exploration |
| Pre-filtered Parquet subset | 1-5 s | ~200 MB | Repeated workflows |

### Filter Pushdown

```python
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    filters={'Z': [92, 94], 'MT': [18, 102]}
)
```

PyArrow evaluates filters in C++ at the Parquet row-group level; only matching
row groups are read from disk.

### Column Pruning

Reading only needed columns reduces I/O by 50-60%:

```python
import pyarrow.parquet as pq
table = pq.read_table(
    'data/exfor_processed.parquet',
    columns=['Z', 'A', 'Energy', 'CrossSection', 'MT'],
    memory_map=True
)
```

### Sparse One-Hot Encoding

MT codes (117 unique values) use sparse arrays to avoid memory explosion:

| Dataset Size | Dense Memory | Sparse Memory | Reduction |
|--------------|-------------|---------------|-----------|
| 1M rows | 0.9 GB | 8 MB | 110x |
| 5M rows | 4.4 GB | 40 MB | 110x |
| 16.9M rows | 14.7 GB | 135 MB | 110x |

### Pre-Filtered Subsets

For repeated workflows, export a smaller Parquet file once:

```python
import pyarrow.parquet as pq
table = pq.read_table(
    'data/exfor_processed.parquet',
    filters=[('Z', 'in', [92, 94]), ('MT', 'in', [18, 102, 16, 17])]
)
pq.write_table(table, 'data/actinides_fission.parquet')
```
