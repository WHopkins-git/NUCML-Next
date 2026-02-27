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

With `--diagnostics`, these additional columns are extracted from `x4pro_ds`:

```
Year (year1), Author (author1), ReactionType (reatyp), NDataPoints (ndat)
```

`FullCode` is added via the `REACODE` table join in MetadataFilter when diagnostics is enabled.

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
| SF8 Tier 3 | Exclude | `ETA`, `ALF`, `RES`, `G`, `S0`, `2G`, `RM` |
| SF5 | Exclude | `PRE`, `SEC`, `TER`, `QTR`, `DL`, `PAR`, `CUM`, `(CUM)`, `(CUM)/M+`, `CHN`, `IND`, `UNW`, `(M)`, `M+`, `EXL`, `POT`, `1`-`4` |
| SF9 | Exclude | `CALC`, `DERIV`, `EVAL`, `RECOM` |
| SPSDD | Exclude | `'1'` (superseded), `'D'` (deprecated) |

Default filtering applies SF8 Tiers 1+2 plus all SF5, SF6, SF9, and SPSDD rules.

**Load-time spectrum-averaged filter:** In addition to ingestion-time filtering above,
`DataSelection(exclude_spectrum_averaged=True)` (the default) removes remaining
spectrum-averaged data at load time. This covers SF8 codes MXW, SPA, FIS, AV, BRA, BRS
(already excluded at ingestion Tier 2) **plus** SDT, FST, and TTA (kept during ingestion
but excluded at load time). Set `exclude_spectrum_averaged=False` to include all sf8 types.
See `selection.py → SPECTRUM_AVERAGED_SF8`.

**Compound SF8 handling:** EXFOR compound modifiers like `BRS/REL` or `BRA/MSC` are split on `/`
and each token is checked independently. For example, `BRS/REL` matches both the Tier 2 `BRS`
(bremsstrahlung) and Tier 1 `REL` (relative) rules.

### Intentionally Kept SF8 Codes

These SF8 modifiers are present in the dataset but are **not excluded** by the filter:

| Code | Count (approx.) | Meaning | Rationale for Keeping |
|------|-----------------|---------|----------------------|
| `RAW` | ~69K | Uncorrected (no dead-time, self-absorption corrections) | Real experimental measurements; may have systematic offsets but are valid data. Toggle ON/OFF in ThresholdExplorer. |
| `TTA` | ~17K | Thick-Target Approximation | Kept during ingestion but **excluded at load time** by default (`exclude_spectrum_averaged=True`). Set `False` to include. |
| `SDT` | ~2.4K | d-T spectrum averaged | Kept during ingestion but **excluded at load time** by default (`exclude_spectrum_averaged=True`). Set `False` to include. |

### EXFOR Code Reference

Comprehensive reference of EXFOR REACTION subfield codes encountered in the database,
their positions, physical meaning, and filter status.

**SF5 (Branch/Qualifier) — Position 5 after residual nucleus:**

| Code | Meaning | Filter Status |
|------|---------|---------------|
| `PRE` | Pre-neutron-emission (prompt fission) | **Excluded** |
| `SEC` | Secondary (post-neutron-emission) | **Excluded** |
| `TER` | Ternary fission | **Excluded** |
| `QTR` | Quaternary fission | **Excluded** |
| `DL` | Delayed (delayed neutrons, not total) | **Excluded** |
| `PAR` | Partial (cross section to specific level) | **Excluded** |
| `CUM` | Cumulative yield | **Excluded** |
| `(CUM)` | Cumulative yield (EXFOR bracket notation) | **Excluded** |
| `(CUM)/M+` | Cumulative including metastable | **Excluded** |
| `CHN` | Chain yield | **Excluded** |
| `IND` | Independent yield | **Excluded** |
| `UNW` | Unweighted average | **Excluded** |
| `(M)` | Metastable state only | **Excluded** |
| `M+` | Metastable + ground | **Excluded** |
| `EXL` | Exclusive (specific exit channel) | **Excluded** |
| `POT` | Potential scattering | **Excluded** |
| `1`-`4` | Numeric level (partial XS to specific excited state) | **Excluded** |
| *(empty)* | No branch qualifier (total cross section) | **Kept** |

**SF6 (Quantity Parameter) — Position 6:**

| Code | Meaning | Filter Status |
|------|---------|---------------|
| `SIG` | Cross section | **Kept** (whitelist) |
| `WID` | Resonance width | **Kept** (whitelist) |
| `SIG/RAT` | Cross section ratio | **Excluded** |
| `DA` | Differential d-sigma/d-Omega | **Excluded** (not in whitelist) |
| `DE` | Energy differential d-sigma/dE | **Excluded** |
| `FY` | Fission yield | **Excluded** |
| `NU` | Neutron multiplicity | **Excluded** |
| `RI` | Resonance integral | **Excluded** |
| `ARE` | Resonance area | **Excluded** |

**SF8 (Modifier) — Position 8:**

| Code | Tier | Meaning | Filter Status |
|------|------|---------|---------------|
| `REL` | 1 | Relative measurement (arbitrary units) | **Excluded** |
| `RAT` | 1 | Ratio to another quantity | **Excluded** |
| `RTE` | 1 | Rate (sigma * v) | **Excluded** |
| `FCT` | 1 | Arbitrary factor applied | **Excluded** |
| `MXW` | 2 | Maxwellian-averaged (E = kT) | **Excluded** |
| `SPA` | 2 | Spectrum-averaged (reactor spectrum) | **Excluded** |
| `FIS` | 2 | Fission-spectrum averaged | **Excluded** |
| `AV` | 2 | Averaged over energy interval | **Excluded** |
| `BRA` | 2 | Bremsstrahlung-spectrum weighted | **Excluded** |
| `BRS` | 2 | Bremsstrahlung-spectrum weighted (4-pi) | **Excluded** |
| `ETA` | 3 | eta = nu * sigma_f / sigma_a | **Excluded** |
| `ALF` | 3 | alpha = sigma_gamma / sigma_f | **Excluded** |
| `RES` | 3 | Resonance integral | **Excluded** |
| `G` | 3 | g-factor modified (Westcott) | **Excluded** |
| `S0` | 3 | S-wave neutron strength function (eV^-1/2) | **Excluded** |
| `2G` | 3 | 2g * Gamma_n^0 spin-statistical factor | **Excluded** |
| `RM` | 3 | R-Matrix parameters (model fit) | **Excluded** |
| `RAW` | — | Uncorrected experimental data | **Kept** (toggle in ThresholdExplorer) |
| `TTA` | — | Thick-Target Approximation | **Kept** at ingestion; **excluded at load time** by default |
| `SDT` | — | d-T spectrum averaged | **Kept** at ingestion; **excluded at load time** by default |
| `FST` | — | Fast-reactor spectrum averaged | **Excluded at load time** (`exclude_spectrum_averaged`) |
| *(empty)* | — | No modifier (standard measurement) | **Kept** |

**SF9 (Data Type) — Position 9:**

| Code | Meaning | Filter Status |
|------|---------|---------------|
| `EXP` | Experimental | **Kept** |
| `CALC` | Calculated (theoretical) | **Excluded** |
| `DERIV` | Derived from other measurements | **Excluded** |
| `EVAL` | Evaluated value | **Excluded** |
| `RECOM` | Recommended value | **Excluded** |
| *(empty)* | Not specified (assumed experimental) | **Kept** |

**Compound codes:** Common compound SF8 codes like `BRS/REL`, `BRA/REL`, `BRA/MSC`, `SDT/AV`,
and `SQ/S0` are split on `/` and each token is checked against the exclusion sets above.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| (default) | ON | Exclude non-pure and superseded data |
| `--include-non-pure` | OFF | Keep all data types including non-pure |
| `--include-superseded` | OFF | Keep superseded entries |
| `--diagnostics` | OFF | Preserve Author, Year, ReactionType, FullCode, NDataPoints for inspection |

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
| `year1` | `Year` (diagnostics only) |
| `author1` | `Author` (diagnostics only) |
| `reatyp` | `ReactionType` (diagnostics only) |
| `ndat` | `NDataPoints` (diagnostics only) |
| `fullCode` | `FullCode` (diagnostics only) |

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

### Local MAD Scoring (Recommended)

Selected with `--outlier-method local_mad`.

**Motivation.** Gaussian Process-based scoring produces 55--93% false-positive rates on
resonance-region nuclear cross-section data. No GP kernel choice (long lengthscale,
short lengthscale, energy-dependent lengthscale, energy-dependent outputscale) can
simultaneously model resonance structure from subsampled data and provide calibrated
uncertainty. The task does not require modelling resonance structure -- it requires
identifying points that are clearly wrong compared to what other experiments measured
at similar energies.

**Algorithm.** Direct statistical scoring without GP fitting:

```
score_dataframe(df)
  -> Group by (Z, A, MT, Projectile)
  -> For each group:
       1. Pool all experiments' log_E, log_sigma
       2. Fit sigma-clipped spline smooth mean on pooled data
       3. Compute residuals: r = log_sigma - mean_fn(log_E)
       4. Compute rolling MAD in sliding energy windows
          Returns callable: mad_fn(log_E) -> local_MAD
       5. Z-score every point: z = |r| / mad_fn(log_E)
       6. Point outlier: z > point_z_threshold (default 3.0)
       7. Experiment discrepancy: fraction(z > exp_z_threshold) > exp_fraction_threshold
```

**Rolling MAD.** The rolling MAD uses adaptive energy windows (default 10% of the
total energy range, minimum 15 points). In sparse regions, it expands to the k-nearest
neighbours. The MAD is smoothed with a median filter and floored at `mad_floor`
(default 0.01) to prevent division by zero.

**Experiment discrepancy.** An experiment is flagged as discrepant if more than
`exp_fraction_threshold` (default 30%) of its points have z-scores above
`exp_z_threshold` (default 3.0). Single-experiment groups cannot be flagged as
discrepant (no comparison baseline).

**Performance.** O(n log n) per group (binary-search windowing via `np.searchsorted`).
No Cholesky decomposition, no subsampling. Groups are processed in parallel across
50% of CPU cores by default via `ProcessPoolExecutor`.

| Config field | Default | Description |
|-------------|---------|-------------|
| `mad_window_fraction` | 0.1 | Rolling window as fraction of energy range |
| `mad_min_window_points` | 15 | Minimum points per MAD window |
| `mad_floor` | 0.01 | Minimum MAD (prevents div-by-zero) |
| `exp_z_threshold` | 3.0 | z-score threshold for counting "bad" points |
| `exp_fraction_threshold` | 0.30 | Fraction of bad points to flag experiment |

**Measurement uncertainty.** When `mad_use_measurement_uncertainty=True` (default),
reported EXFOR measurement uncertainties are combined with the local MAD in quadrature:
`effective_sigma = sqrt(local_MAD² + sigma_measurement²)`, where
`sigma_measurement = 0.434 * (d_sigma / sigma)` in log10 space. This prevents false
positives in thermal/fast regions where MAD is small but systematic differences between
experiments are within reported uncertainties.

**Output columns.** `gp_mean` contains the smooth mean, `gp_std` contains the
effective sigma (local MAD + measurement uncertainty in quadrature),
`calibration_metric` and `outlier_probability` are NaN.

**Edge cases:**

| Condition | Handling |
|-----------|----------|
| Group < 10 points | MAD fallback (global median) |
| Single-experiment group | Point scoring works; `experiment_outlier` always False |
| All data identical | MAD = floor, z-scores near zero |
| NaN/inf values | Filtered before fitting |

---

### Smooth Mean

**Motivation.** A constant mean `mu = mean(log_sigma)` forces the GP residual to contain
5+ orders of magnitude of energy-dependent trend (1/v baseline at thermal energies,
resonance peaks in the keV region, threshold rise toward the MeV continuum). A single
stationary lengthscale cannot simultaneously capture narrow resonances and smooth 1/v
structure. Subtracting a smooth trend first lets the GP model only deviations from the
gross energy envelope.

**Algorithm.** Iterative reweighted cubic B-spline (`scipy.interpolate.UnivariateSpline`)
fitted in log-E vs log-sigma space:

1. Pool all experiments in a (Z, A, MT) group before per-experiment splitting.
2. Sort by log-E, filter NaN/inf.
3. Fit initial spline with smoothing factor `s = n * Var(y)` to force O(10) knots,
   capturing only gross energy dependence.
4. Compute residuals, estimate robust scale via MAD (Median Absolute Deviation).
5. Downweight points with |residual| > 3 * MAD (Huber-like soft clipping).
6. Refit spline with updated weights. Iterate up to 5 times or until convergence
   (max change < 1e-4).

**Smoothing factor override.** The scipy default `s ~ n` allows up to n knots, which
can trace individual resonances for small groups (50-200 points). We override with
`s = n * Var(y)`, producing ~10 knots that capture only the gross energy-dependent
envelope (1/v, threshold rise, giant resonances), not individual narrow resonances.

**Evaluation independence.** The smooth mean is computed from EXFOR data itself,
not from evaluated nuclear data libraries (ENDF/B, JEFF, JENDL). This preserves
the tool's role as an independent quality assessment resource for evaluators.

**Group-level pooling.** The smooth mean is computed from ALL experiments' pooled data.
The pooled mean from multiple measurements provides a robust baseline that dilutes any
single experiment's systematic bias.

**Single-experiment limitation.** For groups with only one experiment, the smooth mean
is fitted to just that experiment. If the experiment is biased, the mean absorbs the
bias and outlier detection power is reduced.

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
experiment_outlier, point_outlier, calibration_metric, experiment_id,
Year, Author, ReactionType, FullCode, NDataPoints
```

| Column Group | Present When |
|--------------|-------------|
| Core (Entry through Uncertainty) | Always |
| Metadata (sf5 through data_type) | Metadata filtering is ON (default) |
| GP columns (log_E through z_score) | `--outlier-method` is set |
| Experiment columns (experiment_outlier through experiment_id) | `--outlier-method local_mad` |
| Diagnostic columns (Year through NDataPoints) | `--diagnostics` is set |

### Diagnostic Columns

When `--diagnostics` is set, these additional columns from `x4pro_ds` are preserved through the pipeline:

| Column | Source | Description |
|--------|--------|-------------|
| `Year` | `x4pro_ds.year1` | Publication year of the experiment |
| `Author` | `x4pro_ds.author1` | First author of the experiment |
| `ReactionType` | `x4pro_ds.reatyp` | X4Pro reaction type code (CS, DAP, RP, etc.) |
| `FullCode` | `REACODE.fullCode` | Full EXFOR REACTION string for parsing SF subfields |
| `NDataPoints` | `x4pro_ds.ndat` | Number of data points in the original dataset |

These columns are intended for use with the `Diagnostics_Interactive_Inspector.ipynb` notebook, which provides Plotly hover-over inspection of individual data points.

---

## 8. CLI Reference

```
python scripts/ingest_exfor.py [FLAGS]
```

### Full Flag Table

**Core:**

| Flag | Default | Description |
|------|---------|-------------|
| `--x4-db` | required | Path to X4Pro SQLite database |
| `--output` | `data/exfor_processed.parquet` | Output Parquet path |

**Subset filtering:**

| Flag | Default | Description |
|------|---------|-------------|
| `--test-subset` | OFF | Use test subset: U (Z=92) + Cl (Z=17) |
| `--z-filter` | None | Comma-separated Z values (e.g., `79,92,26`) |

**Metadata filtering:**

| Flag | Default | Description |
|------|---------|-------------|
| `--include-non-pure` | OFF | Include non-pure data (relative, ratio, averaged, etc.) |
| `--include-superseded` | OFF | Include superseded entries |
| `--diagnostics` | OFF | Add Author, Year, ReactionType, FullCode, NDataPoints columns for interactive inspection |

**Outlier detection:**

| Flag | Default | Description |
|------|---------|-------------|
| `--outlier-method` | None | `local_mad` (recommended) or `svgp` (legacy) |
| `--z-threshold` | 3.0 | Z-score threshold for point outliers |
| `--exp-z-threshold` | 3.0 | Z-score threshold for counting bad points in experiment discrepancy (local_mad only) |
| `--exp-fraction-threshold` | 0.30 | Fraction of bad points to flag experiment as discrepant (local_mad only) |
| `--svgp-device` | `cpu` | `cpu` or `cuda` (SVGP only) |
| `--svgp-likelihood` | `student_t` | Likelihood: `student_t`, `heteroscedastic`, `gaussian` (SVGP only) |

**System:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-threads` | 50% of cores | CPU threads for NumPy/PyTorch linear algebra |

**Deprecated (still accepted, will be removed in a future release):**

| Flag | Replacement | Description |
|------|-------------|-------------|
| `--ame2020-dir` | None (AME is loaded at feature-generation time) | Ignored; AME enrichment no longer happens during ingestion |
| `--run-svgp` | `--outlier-method svgp` | Legacy flag for SVGP outlier detection |
| `--no-svgp` | (default behaviour) | Explicitly skip outlier detection |

### Example Commands

```bash
# Standard ingestion (full database, ~13M points)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

# Test subset (~300K points, minutes not hours)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

# Local MAD outlier detection (recommended)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method local_mad

# Custom experiment discrepancy thresholds
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method local_mad \
    --exp-z-threshold 3 --exp-fraction-threshold 0.25

# Custom element subset (Gold, Uranium, Iron)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --z-filter 79,92,26

# Include all data (no metadata filtering)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --include-non-pure --include-superseded

# Diagnostic mode (adds hover metadata for interactive inspector)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --test-subset --diagnostics
```

---

## 9. Resource Management

| Resource | Policy |
|----------|--------|
| CPU threads | 50% of available cores by default (shared machine etiquette); override with `--num-threads` |
| Parallel workers | Local MAD uses ProcessPoolExecutor with n_workers = --num-threads |
| Memory | O(n) per group for local MAD (no covariance matrices) |

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
