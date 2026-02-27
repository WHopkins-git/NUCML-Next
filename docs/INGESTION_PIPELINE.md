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
| `scoring_method` | `'gp'` | Set to `'local_mad'` for this method |
| `mad_window_fraction` | 0.1 | Rolling window as fraction of energy range |
| `mad_min_window_points` | 15 | Minimum points per MAD window |
| `mad_floor` | 0.01 | Minimum MAD (prevents div-by-zero) |
| `exp_z_threshold` | 3.0 | z-score threshold for counting "bad" points |
| `exp_fraction_threshold` | 0.30 | Fraction of bad points to flag experiment |

**Output columns:** Same as per-experiment GP for downstream compatibility.
`gp_mean` contains the smooth mean, `gp_std` contains the local MAD,
`calibration_metric` and `outlier_probability` are NaN.

**Edge cases:**

| Condition | Handling |
|-----------|----------|
| Group < 10 points | MAD fallback (global median) |
| Single-experiment group | Point scoring works; `experiment_outlier` always False |
| All data identical | MAD = floor, z-scores near zero |
| NaN/inf values | Filtered before fitting |

---

### Per-Experiment GP (Legacy)

Selected with `--outlier-method experiment`.

| Aspect | Detail |
|--------|--------|
| Approach | Fits independent Exact GPs to each experiment (Entry) within a (Z, A, MT) group |
| Kernel | RBF (default) with heteroscedastic noise; Gibbs nonstationary kernel available via `KernelConfig(kernel_type='gibbs')` with RIPL-3 physics-informed lengthscale (see 6.2) |
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

### 6.1 Smooth Mean (Phase 1)

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

**Group-level pooling.** The smooth mean is computed from ALL experiments' pooled data
before per-experiment GP fits. The pooled mean from multiple measurements provides a
robust baseline that dilutes any single experiment's systematic bias.

**Single-experiment limitation.** For groups with only one experiment, the smooth mean
is fitted to just that experiment. If the experiment is biased, the mean absorbs the
bias and outlier detection power is reduced. This is inherent to Phase 1 and requires
cross-isotope information (Phase 4+) to address.

**Configuration.** `SmoothMeanConfig(smooth_mean_type='spline')` opts in. Default
`'constant'` preserves exact pre-Phase-1 behaviour.

### 6.2 Kernel Abstraction & Physics-Informed Gibbs Kernel (Phase 2)

**The fundamental problem.** A stationary RBF kernel `K(x,x') = sigma^2 exp(-dx^2 / 2l^2)`
uses a single lengthscale l across all energies. Nuclear cross-sections have structure at
vastly different scales: narrow resonances (Gamma ~ 0.1 eV) in the resolved resonance
region (RRR) and smooth MeV-scale trends in the continuum. No single l works for both
regimes. For U-235 fission, this mismatch produces a 55% false-positive rate at z > 3
(expected 0.3% for a calibrated GP).

**Kernel abstraction.** All GP code now delegates kernel computation to a `Kernel` object
(abstract base class in `nucml_next.data.kernels`). The RBF formula, previously hardcoded
in 4 locations (`experiment_gp.py` lines 311, 374; `calibration.py` lines 170, 448), is
encapsulated in `RBFKernel`. New kernels can be added by subclassing `Kernel` without
modifying the GP or calibration code.

**Gibbs kernel.** Nonstationary kernel with position-dependent lengthscale:

    K(xi, xj) = sigma^2 * sqrt(2 li lj / (li^2 + lj^2)) * exp(-(xi - xj)^2 / (li^2 + lj^2))

This reduces to RBF when li = lj = l (constant). The energy-dependent lengthscale
l(log_E) adapts automatically: short near narrow resonances, long in smooth regions.

**Physics-informed lengthscale from RIPL-3.** The optimal GP lengthscale should track
the mean level spacing D(E) ~ 1/rho(E), where rho is the nuclear level density.
RIPL-3 provides Constant Temperature (CT) level density parameters (T, U0) per nuclide
from nuclear structure data (discrete level schemes, neutron resonances). These are
evaluation-independent.

CT formula:

    rho(E_x) = (1/T) * exp((E_x - U0) / T)
    D(E_x) = T * exp(-(E_x - U0) / T)

where E_x = S_n + E_n is the compound nucleus excitation energy (S_n = neutron
separation energy, E_n = incident neutron energy).

**Lengthscale parameterization:**

    l(log_E) = softplus(log D(E) + a0 + a1 * log_E)

Only 2 free parameters (a0, a1) on top of the physics. The RIPL-3 data does the heavy
lifting: it captures the 10^6 dynamic range from narrow resonances (D ~ 0.1 eV) to
smooth continuum (D ~ 10^5 eV). The correction terms handle the fact that the optimal
GP lengthscale is not exactly D(E) but is proportional to it.

**Why not polynomial or piecewise?** A polynomial in log E has one extremum (for degree 2),
too rigid for the RRR-to-URR transition, with 3 parameters trying to span 10^6 dynamic
range. Piecewise linear requires arbitrary knot placement unless one uses nuclear structure
information that RIPL-3 already encodes explicitly. The RIPL-3 approach lets physics
determine the shape with minimal free parameters.

**CT formula floor.** The CT formula diverges when S_n + E_n < U0 (unphysical regime).
A floor is applied: `D(E) = max(D_computed, D(S_n))`, where D(S_n) is the mean level
spacing at the neutron separation energy. This ensures the GP lengthscale never exceeds
the thermal-region value, which is physically the smoothest region.

**S_n computation.** The neutron separation energy is needed to convert incident neutron
energy to compound nucleus excitation energy. It can be computed from AME2020 mass
excess data (already available via `AME2020Loader`): `S_n = M(Z,A) + M_n - M(Z,A+1)`.
A rough empirical fallback is used when AME2020 data is not available.

**Wasserstein calibration.** The correction parameters (a0, a1) are optimised via
Nelder-Mead on the Wasserstein distance between LOO |z|-scores and the half-normal
distribution. For RBF, this is unchanged (1D Brent search over lengthscale). For Gibbs,
it is a 2D search that converges reliably given the physical initialisation.

**Outputscale.** Estimated from `Var(residuals) - mean(noise)`, never optimised.
Consistent across both RBF and Gibbs kernels.

**Data source:**

| File | Content | Source |
|------|---------|--------|
| `data/levels/levels-param.data` | CT parameters (T, U0) per nuclide | [IAEA RIPL-3](https://www-nds.iaea.org/RIPL-3/) |

**Configuration.** `KernelConfig(kernel_type='gibbs')` opts in. Default `None` (or
`kernel_type='rbf'`) preserves exact pre-Phase-2 behaviour.

**Injection path.** The RIPL-3 interpolator flows through the system as follows:

    ExperimentOutlierDetector._score_multi_experiment(group_key=(Z, A, MT))
      -> _build_kernel_config_for_group(Z, A)
        -> RIPL3LevelDensity.get_log_D_interpolator(Z, A+1, S_n)  [compound nucleus]
        -> KernelConfig(ripl_log_D_interpolator=interpolator)
      -> _fit_experiment_gp(exp_df, kernel_config=...)
        -> ExactGPExperiment(config_with_kernel)
          -> build_kernel(config.kernel_config)  ->  GibbsKernel or RBF

The kernel object never loads files or knows about (Z, A). File I/O happens once in
the RIPL-3 loader; all subsequent calls use closures over precomputed parameters.

### 6.3 Robust Likelihood — Contaminated Normal (Phase 3)

**Motivation.** The default Gaussian noise model `K += diag(σᵢ²)` treats every
data point as equally trustworthy. When a point is a genuine outlier (transcription
error, wrong units, corrupted digitisation), the GP treats the discrepancy as signal.
This distorts the posterior mean, inflates prediction uncertainty for nearby points,
and produces unreliable z-scores. A contaminated normal mixture provides principled
outlier identification without distorting the GP posterior.

**Model.** Each observation is modeled as a two-component Gaussian mixture:

```
p(yᵢ | fᵢ) = (1-ε)·N(yᵢ; fᵢ, σᵢ²) + ε·N(yᵢ; fᵢ, κ·σᵢ²)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| ε | 0.05 | Prior contamination fraction (fixed) |
| κ | 10.0 | Noise inflation for outlier component (fixed) |

Both ε and κ are fixed (not optimized) — a design choice for simplicity and robustness.
The outlier component has κ times the noise variance, making it "listen less" to outlier
points during GP fitting.

**EM algorithm.** After kernel optimization and initial Cholesky factorisation, the EM
loop iterates:

1. **M-step:** Compute effective noise `σ_eff² = σᵢ²·(1 + wᵢ·(κ-1))` from current
   outlier weights `wᵢ`. Rebuild `K_noisy = K_kernel + diag(σ_eff²) + jitter·I`.
   Cholesky factorize and solve for `alpha`.
2. **E-step:** Compute posterior residuals `eᵢ = σ_eff²[i]·alpha[i]`. Compute
   log-likelihood ratio of outlier vs inlier components. Update outlier weights
   via log-space sigmoid: `wᵢ = σ(log(ε/(1-ε)) + log_r)`.
3. **Convergence:** Stop when `max|wᵢ_new - wᵢ_old| < tol` (default 1e-3, max 10 iterations).

The E-step uses log-space computation with clamped exponents for numerical stability.

**Integration.** The EM runs *after* the initial `_build_prediction_cache()` call in
`ExactGPExperiment.fit()`. It updates `self._L` and `self._alpha` in-place with the
EM-refined effective noise. Crucially, `predict()` and `get_loo_z_scores()` are **not
modified** — they automatically become robust because they use the EM-updated Cholesky
factor and alpha vector.

**Output.** The `outlier_probability` column (continuous float in [0, 1]) gives the
posterior contamination weight per point. When the contaminated likelihood is active,
`point_outlier` is set to `outlier_probability > 0.5` (replacing the z-score threshold).
Points without outlier probability (e.g., from MAD fallback) still use the z-score
threshold.

**Configuration.** Opt-in via `LikelihoodConfig(likelihood_type='contaminated')`.
The default `likelihood_type='gaussian'` preserves exact pre-Phase-3 behaviour with
no EM and no `outlier_probability` column. Implementation in
`nucml_next.data.likelihood`.

### 6.4 Hierarchical Experiment Structure (Phase 4)

#### Motivation

When fitting independent GPs to multiple experiments within a (Z, A, MT) group,
each experiment's hyperparameters (lengthscale, outputscale, and — for the Gibbs
kernel — the correction parameters a₀, a₁) are estimated from that experiment's
data alone. For small experiments (10–50 data points spanning a narrow energy
range), the Wasserstein-calibrated lengthscale can be poorly constrained: the
optimisation landscape may be flat, or a local minimum may produce a physically
implausible lengthscale (e.g. a lengthscale shorter than the energy spacing, or
longer than the entire energy range). The outputscale, estimated as
`Var(residuals) − mean(noise)`, is similarly noisy for small experiments.

This heterogeneity in hyperparameters has three consequences:

1. **Inconsistent predictions.** Two experiments measuring the same cross section
   at overlapping energies can produce very different GP posteriors, making
   consensus building unreliable.
2. **Gibbs correction drift.** The a₀ and a₁ parameters of the Gibbs kernel
   encode corrections to the RIPL-3 level density. These should be consistent
   across experiments of the same nucleus, as they describe nuclear structure
   rather than experimental conditions. Independent fitting allows them to
   absorb experiment-specific noise.
3. **Small-experiment fragility.** Experiments with few data points tend to
   produce extreme lengthscales that cause either over-fitting (short
   lengthscale) or over-smoothing (long lengthscale), both degrading outlier
   detection.

#### Algorithm: Two-Pass Hierarchical Fitting

**Pass 1** (unchanged): Fit all experiments independently using the existing
Wasserstein-calibrated pipeline (kernel optimisation → Cholesky → optional EM).
This produces one `ExactGPExperiment` object per experiment with independently
estimated hyperparameters.

**Pass 2** (new, opt-in via `hierarchical_refitting=True`):

1. **Extract group statistics.** From all successfully fitted GPs in the group
   (minimum `min_experiments_for_refit`, default 3), collect:
   - Outputscale values → compute group median
   - Kernel optimisable parameters (lengthscale for RBF; a₀, a₁ for Gibbs) →
     compute per-parameter Q1, median, Q3

2. **Compute constrained bounds.** For each kernel parameter:
   ```
   IQR = Q3 − Q1
   lower = Q1 − margin × IQR
   upper = Q3 + margin × IQR
   ```
   where `margin = refit_bounds_iqr_margin` (default 1.0). This is the standard
   IQR-based outlier fence used in robust statistics (with the conventional
   factor of 1.5 replaced by a configurable margin).

   **Kernel-type-aware clipping.** RBF lengthscale bounds are floored at 1e-3
   (lengthscale must be strictly positive). Gibbs correction parameters a₀, a₁
   are NOT clipped — they can take any real value, as a negative a₀ means the
   GP lengthscale is shorter than the RIPL-3 prediction.

   **Degenerate IQR handling.** When all experiments produce identical parameters
   (IQR = 0), bounds are expanded to [median × 0.5, median × 2.0] for positive
   parameters, or [median − 0.5, median + 0.5] for parameters near zero.

3. **Re-fit each experiment.** Call `refit_with_constraints()` on each GP with:
   - Shared outputscale: group median (when `refit_share_outputscale=True`)
   - Parameter bounds: the IQR-based bounds computed above
   - Same Wasserstein optimiser, but the search is now constrained to the
     group-informed feasible region

   The refit skips data preparation (input validation, subsampling, mean function
   estimation — all preserved from Pass 1) and only re-runs: kernel parameter
   optimisation → Cholesky factorisation → optional EM.

4. **Update predictions.** The refit GPs produce updated `gp_mean`, `gp_std`,
   `z_score`, `calibration_metric`, and `outlier_probability` values that replace
   the Pass 1 predictions in the result DataFrame.

**Ordering.** Pass 2 runs BEFORE consensus building. This is important because
the consensus builder receives the refit GPs with more consistent hyperparameters,
producing a tighter and more reliable consensus posterior. Discrepancy detection
(experiment-level flagging) then operates on the improved consensus.

#### Failure Handling

Refit failure for any individual experiment is non-fatal: a warning is logged and
the Pass 1 results are preserved. This ensures one problematic experiment cannot
block the group.

When fewer than `min_experiments_for_refit` GPs are successfully fitted in Pass 1,
Pass 2 is skipped entirely for that group (IQR from 2 points is the full range —
too unreliable for constraint estimation).

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hierarchical_refitting` | `False` | Enable two-pass fitting |
| `min_experiments_for_refit` | `3` | Min GPs for reliable group statistics |
| `refit_bounds_iqr_margin` | `1.0` | IQR multiplier for bound width |
| `refit_share_outputscale` | `True` | Share group median outputscale in Pass 2 |

Default `False` preserves exact pre-Phase-4 behaviour.

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
| Experiment columns (experiment_outlier through experiment_id) | `--outlier-method local_mad` or `experiment` |
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
| `--outlier-method` | None | `local_mad` (recommended), `experiment` (legacy GP), or `svgp` (legacy) |
| `--z-threshold` | 3.0 | Z-score threshold for point outliers |
| `--exp-z-threshold` | 3.0 | Z-score threshold for counting bad points in experiment discrepancy (local_mad only) |
| `--exp-fraction-threshold` | 0.30 | Fraction of bad points to flag experiment as discrepant (local_mad only) |
| `--svgp-device` | `cpu` | `cpu` or `cuda` |
| `--max-gpu-points` | 40000 | Max points per experiment on GPU; larger auto-route to CPU |
| `--max-subsample-points` | 15000 | Subsample large experiments to this many points for GP fitting |
| `--svgp-checkpoint-dir` | None | Enable checkpointing for resume on interruption |
| `--svgp-likelihood` | `student_t` | Likelihood: `student_t`, `heteroscedastic`, `gaussian` |
| `--smooth-mean` | `constant` | Mean function for per-experiment GP: `constant` or `spline` (§6.1) |
| `--kernel-type` | `rbf` | GP kernel: `rbf` or `gibbs` (§6.2, requires `--ripl-data-path`) |
| `--ripl-data-path` | None | Path to RIPL-3 `levels-param.data` file (required for `gibbs`) |
| `--likelihood` | `gaussian` | GP likelihood: `gaussian` or `contaminated` (§6.3) |
| `--hierarchical-refitting` | OFF | Enable two-pass group-constrained refit (§6.4) |

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

# Diagnostic mode (adds hover metadata for interactive inspector)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --test-subset --diagnostics

# Full Phase 1-4 GP stack
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment \
    --smooth-mean spline \
    --kernel-type gibbs --ripl-data-path data/levels-param.data \
    --likelihood contaminated \
    --hierarchical-refitting

# Contaminated likelihood + hierarchical refit (no Gibbs kernel)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment \
    --likelihood contaminated \
    --hierarchical-refitting
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
