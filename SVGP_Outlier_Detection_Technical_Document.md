# Gaussian Process Outlier Detection for Nuclear Cross-Section Data Quality Assurance

**NUCML-Next Data Pipeline Documentation**

---

## Abstract

This document describes the implementation of Gaussian Process methodologies
for automated outlier detection in experimental nuclear cross-section data
from the EXFOR database. Two approaches are documented:

1. **Per-Experiment GP (Recommended):** Fits independent Exact GPs to each
   EXFOR experiment (Entry) within a reaction group, builds consensus from
   multiple experiments, and flags discrepant experiments systematically.

2. **Sparse Variational GP (Legacy):** Pools all experiments per reaction
   group and fits a single SVGP for point-level z-scores.

Both approaches address the critical challenge of identifying erroneous
measurements, transcription errors, and systematically discrepant datasets
within the 13.4 million data points comprising the processed EXFOR archive.
Unlike conventional outlier detection methods that treat data as unstructured
point clouds, GP-based approaches explicitly model cross-sections as functions
of incident particle energy, enabling physically meaningful anomaly detection
that respects the underlying functional structure of nuclear reaction data.

---

## 1. Introduction

### 1.1 Background

The Experimental Nuclear Reaction Data (EXFOR) database represents the most
comprehensive compilation of measured nuclear reaction data, maintained by
the International Atomic Energy Agency (IAEA) Nuclear Data Section. The
database contains cross-section measurements spanning seven decades of
experimental nuclear physics, encompassing diverse measurement techniques,
detector technologies, and data reduction methodologies. This heterogeneity,
while providing invaluable breadth of coverage, introduces significant data
quality challenges including transcription errors, digitisation artifacts,
unit conversion mistakes, and systematically discrepant experimental
campaigns.

Machine learning approaches to nuclear data evaluation require high-quality
training datasets. The presence of outliers and erroneous measurements can
significantly degrade model performance, introduce spurious correlations,
and compromise the reliability of predictions. Traditional manual curation
of such large datasets is impractical, necessitating automated quality
assurance methodologies that can identify problematic data points while
preserving legitimate physical phenomena such as resonance structures.

### 1.2 Limitations of Conventional Approaches

Previous work in nuclear data quality assurance has employed statistical
outlier detection methods including Local Outlier Factor (LOF) and One-Class
Support Vector Machines (OC-SVM). These approaches treat the data as
unstructured point clouds in $(E, \sigma)$ space, computing anomaly scores
based on local density or distance to learned decision boundaries. While
computationally tractable, such methods exhibit a fundamental limitation:
they fail to account for the functional relationship between energy and
cross-section.

Consider a resonance peak in the neutron-induced fission cross-section of
$^{235}$U. The cross-section may increase by several orders of magnitude
over a narrow energy range, creating data points that appear isolated in the
joint $(E, \sigma)$ distribution. Density-based methods such as LOF would
flag these physically real phenomena as outliers due to their low local
density, while genuine transcription errors situated near the baseline might
escape detection due to the high data density in that region.

### 1.3 The Functional Perspective

Nuclear cross-sections are fundamentally functions of incident particle
energy: $\sigma = \sigma(E)$. The appropriate question for outlier detection
is not whether a given $(E, \sigma)$ pair is unusual within the dataset, but
rather whether the observed cross-section is anomalous *for this specific
reaction at this specific energy*. This conditional formulation --
$P(\sigma \mid E, Z, A, \text{MT})$ -- motivates the use of Gaussian
Process regression, which explicitly models the functional relationship and
provides calibrated uncertainty estimates.

### 1.4 The Per-Experiment Approach

The original SVGP implementation pools all experiments within a (Z, A, MT)
group. While computationally efficient, this approach has limitations:

1. **Masking effect:** Discrepant experiments pull the GP mean toward them
   and inflate the learned noise variance, reducing sensitivity to outliers.

2. **Resonance over-smoothing:** A single RBF lengthscale optimized for the
   pooled data tends to over-smooth resonance structure, causing high outlier
   rates (e.g., 25% on U-233(n,f) at z=3).

3. **Wrong abstraction level:** Nuclear data evaluators typically assess
   entire experimental datasets, not individual points. Flagging points
   without considering experiment-level systematics misses this structure.

The per-experiment approach addresses these limitations by:

- Fitting independent Exact GPs to each EXFOR Entry (experiment)
- Using heteroscedastic noise from measurement uncertainties
- Building consensus from multiple experiment posteriors
- Flagging entire experiments that deviate from consensus
- Calibrating lengthscale via Wasserstein distance for well-calibrated z-scores

---

## 2. Theoretical Framework

### 2.1 Gaussian Process Regression

A Gaussian Process (GP) defines a probability distribution over functions,
specified by a mean function $m(x)$ and covariance function $k(x, x')$. For
nuclear cross-section modelling, we work in log-space to handle the large
dynamic range of cross-section values, defining the regression problem as:

$$\log_{10}(\sigma) = f(\log_{10}(E)) + \varepsilon, \quad \text{where } f \sim \mathcal{GP}(m, k) \text{ and } \varepsilon \sim \mathcal{N}(0, \sigma_n^2)$$

The covariance function encodes assumptions about the smoothness and
structure of the underlying function. We employ the Radial Basis Function
(RBF) kernel, also known as the squared exponential kernel:

$$k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)$$

where $\sigma_f^2$ is the signal variance controlling the amplitude of
variations and $\ell$ is the lengthscale parameter governing the correlation
distance in log-energy space. The RBF kernel assumes smooth, infinitely
differentiable functions, which is appropriate for cross-section data away
from sharp resonances.

### 2.2 Sparse Variational Approximation

Exact GP inference requires $\mathcal{O}(n^3)$ computation and
$\mathcal{O}(n^2)$ memory for the Cholesky decomposition of the
$n \times n$ covariance matrix, rendering it intractable for large datasets.
Sparse Variational Gaussian Processes (SVGP) address this limitation by
introducing a set of $m$ inducing points
$\{z_j, u_j\}_{j=1}^{m}$ that summarise the posterior distribution.

The variational approximation assumes the posterior factorises through the
inducing points:

$$p(f \mid y) \approx \int p(f \mid u)\, q(u)\, du$$

where $q(u)$ is a variational distribution (typically multivariate Gaussian)
optimised to minimise the Kullback-Leibler divergence to the true posterior.
This reduces computational complexity to $\mathcal{O}(nm^2)$, enabling
application to datasets with millions of observations.

The variational parameters are optimised jointly with the kernel
hyperparameters by maximising the Evidence Lower Bound (ELBO):

$$\text{ELBO} = \sum_{i=1}^{n} \mathbb{E}_{q(f_i)}\!\left[\log p(y_i \mid f_i)\right] - \text{KL}\!\left[q(u) \,\|\, p(u)\right]$$

### 2.3 Outlier Score Formulation

Given the trained SVGP model, the predictive distribution at each observed
energy point $E_i$ is Gaussian with mean $\mu(E_i)$ and variance
$\sigma^2(E_i)$. The outlier score is defined as the standardised residual
(z-score):

$$z_i = \frac{|\log_{10}(\sigma_i) - \mu(E_i)|}{\sigma(E_i)}$$

Under the GP model assumptions, z-scores follow a folded standard normal
distribution. Points with $|z|$ exceeding a user-specified threshold
(typically 3.0, corresponding to 99.7% of the distribution) are flagged as
potential outliers. Critically, the uncertainty $\sigma(E_i)$ is
heteroscedastic -- it varies with energy, expanding in regions of sparse
data or complex structure and contracting where measurements are dense and
consistent.

**MAD fallback and the consistency constant.** For groups too small for
reliable GP fitting (fewer than 10 points), we employ the Median Absolute
Deviation (MAD) as a robust scale estimator:

$$\text{MAD} = \text{median}\!\left(|x_i - \tilde{x}|\right), \quad \hat{\sigma} = 1.4826 \times \text{MAD}$$

The constant 1.4826 is the reciprocal of the 75th percentile of the
standard normal distribution: $1 / \Phi^{-1}(3/4) = 1 / 0.6745 \approx 1.4826$.
This scaling makes $\hat{\sigma}$ a consistent estimator of the standard
deviation for normally distributed data, ensuring that z-scores computed via
MAD are comparable to those from the GP.

### 2.4 Likelihood Models

The choice of likelihood function significantly impacts how the GP handles
measurement noise and outliers. Three likelihood options are supported,
addressing different data characteristics.

#### 2.4.1 Student-t Likelihood (Default)

Nuclear cross-section measurements often exhibit heavy-tailed error
distributions due to systematic effects, calibration uncertainties, and
occasional gross errors. The Student-t likelihood provides robustness to
such outliers by using a probability density with heavier tails than the
Gaussian:

$$p(y \mid f, \sigma, \nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\pi\nu}\sigma} \left(1 + \frac{(y-f)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$$

where $\nu$ is the degrees of freedom parameter (learned from data) and
$\sigma$ is the scale parameter. For $\nu > 2$, the variance is:

$$\text{Var}(y \mid f) = \sigma^2 \cdot \frac{\nu}{\nu - 2}$$

Nuclear data typically learns $\nu \approx 2$--$4$, indicating substantial
heavy tails. This allows genuine outliers to have less influence on the GP
fit, improving the quality of the smooth trend estimate.

#### 2.4.2 Heteroscedastic Gaussian Likelihood

When calibrated measurement uncertainties are available (stored in the
`Uncertainty` column), heteroscedastic modelling uses per-point noise
variance derived from these uncertainties:

$$y_i \sim \mathcal{N}(f(x_i), \sigma_i^2)$$

where $\sigma_i$ varies per observation. The transformation from relative
uncertainty $\delta_{\text{rel}} = \Delta\sigma/\sigma$ to log-space
uncertainty follows from error propagation:

$$\sigma_{\log_{10}} = \frac{\delta_{\text{rel}}}{\ln(10)} \approx 0.434 \cdot \delta_{\text{rel}}$$

This approach uses GPyTorch's `FixedNoiseGaussianLikelihood`, with an
optional learnable residual noise term for unmodelled variance. Missing
uncertainties are conservatively filled with the maximum valid uncertainty
in the group.

#### 2.4.3 Homoscedastic Gaussian Likelihood (Legacy)

The original implementation assumes constant noise variance across all
observations:

$$y_i \sim \mathcal{N}(f(x_i), \sigma_n^2)$$

where a single $\sigma_n^2$ is learned from data. While computationally
simple, this assumption leads to constant-width uncertainty bands that fail
to reflect the varying data density and measurement quality across the
energy range. This option is retained for backwards compatibility but is no
longer recommended.

### 2.5 Wasserstein Calibration (Per-Experiment Method)

For the per-experiment approach, lengthscale is optimized to produce
well-calibrated z-scores rather than maximizing marginal likelihood. Under
a well-calibrated GP, the leave-one-out (LOO) z-scores should follow a
standard normal distribution, and their absolute values should follow a
half-normal distribution.

The Wasserstein distance (1-Wasserstein, or Earth Mover's Distance) quantifies
how far the empirical |z| distribution deviates from the half-normal:

$$W_1 = \int_0^\infty |F_{|z|}(t) - F_{\text{half-normal}}(t)| \, dt$$

**Algorithm:**
1. Grid search over lengthscales (log-spaced, 20 points)
2. For each lengthscale, fit GP and compute LOO z-scores efficiently
3. Compute $W_1(|z|, \text{half-normal})$ -- lower is better
4. Refine using `scipy.optimize.minimize_scalar` (Brent's method)

**Efficient LOO z-scores:** Rather than refitting the GP $N$ times, LOO
predictions can be computed analytically from the inverse covariance matrix:

$$z_i^{LOO} = \frac{y_i - \mu_{-i}(x_i)}{\sigma_{-i}(x_i)} = \frac{(K^{-1}y)_i}{\sqrt{K^{-1}_{ii}}}$$

This requires only a single Cholesky decomposition: $\mathcal{O}(N^3)$ for
the decomposition, then $\mathcal{O}(N^2)$ for all LOO predictions.

### 2.6 Consensus Building (Per-Experiment Method)

When multiple experiments measure the same reaction, consensus is built
from their individual GP posteriors to identify experiments that deviate
systematically.

**Algorithm:**
1. Define a common energy grid spanning all experiments
2. For each experiment $i$, compute $(\mu_i(E_j), \sigma_i(E_j))$ at grid points
3. Only include grid points within experiment's training data range (no extrapolation)
4. Consensus mean = weighted median of $\mu_i$ with weights $1/\sigma_i^2$
5. Consensus std = derived from weighted scatter

**Discrepancy detection:**
For each experiment, compute the fraction of grid points where:

$$|z_{ij}| = \frac{|\mu_i(E_j) - \mu_{\text{consensus}}(E_j)|}{\sqrt{\sigma_i^2 + \sigma_{\text{consensus}}^2}} > 2$$

An experiment is flagged as discrepant if this fraction exceeds 20%. This
fraction-based threshold avoids DoF problems from correlated GP predictions.

---

## 3. Implementation

### 3.1 Data Partitioning

The EXFOR dataset is partitioned into reaction groups identified by the
tuple $(Z, A, \text{MT})$, where $Z$ is the target atomic number, $A$ is
the target mass number, and MT is the ENDF reaction type code. Each group
represents a distinct nuclear reaction channel (e.g., $^{235}$U(n,f)
corresponds to $Z=92$, $A=235$, $\text{MT}=18$). An independent SVGP model
is fitted to each group, respecting the physical constraint that
cross-section behaviour varies fundamentally between different reactions.

### 3.2 Model Architecture

The implementation utilises GPyTorch, a Gaussian Process library built on
PyTorch that provides GPU acceleration and modern variational inference
algorithms. The model architecture comprises:

| Component | Specification |
|-----------|--------------|
| Mean Function | Constant mean, learned from data |
| Covariance Function | `ScaleKernel(RBFKernel())` with learned lengthscale and variance |
| Inducing Points | 50 points, evenly spaced across the $\log_{10}(E)$ range via `torch.linspace` |
| Variational Distribution | Cholesky-parameterised multivariate Gaussian (`CholeskyVariationalDistribution`) |
| Variational Strategy | `VariationalStrategy` with `learn_inducing_locations=True` |
| Likelihood | Three options: (1) `StudentTLikelihood` with learned $\nu$ (default), (2) `FixedNoiseGaussianLikelihood` for heteroscedastic, (3) `GaussianLikelihood` for legacy |

The `VariationalStrategy` wraps the inducing point framework, allowing the
model to jointly optimise the inducing point locations alongside the kernel
hyperparameters and variational parameters. The Cholesky parameterisation of
the variational distribution ensures positive-definiteness of the
variational covariance matrix.

### 3.3 Training Procedure

Model parameters (kernel hyperparameters, inducing point locations,
variational parameters, and likelihood noise) are optimised by maximising
the Evidence Lower Bound (ELBO) using the Adam optimiser with learning rate
0.05. Training employs early stopping with a patience of 10 epochs,
monitoring the ELBO for convergence (tolerance $10^{-3}$). The maximum
number of epochs is set to 300, though most groups converge within 50--100
epochs.

**Configuration defaults (`SVGPConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_inducing` | 50 | Number of inducing points per group |
| `max_epochs` | 300 | Maximum training epochs |
| `lr` | 0.05 | Adam learning rate |
| `convergence_tol` | $10^{-3}$ | Early stopping tolerance |
| `patience` | 10 | Epochs without improvement before stopping |
| `min_group_size_svgp` | 10 | Minimum points for SVGP (below: MAD fallback) |
| `device` | `'cpu'` | PyTorch device (`'cpu'` or `'cuda'`) |
| `checkpoint_interval` | 1000 | Save checkpoint every $N$ groups |
| `likelihood` | `'student_t'` | Likelihood type: `'student_t'`, `'heteroscedastic'`, or `'gaussian'` |
| `learn_additional_noise` | `False` | For heteroscedastic only: learn residual noise on top of measurement uncertainties |

### 3.4 Likelihood Selection Guide

The choice of likelihood significantly impacts outlier detection performance.
This section provides guidance on selecting the appropriate likelihood for
different data characteristics.

**Student-t Likelihood (Default)**

Recommended for most nuclear data applications. The Student-t likelihood is
robust to outliers in the training data, preventing gross errors from
distorting the GP fit. Key characteristics:

- Automatically learns degrees of freedom $\nu$ from data
- Nuclear data typically yields $\nu \approx 2$--$4$, confirming heavy tails
- Performance: 2.65x max/min uncertainty ratio (vs 1.47x Gaussian baseline)
- No additional data requirements beyond energy and cross-section

**Heteroscedastic Likelihood**

Best choice when calibrated measurement uncertainties are available and
trustworthy. Uses per-point noise variance derived from the `Uncertainty`
column:

- Requires `Uncertainty` column with valid positive values
- Missing uncertainties filled with maximum in group (conservative)
- With `learn_additional_noise=False`: 17.55x max/min ratio (12x improvement!)
- With `learn_additional_noise=True`: 1.28x ratio (additional noise dominates)

The dramatic difference between strict (`learn_additional_noise=False`) and
relaxed modes demonstrates that measurement uncertainties contain significant
information about data quality variation.

**Gaussian Likelihood (Legacy)**

Retained for backwards compatibility. Not recommended for new applications:

- Single learned noise parameter for all observations
- Results in constant-width uncertainty bands regardless of data density
- Performance: 1.47x max/min ratio (baseline)

### 3.5 Per-Experiment Implementation

The per-experiment approach is implemented in `nucml_next/data/experiment_outlier.py`
with supporting modules:

| Module | Class | Purpose |
|--------|-------|---------|
| `experiment_outlier.py` | `ExperimentOutlierDetector` | Main detector, `score_dataframe()` API |
| `experiment_gp.py` | `ExactGPExperiment` | Per-experiment Exact GP fitting |
| `consensus.py` | `ConsensusBuilder` | Multi-experiment consensus |
| `calibration.py` | (functions) | Wasserstein calibration utilities |

**Configuration (`ExperimentOutlierConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `point_z_threshold` | 3.0 | Z-score threshold for point outliers |
| `min_group_size` | 10 | Below this, use MAD fallback |
| `gp_config.min_points_for_gp` | 5 | Minimum points to fit GP to experiment |
| `gp_config.use_wasserstein_calibration` | True | Enable Wasserstein lengthscale optimization |
| `consensus_config.discrepancy_fraction_threshold` | 0.2 | Flag if >20% points discrepant |
| `checkpoint_dir` | None | Directory for checkpoints |

**Processing logic per (Z, A, MT) group:**

```
IF n_total_points < 10:
    → MAD fallback

ELSE IF n_experiments == 1:
    → Fit ExactGP if n >= 5, else MAD
    → Cannot flag experiment_outlier (no comparison)

ELSE (n_experiments >= 2):
    large_exps = [e for e in experiments if len(e) >= 5]
    small_exps = [e for e in experiments if len(e) < 5]

    IF len(large_exps) >= 2:
        → Fit ExactGP to each large experiment
        → Build consensus from posteriors
        → Flag discrepant experiments
        → Evaluate small experiments against consensus

    ELIF len(large_exps) == 1:
        → Use single GP as reference for small experiments

    ELSE:
        → All experiments small - MAD within each
```

### 3.6 Edge Case Handling

The implementation addresses several edge cases that arise in real EXFOR
data:

**Small groups ($n < 10$):** Insufficient data for reliable GP fitting. Fall
back to Median Absolute Deviation (MAD) scoring:

$$z = \frac{|\log_{10}(\sigma) - \text{median}|}{1.4826 \times \text{MAD}}$$

When all values in the group are identical (MAD = 0), a minimum scale of
$10^{-6}$ is used to avoid division by zero.

**Single-point groups:** Cannot determine outlier status without a reference
population. Assign $z\text{-score} = 0$, $\text{gp\_mean} = \log_{10}(\sigma)$,
$\text{gp\_std} = 1.0$.

**Numerical instability:** Cholesky decomposition failures due to
near-singular covariance matrices. Detected via `RuntimeError`,
`ValueError`, or `np.linalg.LinAlgError` and handled by falling back to MAD
scoring. These exceptions are logged at WARNING level.

**Non-convergence:** Groups where ELBO fails to improve within the patience
window. Detected by the early stopping mechanism and handled by returning
the best model state achieved during training.

### 3.6 Computational Considerations

The pipeline processes 9,357 reaction groups comprising 13.4 million data
points. On CPU (Intel Core i7), processing requires approximately 5--6
hours with an average of 2.2 seconds per group. GPU acceleration via CUDA
reduces this to under 1 hour. Checkpointing support enables interruption and
resumption of long-running jobs without loss of progress.

Checkpoints are saved as PyTorch `.pt` files at configurable intervals
(default: every 1,000 groups). Each checkpoint stores:
- The current group index
- Accumulated results (gp_mean, gp_std, z_score per row)
- Processing statistics (SVGP/MAD/single-point counts)

On restart, the pipeline detects existing checkpoints, loads the saved
state, and resumes from the next unprocessed group.

### 3.7 Defensive Engineering

Two defensive patterns were introduced to prevent silent failure modes:

**Fail-fast import validation.** The `SVGPOutlierDetector` constructor
validates that `gpytorch` and `torch` are importable at instantiation time,
raising a clear `ImportError` with installation instructions if either
package is missing. This prevents a failure mode where the lazy import
inside `_fit_svgp()` would raise `ModuleNotFoundError` -- a subclass of
`Exception` -- which was previously caught by the broad exception handler
and silently routed to MAD fallback for every group, producing a 100% SVGP
failure rate with no visible error.

```python
# Constructor-time validation (fail fast)
try:
    import gpytorch
    import torch
except ImportError as e:
    raise ImportError(
        "SVGP outlier detection requires gpytorch and torch. "
        "Install with: pip install gpytorch"
    ) from e
```

**Narrowed exception handling.** The SVGP fitting exception handler was
narrowed from `except Exception` to
`except (RuntimeError, ValueError, np.linalg.LinAlgError)`, catching only
the expected numerical and convergence failures. Unexpected errors
(including import errors, type errors, and programming bugs) now propagate
immediately rather than being silently absorbed. Failed fits are logged at
WARNING level (previously DEBUG), ensuring visibility in standard logging
configurations.

---

## 4. Results and Validation

### 4.1 Processing Summary

| Metric | Value | Percentage |
|--------|------:|----------:|
| Total data points | 13,419,082 | 100% |
| Reaction groups $(Z, A, \text{MT})$ | 9,357 | -- |
| SVGP successfully fitted | 6,776 | 72.4% |
| MAD fallback (small groups) | 1,783 | 19.1% |
| Single-point groups | 797 | 8.5% |
| SVGP numerical failures | 1 | 0.01% |
| Outliers detected ($|z| > 3$) | 271,345 | 2.02% |

### 4.2 Likelihood Performance Comparison

The choice of likelihood significantly impacts uncertainty calibration. The
key metric is the max/min ratio of predicted uncertainty (`gp_std`) across
the energy range -- a well-calibrated model should show larger uncertainty
in sparse regions and smaller uncertainty in dense regions.

| Likelihood | Max/Min Ratio | Improvement | Notes |
|------------|---------------|-------------|-------|
| Gaussian (legacy) | 1.47x | baseline | Constant-width bands; noise dominates signal |
| Student-t | 2.65x | 1.8x | Robust to outliers; learns $\nu \approx 2.07$ |
| Heteroscedastic (relaxed) | 1.28x | 0.9x | `learn_additional_noise=True`; additional noise dominates |
| Heteroscedastic (strict) | 17.55x | 12x | `learn_additional_noise=False`; best calibration |

The heteroscedastic likelihood with strict mode (`learn_additional_noise=False`)
achieves a 12x improvement over baseline, demonstrating that measurement
uncertainties contain substantial information about data quality variation
that should be preserved rather than averaged away.

### 4.3 Outlier Rate Analysis

The observed outlier rate of 2.02% at $|z| > 3$ exceeds the theoretical
expectation of 0.27% for a standard normal distribution. This deviation is
expected and attributable to several factors:

1. **Genuine measurement errors** and transcription mistakes in EXFOR.
2. **Heavier-than-Gaussian tails** in the true error distribution of
   nuclear cross-section measurements.
3. **Model misspecification** where the GP's smoothness assumptions (RBF
   kernel) do not perfectly capture sharp resonance structure, leading to
   elevated z-scores at resonance boundaries.

The elevated rate indicates the method is successfully identifying
problematic data rather than merely flagging statistical noise.

### 4.4 User-Configurable Threshold

The z-score is stored as a column in the output Parquet file, enabling users
to apply custom thresholds at data loading time without rerunning the
computationally expensive SVGP fitting. Recommended thresholds:

| Threshold | Approximate retention | Use case |
|----------:|---------------------:|----------|
| $z = 2.0$ | ~96% | Aggressive cleaning for noise-sensitive models |
| $z = 3.0$ | ~98% | Conservative cleaning (recommended default) |
| $z = 4.0$ | ~99.7% | Minimal cleaning for resonance-focused studies |
| $z = 5.0$ | ~99.9% | Retain nearly all data; flag only extreme outliers |

The `NucmlDataset.outlier_summary()` method provides a tabular breakdown of
outlier counts at multiple thresholds, enabling informed threshold selection:

```python
dataset.outlier_summary()
#    threshold  outliers     pct  retained
# 0        2.0    543200    4.05  12875882
# 1        3.0    271345    2.02  13147737
# 2        4.0     45600    0.34  13373482
# 3        5.0     12300    0.09  13406782
```

---

## 5. Integration

### 5.1 Command-Line Interface

Outlier detection is integrated into the EXFOR ingestion pipeline via
CLI flags on `scripts/ingest_exfor.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--outlier-method` | none | `experiment` (recommended) or `svgp` (legacy) |
| `--z-threshold` | 3.0 | Z-score threshold for point outliers |
| `--svgp-device` | `cpu` | PyTorch device: `cpu` or `cuda` |
| `--svgp-checkpoint-dir` | none | Directory for checkpoint files |
| `--svgp-likelihood` | `student_t` | Likelihood for SVGP method |
| `--test-subset` | off | Use test subset: Uranium (Z=92) + Chlorine (Z=17) only |
| `--z-filter` | none | Comma-separated atomic numbers to include (e.g., `79,92,26`) |

```bash
# Per-experiment GP (recommended)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment

# Per-experiment GP with GPU and checkpointing
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --outlier-method experiment --svgp-device cuda \
    --svgp-checkpoint-dir data/checkpoints/

# Legacy SVGP
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp

# Test subset for fast iteration (Uranium + Chlorine only, ~300K points)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

# Custom element subset (Gold, Uranium, Iron)
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --z-filter 79,92,26

# Full pipeline: test subset + per-experiment + checkpointing
python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db \
    --test-subset --outlier-method experiment \
    --svgp-checkpoint-dir data/checkpoints/
```

### 5.2 Programmatic API

**Per-Experiment GP (Recommended):**

```python
from nucml_next.data import ExperimentOutlierDetector, ExperimentOutlierConfig

config = ExperimentOutlierConfig(point_z_threshold=3.0)
detector = ExperimentOutlierDetector(config)
df_scored = detector.score_dataframe(df)
# df_scored has: experiment_outlier, point_outlier, z_score,
#                calibration_metric, experiment_id, log_E, log_sigma, gp_mean, gp_std

# Filter to non-discrepant experiments
df_clean = df_scored[~df_scored['experiment_outlier']]

# Get processing statistics
stats = detector.get_statistics()
print(f"GP experiments: {stats['gp_experiments']}")
print(f"Discrepant: {stats['discrepant_experiments']}")
```

**Legacy SVGP:**

```python
from nucml_next.data.outlier_detection import SVGPOutlierDetector, SVGPConfig

config = SVGPConfig(device='cuda', likelihood='student_t')
detector = SVGPOutlierDetector(config)
df_scored = detector.score_dataframe(df)
# df_scored has: log_E, log_sigma, gp_mean, gp_std, z_score
```

Both methods implement the same `score_dataframe()` interface:
1. Partition input DataFrame by $(Z, A, \text{MT})$
2. Fit GP models (per-experiment or pooled)
3. Compute z-scores and outlier flags
4. Return augmented DataFrame

Filtering at load time is handled by `DataSelection`:

```python
from nucml_next.data import DataSelection, NucmlDataset

selection = DataSelection(
    z_threshold=3.0,          # Exclude points with z_score > 3
    include_outliers=False,   # Remove (not just flag) outliers
)
dataset = NucmlDataset('data/exfor_processed.parquet', selection=selection)
```

### 5.3 Interactive Threshold Explorer

For exploratory analysis in Jupyter notebooks, the `ThresholdExplorer`
widget provides an interactive interface for inspecting GP fits and
selecting appropriate thresholds:

```python
from nucml_next.visualization.threshold_explorer import ThresholdExplorer

explorer = ThresholdExplorer('data/exfor_processed.parquet')
explorer.show()
```

The widget provides:
- **Cascading dropdowns** for element ($Z$), mass number ($A$), and
  reaction channel (MT), updated dynamically based on available data.
- **Z-score threshold slider** (range 1.0--5.0, step 0.5).
- **GP probability surface** -- a heatmap of $P(\sigma \mid E)$ computed
  from the stored GP predictions, with contour lines at
  $P = 0.01, 0.05, 0.1, 0.25$.
- **Z-score band plot** -- GP mean curve with $\pm 1\sigma$, $\pm 2\sigma$,
  and $\pm z_\text{threshold}$ shaded bands, inlier/outlier scatter, and
  auto-annotated extreme outliers with EXFOR Entry IDs.
- **Marginal z-score histogram** showing the distribution of z-scores for
  the selected group.
- **Statistics panel** with total points, inlier/outlier counts, and
  z-score range.

For MAD fallback or single-point groups, the explorer displays a simplified
view with a diagnostic message indicating why the full probability surface
is unavailable.

---

## 6. Conclusions

NUCML-Next provides two GP-based outlier detection methodologies for nuclear
cross-section data quality assurance:

**Per-Experiment GP (Recommended):** The per-experiment approach fits
independent Exact GPs to each EXFOR experiment, uses measurement uncertainties
for heteroscedastic noise, and builds consensus to identify systematically
discrepant experiments. This method addresses limitations of pooled fitting:
it avoids the masking effect, handles resonance structure without
over-smoothing, and flags entire experiments rather than isolated points.
Wasserstein calibration ensures well-calibrated z-scores.

**SVGP (Legacy):** The original SVGP approach pools experiments per reaction
group. While computationally efficient, it can exhibit elevated outlier rates
on resonance-rich data and cannot identify experiment-level systematics.
Retained for backward compatibility.

Both methods explicitly model cross-sections as functions of energy,
correctly identifying anomalous measurements while preserving legitimate
physical structures such as resonance peaks. Integration into the NUCML-Next
ingestion pipeline ensures that outlier scores are computed once and stored
persistently, enabling efficient downstream filtering without repeated
computation. The configurable threshold mechanism provides flexibility for
different use cases, from aggressive cleaning for baseline model development
to inclusive datasets for systematic uncertainty studies.

---

## References

1. Rasmussen, C.E. and Williams, C.K.I. (2006). *Gaussian Processes for
   Machine Learning*. MIT Press.

2. Hensman, J., Fusi, N. and Lawrence, N.D. (2013). Gaussian Processes for
   Big Data. *Proceedings of UAI*.

3. Hensman, J., Matthews, A.G. de G. and Ghahramani, Z. (2015). Scalable
   Variational Gaussian Process Classification. *Proceedings of AISTATS*.

4. Gardner, J.R. et al. (2018). GPyTorch: Blackbox Matrix-Matrix Gaussian
   Process Inference with GPU Acceleration. *NeurIPS*.

5. Otuka, N. et al. (2014). Towards a More Complete and Accurate
   Experimental Nuclear Reaction Data Library (EXFOR). *Nuclear Data
   Sheets* 120, 272--276.
