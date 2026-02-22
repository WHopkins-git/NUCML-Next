# ML Training Guide

This document outlines the machine-learning training workflow for NUCML-Next,
covering feature selection, data transformations, baseline models, and evaluation.
Each section is skeletal and will be expanded as the project matures.

---

## 1. Feature Tier Selection

Features are organized into additive tiers (A through E). Selecting a set of tiers
builds the full feature vector by concatenating every tier in the selection.

| Tier | Name                   | Count | Features |
|------|------------------------|------:|----------|
| A    | Core + particle vector |    13 | Z, A, N, Energy, out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met |
| B    | Geometric              |    +2 | R_fm, kR |
| C    | Energetics             |    +7 | Mass_Excess_MeV, Binding_Energy_MeV, Binding_Per_Nucleon_MeV, S_1n, S_2n, S_1p, S_2p |
| D    | Topological            |    +9 | Spin, Parity, Isomer_Level, Half_Life_log10_s, Valence_N, Valence_P, P_Factor, Shell_Closure_N, Shell_Closure_P |
| E    | Q-values               |    +8 | Q_alpha, Q_2beta_minus, Q_ep, Q_beta_n, Q_4beta_minus, Q_d_alpha, Q_p_alpha, Q_n_alpha |

```python
from nucml_next.data import DataSelection, NucmlDataset

selection = DataSelection(tiers=['A', 'C', 'D'])
dataset = NucmlDataset('data/exfor_processed.parquet', selection=selection)
```

TBD: guidance on which tier combinations work best for different reaction channels.

---

## 2. Transformation Pipeline

Order of operations applied before any model sees the data:

1. Log-transform cross-section: `log10(sigma + epsilon)`
2. Log-transform energy: `log10(E)`
3. Scale all features (MinMax by default)

```python
from nucml_next.data.selection import TransformationConfig

config = TransformationConfig(
    log_target=True,
    target_epsilon=1e-10,
    log_energy=True,
    scaler_type='minmax',
)
```

TBD: document inverse-transform utilities and custom scaler options.

---

## 3. Baseline Models

### Decision Tree

```python
from nucml_next.baselines import DecisionTreeEvaluator

evaluator = DecisionTreeEvaluator(max_depth=6)
# TBD: full training example
```

### XGBoost

```python
from nucml_next.baselines import XGBoostEvaluator

evaluator = XGBoostEvaluator(n_estimators=500)
# TBD: full training example
```

TBD: hyperparameter search recipes and cross-validation harness.

---

## 4. GNN-Transformer (Planned)

Architecture combining graph neural networks with transformer attention for
nuclear cross-section prediction.

TBD.

---

## 5. Evaluation Methodology

Key principles:

- Holdout by isotope (not random split) to test generalization.
- RMSE in log-space as the primary metric.
- Physics-informed loss options.
- OpenMC reactor benchmark validation (planned).

TBD: detailed metrics, evaluation pipeline, and benchmark result tables.
