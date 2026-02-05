"""
Experiment Management & Phase-Space Holdout
============================================

Unified persistence for NUCML-Next training experiments and a rich
holdout mechanism that supports isotope-level, reaction-level,
energy-window, and EXFOR-entry-level holdout rules with intersection
logic.

Key Components:
    HoldoutConfig        -- Phase-space holdout specification
    ExperimentManager    -- Directory-based experiment persistence
    compute_holdout_metrics -- Evaluate a trained model on holdout data

Directory layout produced by ExperimentManager::

    save/experiments/{YYYYMMDD_HHMMSS}_{model_type}/
        model.joblib          -- sklearn model + feature_columns + params
        scaler_state.pkl      -- fitted TransformationPipeline
        properties.yaml       -- all metadata (human-readable)
        figures/              -- auto-linked .png / .pdf plots

Example::

    from nucml_next.experiment import (
        HoldoutConfig, ExperimentManager, compute_holdout_metrics,
    )

    # Define holdout: U-235 capture in resolved resonance region
    holdout = HoldoutConfig(rules=[
        {'Z': 92, 'A': 235, 'MT': 102, 'energy_range': (1e-3, 1.0)},
    ])

    # After training ...
    mgr = ExperimentManager()
    exp_dir = mgr.save_experiment(model, 'xgboost', holdout_config=holdout)

    # Reload later
    env = ExperimentManager.load_experiment(exp_dir)
    preds = env['model'].predict(new_df)
"""

from __future__ import annotations

import datetime
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# HoldoutConfig
# ---------------------------------------------------------------------------

@dataclass
class HoldoutConfig:
    """
    Phase-space holdout specification with intersection logic.

    Each *rule* is a dictionary whose keys are intersected (AND).
    Multiple rules are unioned (OR): a row is held out if it matches
    **any** rule.

    Supported rule keys (all optional within a rule):

    +-----------------+-------------------------------+---------------------------+
    | Key             | Type                          | Meaning                   |
    +=================+===============================+===========================+
    | ``Z``           | int                           | Atomic number             |
    +-----------------+-------------------------------+---------------------------+
    | ``A``           | int                           | Mass number               |
    +-----------------+-------------------------------+---------------------------+
    | ``MT``          | int or List[int]              | Reaction channel(s)       |
    +-----------------+-------------------------------+---------------------------+
    | ``energy_range``| Tuple[float, float]           | (E_min, E_max) in eV      |
    +-----------------+-------------------------------+---------------------------+
    | ``xs_range``    | Tuple[float, float]           | (XS_min, XS_max) in barns |
    +-----------------+-------------------------------+---------------------------+
    | ``Entry``       | str or List[str]              | EXFOR Entry ID(s)         |
    +-----------------+-------------------------------+---------------------------+

    Examples::

        # Classic isotope extrapolation
        HoldoutConfig(rules=[{'Z': 92, 'A': 235}])

        # Resonance gap interpolation
        HoldoutConfig(rules=[
            {'Z': 92, 'A': 235, 'MT': 102, 'energy_range': (1e-3, 1.0)},
        ])

        # Multiple rules (union)
        HoldoutConfig(rules=[
            {'Z': 92, 'A': 235},   # all U-235
            {'MT': 18},             # all fission globally
        ])
    """

    rules: List[Dict[str, Any]] = field(default_factory=list)

    # ---- mask construction -------------------------------------------------

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a boolean Series that is True for rows matching **any** rule.

        Within each rule the criteria are intersected (AND).
        Across rules the masks are unioned (OR).
        """
        if not self.rules:
            return pd.Series(False, index=df.index)

        combined = pd.Series(False, index=df.index)
        for rule in self.rules:
            combined |= self._eval_single_rule(rule, df)
        return combined

    def split(
        self, df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split *df* into (training, holdout) DataFrames.

        Returns
        -------
        df_train : pd.DataFrame
            Rows that do NOT match any holdout rule.
        df_holdout : pd.DataFrame
            Rows that match at least one holdout rule.
        """
        mask = self.build_mask(df)
        return df.loc[~mask].copy(), df.loc[mask].copy()

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialise rules to a list of plain dicts (YAML-safe)."""
        out: List[Dict[str, Any]] = []
        for rule in self.rules:
            d: Dict[str, Any] = {}
            for key in ('Z', 'A', 'MT', 'Entry'):
                if key in rule:
                    d[key] = rule[key]
            if 'energy_range' in rule:
                lo, hi = rule['energy_range']
                d['energy_range'] = [float(lo), float(hi)]
            if 'xs_range' in rule:
                lo, hi = rule['xs_range']
                d['xs_range'] = [float(lo), float(hi)]
            out.append(d)
        return out

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> 'HoldoutConfig':
        """Deserialise from the list-of-dicts format written by *to_dict*."""
        rules: List[Dict[str, Any]] = []
        for d in data:
            rule: Dict[str, Any] = {}
            for key in ('Z', 'A', 'MT', 'Entry'):
                if key in d:
                    rule[key] = d[key]
            if 'energy_range' in d:
                er = d['energy_range']
                rule['energy_range'] = (float(er[0]), float(er[1]))
            if 'xs_range' in d:
                xr = d['xs_range']
                rule['xs_range'] = (float(xr[0]), float(xr[1]))
            rules.append(rule)
        return cls(rules=rules)

    @classmethod
    def from_legacy(
        cls, holdout_isotopes: List[Tuple[int, int]],
    ) -> 'HoldoutConfig':
        """
        Bridge from the old ``holdout_isotopes=[(Z, A), ...]`` API.

        Each (Z, A) pair becomes a single isotope-level rule.
        """
        rules = [{'Z': z, 'A': a} for z, a in holdout_isotopes]
        return cls(rules=rules)

    # ---- display -----------------------------------------------------------

    def __repr__(self) -> str:
        if not self.rules:
            return 'HoldoutConfig(rules=[])'
        lines = ['HoldoutConfig(']
        for i, rule in enumerate(self.rules):
            parts: List[str] = []
            if 'Z' in rule and 'A' in rule:
                parts.append(f'Z={rule["Z"]}, A={rule["A"]}')
            elif 'Z' in rule:
                parts.append(f'Z={rule["Z"]}')
            elif 'A' in rule:
                parts.append(f'A={rule["A"]}')
            if 'MT' in rule:
                parts.append(f'MT={rule["MT"]}')
            if 'energy_range' in rule:
                lo, hi = rule['energy_range']
                parts.append(f'E=[{lo:.2e}, {hi:.2e}]')
            if 'xs_range' in rule:
                lo, hi = rule['xs_range']
                parts.append(f'XS=[{lo:.2e}, {hi:.2e}]')
            if 'Entry' in rule:
                parts.append(f'Entry={rule["Entry"]}')
            lines.append(f'  Rule {i+1}: {", ".join(parts)}')
        lines.append(')')
        return '\n'.join(lines)

    def __bool__(self) -> bool:
        """Truthy if there is at least one rule."""
        return bool(self.rules)

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _eval_single_rule(
        rule: Dict[str, Any], df: pd.DataFrame,
    ) -> pd.Series:
        """Evaluate one rule: all criteria are AND-ed."""
        mask = pd.Series(True, index=df.index)

        if 'Z' in rule:
            mask &= df['Z'] == rule['Z']
        if 'A' in rule:
            mask &= df['A'] == rule['A']

        if 'MT' in rule:
            mt_val = rule['MT']
            if isinstance(mt_val, list):
                mask &= df['MT'].isin(mt_val)
            else:
                mask &= df['MT'] == mt_val

        if 'energy_range' in rule:
            lo, hi = rule['energy_range']
            mask &= (df['Energy'] >= lo) & (df['Energy'] <= hi)

        if 'xs_range' in rule:
            lo, hi = rule['xs_range']
            mask &= (df['CrossSection'] >= lo) & (df['CrossSection'] <= hi)

        if 'Entry' in rule:
            entries = rule['Entry']
            if isinstance(entries, list):
                mask &= df['Entry'].isin(entries)
            else:
                mask &= df['Entry'] == entries

        return mask


# ---------------------------------------------------------------------------
# compute_holdout_metrics
# ---------------------------------------------------------------------------

def compute_holdout_metrics(
    model,
    df_holdout: pd.DataFrame,
    pipeline=None,
) -> Dict[str, Any]:
    """
    Evaluate a trained evaluator on a holdout DataFrame.

    Parameters
    ----------
    model : XGBoostEvaluator or DecisionTreeEvaluator
        Trained evaluator with ``model.predict(df)`` available.
    df_holdout : pd.DataFrame
        Holdout data **already projected to tabular format** (must contain
        the same feature columns the model was trained on, plus
        ``CrossSection`` for ground truth).
    pipeline : TransformationPipeline, optional
        If None, uses ``model.pipeline``.

    Returns
    -------
    dict
        Metrics in both log-space and physical barns::

            {
                'holdout_n': int,
                'holdout_mse_log': float,
                'holdout_mae_log': float,
                'holdout_r2_log': float,
                'holdout_mse_barns': float,
                'holdout_mae_barns': float,
                'holdout_r2_barns': float,
                'holdout_medae_barns': float,
            }
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    pipe = pipeline or model.pipeline
    if pipe is None:
        raise RuntimeError(
            'No pipeline available.  Pass pipeline= or ensure model.pipeline is set.'
        )

    # Feature columns the model expects
    feature_cols = model.feature_columns
    exclude = {'CrossSection', 'Uncertainty', 'Energy_Uncertainty', 'Entry', 'MT'}
    present_features = [c for c in feature_cols if c in df_holdout.columns]
    if len(present_features) < len(feature_cols):
        missing = set(feature_cols) - set(present_features)
        raise ValueError(
            f'Holdout DataFrame is missing feature columns: {missing}'
        )

    X_features = df_holdout[feature_cols]
    y_true_barns = df_holdout['CrossSection'].values
    energy = (
        df_holdout['Energy'] if 'Energy' in df_holdout.columns else None
    )

    # --- transform features (same as model.predict internals) ---------------
    X_transformed = pipe.transform(X_features, energy)
    X_arr = X_transformed[feature_cols].values
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e10, neginf=-1e10)

    # --- predict in transformed space ---------------------------------------
    y_pred_log = model.model.predict(X_arr)

    # --- log-space ground truth ---------------------------------------------
    y_true_log = pipe.transform_target(
        pd.Series(y_true_barns)
    ).values

    # --- inverse-transform predictions to barns -----------------------------
    y_pred_barns = pipe.inverse_transform_target(
        pd.Series(y_pred_log)
    ).values

    # --- compute metrics in both spaces -------------------------------------
    # Filter out any NaN / inf that may arise from edge cases
    valid = (
        np.isfinite(y_true_log) & np.isfinite(y_pred_log)
        & np.isfinite(y_true_barns) & np.isfinite(y_pred_barns)
    )
    y_true_log_v = y_true_log[valid]
    y_pred_log_v = y_pred_log[valid]
    y_true_barns_v = y_true_barns[valid]
    y_pred_barns_v = y_pred_barns[valid]

    n = int(valid.sum())

    metrics: Dict[str, Any] = {'holdout_n': n}

    if n > 1:
        metrics['holdout_mse_log'] = float(mean_squared_error(y_true_log_v, y_pred_log_v))
        metrics['holdout_mae_log'] = float(mean_absolute_error(y_true_log_v, y_pred_log_v))
        metrics['holdout_r2_log'] = float(r2_score(y_true_log_v, y_pred_log_v))

        metrics['holdout_mse_barns'] = float(mean_squared_error(y_true_barns_v, y_pred_barns_v))
        metrics['holdout_mae_barns'] = float(mean_absolute_error(y_true_barns_v, y_pred_barns_v))
        metrics['holdout_r2_barns'] = float(r2_score(y_true_barns_v, y_pred_barns_v))
        metrics['holdout_medae_barns'] = float(median_absolute_error(y_true_barns_v, y_pred_barns_v))
    else:
        for k in (
            'holdout_mse_log', 'holdout_mae_log', 'holdout_r2_log',
            'holdout_mse_barns', 'holdout_mae_barns', 'holdout_r2_barns',
            'holdout_medae_barns',
        ):
            metrics[k] = float('nan')

    return metrics


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """
    Unified experiment persistence.

    Creates a timestamped directory for each training run and stores:

    - ``model.joblib``     -- sklearn model + feature columns + params
    - ``scaler_state.pkl`` -- fitted :class:`TransformationPipeline`
    - ``properties.yaml``  -- all metadata (human-readable, git-diffable)
    - ``figures/``         -- auto-linked by :class:`IsotopePlotter`

    Parameters
    ----------
    base_dir : str or Path
        Root directory for all experiments.
        Default: ``save/experiments``
    """

    def __init__(self, base_dir: Union[str, Path] = 'save/experiments'):
        self.base_dir = Path(base_dir)

    # ---- save --------------------------------------------------------------

    def save_experiment(
        self,
        model,
        model_type: str,
        *,
        selection=None,
        holdout_config: Optional[HoldoutConfig] = None,
        holdout_metrics: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Persist a trained model, scaler, and metadata.

        Parameters
        ----------
        model
            Trained evaluator (XGBoostEvaluator or DecisionTreeEvaluator).
        model_type : str
            Short identifier, e.g. ``'xgboost'`` or ``'decision_tree'``.
        selection : DataSelection, optional
            The DataSelection used for training (recorded in YAML).
        holdout_config : HoldoutConfig, optional
            Holdout specification (recorded in YAML).
        holdout_metrics : dict, optional
            Metrics computed on the holdout set.
        extra_metadata : dict, optional
            Arbitrary extra metadata to store.

        Returns
        -------
        Path
            Absolute path to the experiment directory.
        """
        if not getattr(model, 'is_trained', False):
            raise RuntimeError('Cannot save an untrained model.')

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = self.base_dir / f'{ts}_{model_type}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / 'figures').mkdir(exist_ok=True)

        # 1. Model weights + feature columns + params
        model_bundle = {
            'model': model.model,
            'feature_columns': model.feature_columns,
            'params': getattr(model, 'params', None) or self._extract_params(model),
            'metrics': model.metrics,
            'model_type': model_type,
        }
        joblib.dump(model_bundle, exp_dir / 'model.joblib')

        # 2. Scaler / pipeline state
        if model.pipeline is not None:
            model.pipeline.save(str(exp_dir / 'scaler_state.pkl'))

        # 3. Properties YAML
        props = self._build_properties(
            model, model_type, ts, selection,
            holdout_config, holdout_metrics, extra_metadata,
        )
        with open(exp_dir / 'properties.yaml', 'w') as f:
            yaml.dump(props, f, default_flow_style=False, sort_keys=False)

        print(f'[OK] Experiment saved to {exp_dir}')
        return exp_dir

    # ---- load --------------------------------------------------------------

    @staticmethod
    def load_experiment(exp_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Reconstruct the full environment from a saved experiment.

        The returned evaluator's ``predict(raw_df)`` works immediately
        without manual pre-processing.

        Parameters
        ----------
        exp_path : str or Path
            Path to the experiment directory.

        Returns
        -------
        dict
            Keys: ``'model'``, ``'pipeline'``, ``'properties'``,
            ``'holdout_config'``.
        """
        from nucml_next.data.transformations import TransformationPipeline

        exp_path = Path(exp_path)

        # 1. Properties
        with open(exp_path / 'properties.yaml') as f:
            props = yaml.safe_load(f)

        # 2. Pipeline / scaler
        scaler_path = exp_path / 'scaler_state.pkl'
        pipeline = (
            TransformationPipeline.load(str(scaler_path))
            if scaler_path.exists() else None
        )

        # 3. Model bundle
        bundle = joblib.load(exp_path / 'model.joblib')
        model_type = bundle.get('model_type', props.get('model_type', 'unknown'))

        # Instantiate the correct evaluator class
        evaluator = ExperimentManager._make_evaluator(
            model_type, bundle, pipeline,
        )

        # 4. Holdout config
        holdout_cfg = None
        if 'holdout' in props and props['holdout']:
            holdout_cfg = HoldoutConfig.from_dict(props['holdout'])

        print(f'[OK] Experiment loaded from {exp_path}')
        return {
            'model': evaluator,
            'pipeline': pipeline,
            'properties': props,
            'holdout_config': holdout_cfg,
        }

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _extract_params(model) -> Dict[str, Any]:
        """Extract hyperparams from evaluator attributes (DT fallback)."""
        params: Dict[str, Any] = {}
        for attr in (
            'max_depth', 'min_samples_leaf', 'min_samples_split',
            'min_impurity_decrease', 'max_features', 'random_state',
            'n_estimators', 'learning_rate', 'subsample',
            'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
            'min_child_weight',
        ):
            if hasattr(model, attr):
                params[attr] = getattr(model, attr)
        return params

    @staticmethod
    def _make_evaluator(
        model_type: str,
        bundle: Dict[str, Any],
        pipeline,
    ):
        """Instantiate the correct evaluator and attach loaded state."""
        if model_type in ('xgboost', 'XGBoost'):
            from nucml_next.baselines import XGBoostEvaluator
            ev = XGBoostEvaluator.__new__(XGBoostEvaluator)
            ev.params = bundle.get('params', {})
            # Restore individual param attrs expected by XGBoostEvaluator
            for k, v in (ev.params or {}).items():
                setattr(ev, k, v)
        elif model_type in ('decision_tree', 'DecisionTree'):
            from nucml_next.baselines import DecisionTreeEvaluator
            ev = DecisionTreeEvaluator.__new__(DecisionTreeEvaluator)
            params = bundle.get('params', {})
            for k, v in params.items():
                setattr(ev, k, v)
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'")

        ev.model = bundle['model']
        ev.feature_columns = bundle['feature_columns']
        ev.metrics = bundle.get('metrics', {})
        ev.pipeline = pipeline
        ev.is_trained = True
        return ev

    def _build_properties(
        self, model, model_type, timestamp, selection,
        holdout_config, holdout_metrics, extra_metadata,
    ) -> Dict[str, Any]:
        """Assemble the metadata dictionary for properties.yaml."""
        props: Dict[str, Any] = {
            'model_type': model_type,
            'timestamp': timestamp,
        }

        # Model hyperparams
        params = getattr(model, 'params', None) or self._extract_params(model)
        # Convert numpy types to Python builtins for YAML
        props['model_params'] = {
            k: _yaml_safe(v) for k, v in params.items()
        }

        # Transformation config
        if model.pipeline is not None:
            cfg = model.pipeline.config
            props['transformation'] = {
                'log_target': cfg.log_target,
                'target_epsilon': float(cfg.target_epsilon),
                'log_base': _yaml_safe(cfg.log_base),
                'log_energy': cfg.log_energy,
                'energy_log_base': _yaml_safe(cfg.energy_log_base),
                'scaler_type': cfg.scaler_type,
            }

        # Data selection
        if selection is not None:
            props['selection'] = {
                'projectile': selection.projectile,
                'energy_min': float(selection.energy_min),
                'energy_max': float(selection.energy_max),
                'mt_mode': selection.mt_mode,
                'tiers': list(selection.tiers),
            }
            if selection.z_threshold is not None:
                props['selection']['z_threshold'] = float(selection.z_threshold)

        # Holdout
        if holdout_config is not None and holdout_config.rules:
            props['holdout'] = holdout_config.to_dict()

        # Holdout metrics
        if holdout_metrics is not None:
            props['holdout_metrics'] = {
                k: _yaml_safe(v) for k, v in holdout_metrics.items()
            }

        # Training metrics
        if model.metrics:
            props['training_metrics'] = {
                k: _yaml_safe(v) for k, v in model.metrics.items()
            }

        # Feature columns
        props['feature_columns'] = list(model.feature_columns)

        # Extra
        if extra_metadata:
            props['extra'] = extra_metadata

        return props


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaml_safe(val):
    """Convert numpy / non-standard types to YAML-serialisable builtins."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val
