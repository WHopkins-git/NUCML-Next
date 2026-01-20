"""
Baseline Models Module
======================

Legacy machine learning models that demonstrate the limitations of
classical approaches for nuclear data evaluation.

Key Components:
    XGBoostEvaluator: Gradient boosting baseline
    DecisionTreeEvaluator: Decision tree baseline (shows "staircase effect")

Pedagogical Goal:
    These models reveal why physics-informed deep learning is necessary:
    - Jagged predictions in resonance regions
    - Poor extrapolation beyond training data
    - No awareness of nuclear physics constraints
"""

from nucml_next.baselines.xgboost_evaluator import XGBoostEvaluator
from nucml_next.baselines.decision_tree_evaluator import DecisionTreeEvaluator

__all__ = [
    "XGBoostEvaluator",
    "DecisionTreeEvaluator",
]
