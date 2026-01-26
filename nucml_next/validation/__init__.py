"""
OpenMC Validation Module
========================

Integration with OpenMC for reactor physics validation.

Key Components:
    OpenMCValidator: Main interface to OpenMC Python API
    SensitivityAnalyzer: Computes sensitivity coefficients S_k,σ
    ReactorBenchmark: Runs standard reactor benchmarks (PWR, BWR, etc.)

Workflow:
    1. Train model on cross-section data
    2. Export predictions to OpenMC-compatible format (ENDF)
    3. Run k-eigenvalue calculation
    4. Extract sensitivity coefficients: ∂k/∂σ
    5. Retrain model with sensitivity-weighted loss

This solves:
    The Validation Paradox: Prioritize accuracy on safety-critical reactions
    rather than minimizing geometric MSE uniformly across all data points.
"""

from nucml_next.validation.openmc_validator import OpenMCValidator
from nucml_next.validation.sensitivity_analyzer import SensitivityAnalyzer
from nucml_next.validation.reactor_benchmark import ReactorBenchmark

__all__ = [
    "OpenMCValidator",
    "SensitivityAnalyzer",
    "ReactorBenchmark",
]
