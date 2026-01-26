"""
Reactor Benchmark Module
========================

Standard reactor physics benchmarks for validation.

Benchmarks:
    - PWR Pin Cell
    - BWR Assembly
    - Fast Reactor Core
    - CRIT experiments

These provide reference k_eff values for model validation.
"""

from typing import Dict, List, Tuple
import numpy as np


class ReactorBenchmark:
    """
    Manages reactor physics benchmarks for validation.

    Educational Purpose:
        Students validate their models against real reactor configurations,
        not just geometric MSE on test data.
    """

    BENCHMARKS = {
        'PWR_PIN': {
            'description': 'Pressurized Water Reactor - Single Pin Cell',
            'k_eff_reference': 1.18647,
            'k_eff_uncertainty': 0.00012,
            'temperature': 900.0,  # K
            'fuel_enrichment': 0.04,  # 4% U-235
            'moderator': 'H2O',
            'critical_isotopes': [(92, 235), (92, 238), (1, 1), (8, 16)],
        },
        'BWR_ASSEMBLY': {
            'description': 'Boiling Water Reactor - 8x8 Assembly',
            'k_eff_reference': 1.17234,
            'k_eff_uncertainty': 0.00015,
            'temperature': 850.0,
            'fuel_enrichment': 0.035,
            'moderator': 'H2O (boiling)',
            'critical_isotopes': [(92, 235), (92, 238), (1, 1)],
        },
        'FAST_CORE': {
            'description': 'Sodium-Cooled Fast Reactor',
            'k_eff_reference': 1.03567,
            'k_eff_uncertainty': 0.00020,
            'temperature': 800.0,
            'fuel_enrichment': 0.20,  # 20% Pu-239
            'moderator': 'None (fast spectrum)',
            'critical_isotopes': [(92, 238), (94, 239), (94, 240)],
        },
        'JEZEBEL': {
            'description': 'Pu-239 Critical Assembly (LANL)',
            'k_eff_reference': 1.00000,  # Critical by definition
            'k_eff_uncertainty': 0.00010,
            'temperature': 300.0,
            'fuel_enrichment': 1.00,  # Pure Pu-239
            'moderator': 'None',
            'critical_isotopes': [(94, 239)],
        },
    }

    def __init__(self, benchmark_name: str = 'PWR_PIN'):
        """
        Initialize benchmark.

        Args:
            benchmark_name: Name of benchmark to use
        """
        if benchmark_name not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. "
                           f"Available: {list(self.BENCHMARKS.keys())}")

        self.benchmark_name = benchmark_name
        self.benchmark = self.BENCHMARKS[benchmark_name]

    def get_reference_k_eff(self) -> Tuple[float, float]:
        """
        Get reference k_eff value.

        Returns:
            (k_eff_mean, k_eff_uncertainty)
        """
        return (
            self.benchmark['k_eff_reference'],
            self.benchmark['k_eff_uncertainty']
        )

    def get_critical_isotopes(self) -> List[Tuple[int, int]]:
        """
        Get list of critical isotopes for this benchmark.

        Returns:
            List of (Z, A) tuples
        """
        return self.benchmark['critical_isotopes']

    def validate_prediction(
        self,
        k_eff_predicted: float,
        k_eff_uncertainty: float = 0.0,
    ) -> Dict[str, any]:
        """
        Validate predicted k_eff against benchmark.

        Args:
            k_eff_predicted: Predicted k_eff value
            k_eff_uncertainty: Uncertainty in prediction

        Returns:
            Validation results
        """
        k_ref, k_ref_unc = self.get_reference_k_eff()

        # Absolute error
        error = k_eff_predicted - k_ref

        # Error in pcm (percent-mille)
        error_pcm = error * 1e5 / k_ref

        # Combined uncertainty
        total_unc = np.sqrt(k_ref_unc**2 + k_eff_uncertainty**2)

        # Statistical significance (sigma)
        if total_unc > 0:
            sigma = abs(error) / total_unc
        else:
            sigma = float('inf') if error != 0 else 0.0

        # Pass/fail criteria
        # Typically: |error| < 300 pcm for reactor design
        pass_criteria = abs(error_pcm) < 300

        return {
            'benchmark': self.benchmark_name,
            'k_eff_predicted': k_eff_predicted,
            'k_eff_reference': k_ref,
            'error': error,
            'error_pcm': error_pcm,
            'uncertainty': total_unc,
            'sigma': sigma,
            'pass': pass_criteria,
        }

    def print_benchmark_info(self):
        """Print benchmark information."""
        print("\n" + "="*70)
        print(f"Reactor Benchmark: {self.benchmark_name}")
        print("="*70)
        print(f"Description:     {self.benchmark['description']}")
        print(f"k_eff (ref):     {self.benchmark['k_eff_reference']:.5f} ± "
              f"{self.benchmark['k_eff_uncertainty']:.5f}")
        print(f"Temperature:     {self.benchmark['temperature']:.1f} K")
        print(f"Fuel:            {self.benchmark['fuel_enrichment']*100:.1f}% enrichment")
        print(f"Moderator:       {self.benchmark['moderator']}")
        print(f"Critical Isotopes: {self.benchmark['critical_isotopes']}")
        print("="*70 + "\n")

    def print_validation_results(self, results: Dict[str, any]):
        """
        Print validation results.

        Args:
            results: Results from validate_prediction()
        """
        print("\n" + "="*70)
        print(f"Validation Results: {results['benchmark']}")
        print("="*70)
        print(f"  k_eff (predicted):   {results['k_eff_predicted']:.5f}")
        print(f"  k_eff (reference):   {results['k_eff_reference']:.5f}")
        print(f"  Error:               {results['error']:+.5f} ({results['error_pcm']:+.1f} pcm)")
        print(f"  Uncertainty:         ±{results['uncertainty']:.5f}")
        print(f"  Statistical Sig:     {results['sigma']:.1f} σ")
        print(f"  Status:              {'PASS ✓' if results['pass'] else 'FAIL ✗'}")
        print("="*70)

        # Interpretation
        if abs(results['error_pcm']) < 50:
            print("🌟 Excellent agreement with benchmark!")
        elif abs(results['error_pcm']) < 100:
            print("✓ Good agreement - acceptable for most applications")
        elif abs(results['error_pcm']) < 300:
            print("⚠️  Acceptable but needs improvement for safety-critical use")
        else:
            print("❌ Unacceptable error - model needs significant improvement")

        print()

    @staticmethod
    def list_available_benchmarks():
        """List all available benchmarks."""
        print("\nAvailable Reactor Benchmarks:")
        print("="*70)
        for name, info in ReactorBenchmark.BENCHMARKS.items():
            print(f"\n{name}:")
            print(f"  {info['description']}")
            print(f"  k_eff = {info['k_eff_reference']:.5f} ± {info['k_eff_uncertainty']:.5f}")
        print("="*70 + "\n")
