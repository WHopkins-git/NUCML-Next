"""
Sensitivity Analyzer
====================

Computes sensitivity coefficients S_k,σ = ∂k/∂σ

These coefficients tell us how much k_eff changes when a cross section changes.

High sensitivity → errors in this XS strongly affect reactor safety!
Low sensitivity → errors are less critical

This is the KEY to solving the validation paradox.
"""

from typing import Dict, Tuple, List
import numpy as np


class SensitivityAnalyzer:
    """
    Computes and analyzes cross-section sensitivity coefficients.

    Educational Value:
        Students learn which reactions matter most for reactor safety:
        - U-235 fission: HIGH sensitivity
        - U-238 capture: MEDIUM sensitivity
        - Trace actinide captures: LOW sensitivity
    """

    def __init__(self, reactor_type: str = 'PWR'):
        """
        Initialize sensitivity analyzer.

        Args:
            reactor_type: Reactor type ('PWR', 'BWR', 'Fast')
        """
        self.reactor_type = reactor_type

        # Typical sensitivity coefficients (from literature/benchmarks)
        # Format: (Z, A, MT) -> sensitivity
        self.reference_sensitivities = self._get_reference_sensitivities()

    def _get_reference_sensitivities(self) -> Dict[Tuple[int, int, int], float]:
        """
        Get reference sensitivity coefficients for common reactions.

        These are based on typical PWR/BWR/Fast reactor benchmarks.

        Returns:
            Dictionary mapping (Z, A, MT) -> sensitivity coefficient
        """
        if self.reactor_type == 'PWR':
            # PWR typical sensitivities
            sensitivities = {
                # U-235 (highly sensitive!)
                (92, 235, 18): 0.500,   # (n,f) fission - VERY HIGH
                (92, 235, 102): -0.150, # (n,γ) capture - HIGH (negative)
                (92, 235, 2): 0.020,    # Elastic - LOW

                # U-238 (fertile, moderately sensitive)
                (92, 238, 102): -0.080, # (n,γ) capture - MEDIUM
                (92, 238, 18): 0.015,   # Fast fission - LOW
                (92, 238, 2): -0.010,   # Elastic - LOW

                # Pu-239 (if present, buildup product)
                (94, 239, 18): 0.300,   # Fission - HIGH
                (94, 239, 102): -0.100, # Capture - MEDIUM

                # H-1 (moderator - critical!)
                (1, 1, 2): 0.180,       # Elastic scattering - HIGH
                (1, 1, 102): -0.050,    # Capture - MEDIUM

                # O-16 (coolant)
                (8, 16, 2): 0.005,      # Elastic - LOW
            }

        elif self.reactor_type == 'BWR':
            # Similar to PWR but slightly different spectrum
            sensitivities = {
                (92, 235, 18): 0.520,
                (92, 235, 102): -0.160,
                (92, 238, 102): -0.090,
                (1, 1, 2): 0.200,
            }

        elif self.reactor_type == 'Fast':
            # Fast reactor: harder spectrum, different sensitivities
            sensitivities = {
                (92, 235, 18): 0.400,
                (92, 238, 18): 0.250,   # U-238 fission more important in fast spectrum!
                (94, 239, 18): 0.450,
                (92, 238, 2): 0.150,    # Inelastic scattering important
            }

        else:
            # Default to PWR
            sensitivities = self._get_reference_sensitivities_for_reactor('PWR')

        return sensitivities

    def _get_reference_sensitivities_for_reactor(self, reactor_type: str) -> Dict:
        """Helper to get sensitivities for specific reactor type."""
        old_type = self.reactor_type
        self.reactor_type = reactor_type
        result = self._get_reference_sensitivities()
        self.reactor_type = old_type
        return result

    def compute_sensitivity(
        self,
        isotope: Tuple[int, int],
        reaction_mt: int,
        energy_idx: int = 0,
    ) -> float:
        """
        Compute sensitivity coefficient for a specific reaction.

        Args:
            isotope: (Z, A) tuple
            reaction_mt: MT code
            energy_idx: Energy index (for energy-dependent sensitivities)

        Returns:
            Sensitivity coefficient S_k,σ
        """
        key = (isotope[0], isotope[1], reaction_mt)

        # Look up reference sensitivity
        sensitivity = self.reference_sensitivities.get(key, 0.0)

        # In a full implementation, this would:
        # 1. Run OpenMC with perturbed cross section
        # 2. Compute Δk / Δσ numerically
        # 3. Return the sensitivity

        return sensitivity

    def get_top_sensitive_reactions(self, top_n: int = 10) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Get the most sensitive reactions (by absolute value).

        Args:
            top_n: Number of top reactions to return

        Returns:
            List of ((Z, A, MT), sensitivity) sorted by |sensitivity|

        Educational Use:
            Show students which cross sections matter most!
        """
        # Sort by absolute sensitivity
        sorted_sensitivities = sorted(
            self.reference_sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_sensitivities[:top_n]

    def get_sensitivity_weights(
        self,
        cross_section_data: Dict[Tuple[int, int, int], np.ndarray],
        normalize: bool = True,
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Get sensitivity weights for all cross sections.

        Args:
            cross_section_data: Cross section data dict
            normalize: Normalize weights to [0, 1]

        Returns:
            Weight arrays for each reaction

        Use Case:
            Feed these weights to SensitivityWeightedLoss during training!
        """
        weights = {}

        for key in cross_section_data:
            Z, A, MT = key
            sensitivity = self.compute_sensitivity((Z, A), MT)

            # Create weight array (constant across energy for now)
            num_points = len(cross_section_data[key])
            weight_array = np.full(num_points, abs(sensitivity))

            weights[key] = weight_array

        # Normalize if requested
        if normalize and weights:
            max_weight = max(w.max() for w in weights.values())
            if max_weight > 0:
                weights = {k: v / max_weight for k, v in weights.items()}

        return weights

    def print_sensitivity_report(self):
        """
        Print formatted sensitivity report.

        Educational output showing which reactions matter!
        """
        print("\n" + "="*80)
        print(f"Sensitivity Analysis Report - {self.reactor_type}")
        print("="*80)
        print(f"{'Isotope':<12} {'Reaction':<15} {'MT':<6} {'Sensitivity':>12} {'Impact':<12}")
        print("-"*80)

        # Get top reactions
        top_reactions = self.get_top_sensitive_reactions(top_n=15)

        for (Z, A, MT), sensitivity in top_reactions:
            # Determine isotope name
            isotope_names = {
                (92, 235): 'U-235',
                (92, 238): 'U-238',
                (94, 239): 'Pu-239',
                (94, 240): 'Pu-240',
                (1, 1): 'H-1',
                (8, 16): 'O-16',
            }
            isotope = isotope_names.get((Z, A), f'{Z}-{A}')

            # Determine reaction name
            reaction_names = {
                2: 'Elastic',
                18: 'Fission',
                102: 'Capture',
                16: '(n,2n)',
            }
            reaction = reaction_names.get(MT, f'MT={MT}')

            # Classify impact
            abs_sens = abs(sensitivity)
            if abs_sens > 0.3:
                impact = 'CRITICAL'
            elif abs_sens > 0.1:
                impact = 'HIGH'
            elif abs_sens > 0.05:
                impact = 'MEDIUM'
            elif abs_sens > 0.01:
                impact = 'LOW'
            else:
                impact = 'NEGLIGIBLE'

            print(f"{isotope:<12} {reaction:<15} {MT:<6} {sensitivity:>+12.4f} {impact:<12}")

        print("="*80)
        print("\n💡 Educational Insights:")
        print("   • Positive sensitivity: Increasing σ increases k_eff (e.g., fission)")
        print("   • Negative sensitivity: Increasing σ decreases k_eff (e.g., capture)")
        print("   • Magnitude matters: Focus ML accuracy on high-sensitivity reactions!")
        print("\n🎯 Training Strategy:")
        print("   Use SensitivityWeightedLoss to prioritize these critical reactions.\n")

    def compute_uncertainty_propagation(
        self,
        cross_section_uncertainties: Dict[Tuple[int, int, int], float],
    ) -> float:
        """
        Propagate cross-section uncertainties to k_eff uncertainty.

        σ²_k ≈ Σ_i S²_i · σ²_σi

        Args:
            cross_section_uncertainties: Dict of relative uncertainties (%)

        Returns:
            k_eff relative uncertainty (%)

        Educational Purpose:
            Show that small errors in high-sensitivity XS cause large k_eff errors!
        """
        variance_k = 0.0

        for (Z, A, MT), xs_uncertainty in cross_section_uncertainties.items():
            sensitivity = self.compute_sensitivity((Z, A), MT)

            # Contribution to k_eff variance
            variance_k += (sensitivity * xs_uncertainty / 100.0) ** 2

        # Standard deviation
        std_k = np.sqrt(variance_k)

        # Convert to percentage
        std_k_percent = std_k * 100.0

        return std_k_percent
