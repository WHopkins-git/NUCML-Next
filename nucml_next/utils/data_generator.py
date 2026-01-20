"""Synthetic data generation for demos."""
import numpy as np
import pandas as pd

class DataGenerator:
    """Generate synthetic nuclear data."""

    @staticmethod
    def generate_resonance_curve(energies, E_r=10.0, Gamma=2.0, sigma_0=100.0):
        """Generate Breit-Wigner resonance."""
        return sigma_0 * Gamma / ((energies - E_r)**2 + Gamma**2 / 4)
