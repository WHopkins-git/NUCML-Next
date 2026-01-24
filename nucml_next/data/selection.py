"""
Data Selection and Filtering
=============================

Physics-aware data selection for nuclear cross-section ML.

Provides structured configuration for:
- Projectile/channel domain (neutrons vs all)
- Energy range (reactor physics vs full spectrum)
- Reaction (MT) mode (reactor core, threshold, fission, all physical)
- Data validity (drop NaN/non-positive cross-sections)
- Evaluation controls (holdout isotopes for true extrapolation testing)

Design Philosophy:
- Predicate pushdown: Filter at PyArrow fragment level to minimize I/O
- Scientific defaults: Reactor physics applications (10^-5 eV to 20 MeV, neutrons)
- Explicit exclusions: Avoid bookkeeping codes that lead to arithmetic memorization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set
import numpy as np


# MT Code Categories (based on ENDF-6 format manual)
# Reference: https://www.oecd-nea.org/dbdata/data/manual-endf/endf102.pdf

# Neutron-induced reactions (MT codes strongly associated with neutron projectiles)
NEUTRON_MT_CODES = {
    2,    # Elastic scattering
    4,    # Inelastic scattering
    16,   # (n,2n)
    17,   # (n,3n)
    18,   # Fission
    19,   # (n,f) - first chance fission
    20,   # (n,nf) - second chance fission
    21,   # (n,2nf) - third chance fission
    38,   # (n,4nf) - fourth chance fission
    102,  # (n,γ) - radiative capture
    103,  # (n,p)
    104,  # (n,d)
    105,  # (n,t)
    106,  # (n,³He)
    107,  # (n,α)
    # Add more as needed - these are the most common
}

# Reactor core essential reactions (for criticality, burnup, shielding)
REACTOR_CORE_MT = [2, 4, 16, 18, 102, 103, 107]

# Threshold reactions (require energy above neutron separation energy)
THRESHOLD_MT = [16, 17, 103, 104, 105, 106, 107]

# Fission details (breakdown of fission channels)
FISSION_DETAILS_MT = [18, 19, 20, 21, 38]

# Bookkeeping codes to exclude (totals, derived quantities)
BOOKKEEPING_MT = {0, 1}  # MT=1 is total cross-section, MT=0 is undefined


@dataclass
class DataSelection:
    """
    Configuration for physics-aware data selection.

    This class encapsulates all filtering logic for nuclear cross-section data,
    enabling efficient predicate pushdown to PyArrow fragment reading.

    Attributes:
        projectile: Particle type ('neutron', 'all'). Default: 'neutron'
        energy_min: Minimum energy in eV. Default: 1e-5 eV (thermal)
        energy_max: Maximum energy in eV. Default: 2e7 eV (20 MeV)
        mt_mode: Reaction selection mode. Options:
            - 'reactor_core': Essential for reactor physics (MT 2,4,16,18,102,103,107)
            - 'threshold_only': Reactions with energy thresholds (MT 16,17,103-107)
            - 'fission_details': Fission breakdown (MT 18,19,20,21,38)
            - 'all_physical': All codes < 9000, excluding bookkeeping
            - 'custom': Use custom_mt_codes list
        custom_mt_codes: Custom MT code list (used when mt_mode='custom')
        exclude_bookkeeping: Exclude MT 0,1 and MT >= 9000. Default: True
        drop_invalid: Drop NaN or non-positive cross-sections. Default: True
        holdout_isotopes: List of (Z,A) tuples to exclude from training.
                         Use for measuring true extrapolation capability.
                         Default: None (no holdouts)
        tiers: List of feature tiers to include (e.g., ['A', 'C']). Options:
            - 'A': Core features (Z, A, Energy, particle emission)
            - 'B': + Geometric features (nuclear radius, kR)
            - 'C': + Energetics (mass excess, binding energy, separation energies)
            - 'D': + Topological (spin, parity, valence, magic numbers)
            - 'E': + Complete Q-values (all reaction energetics)
                         Default: ['A'] (core features only, equivalent to 'naive' mode)

    Example:
        >>> # Default: reactor physics, neutrons only, Tier A features
        >>> selection = DataSelection()

        >>> # Tier C features (Energetics): includes mass excess, binding energy, separation energies
        >>> selection = DataSelection(
        ...     tiers=['A', 'B', 'C'],
        ...     energy_min=1e-5,
        ...     energy_max=2e7
        ... )

        >>> # Holdout U-235 and Cl-35 for evaluation with Tier E features
        >>> selection = DataSelection(
        ...     tiers=['A', 'C', 'E'],
        ...     holdout_isotopes=[(92, 235), (17, 35)]
        ... )
    """

    # Projectile selection
    projectile: str = 'neutron'  # 'neutron' or 'all'

    # Energy range (eV)
    energy_min: float = 1e-5   # Thermal neutrons
    energy_max: float = 2e7    # 20 MeV (reactor physics upper bound)

    # Reaction (MT) selection
    mt_mode: str = 'reactor_core'  # 'reactor_core', 'threshold_only', 'fission_details', 'all_physical', 'custom'
    custom_mt_codes: Optional[List[int]] = None

    # Exclusion rules
    exclude_bookkeeping: bool = True  # Exclude MT 0,1 and MT >= 9000

    # Data validity
    drop_invalid: bool = True  # Drop NaN or non-positive cross-sections

    # Evaluation controls
    holdout_isotopes: Optional[List[Tuple[int, int]]] = None  # [(Z, A), ...] to exclude

    # Feature tier selection (Valdez 2021 hierarchy)
    tiers: List[str] = field(default_factory=lambda: ['A'])  # Default: Core features only

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate projectile
        if self.projectile not in ['neutron', 'all']:
            raise ValueError(f"projectile must be 'neutron' or 'all', got '{self.projectile}'")

        # Validate energy range
        if self.energy_min <= 0:
            raise ValueError(f"energy_min must be positive, got {self.energy_min}")
        if self.energy_max <= self.energy_min:
            raise ValueError(f"energy_max ({self.energy_max}) must be > energy_min ({self.energy_min})")

        # Validate mt_mode
        valid_modes = ['reactor_core', 'threshold_only', 'fission_details', 'all_physical', 'custom']
        if self.mt_mode not in valid_modes:
            raise ValueError(f"mt_mode must be one of {valid_modes}, got '{self.mt_mode}'")

        # Validate custom_mt_codes
        if self.mt_mode == 'custom' and not self.custom_mt_codes:
            raise ValueError("custom_mt_codes must be provided when mt_mode='custom'")

        # Validate tiers
        valid_tiers = ['A', 'B', 'C', 'D', 'E']
        for tier in self.tiers:
            if tier not in valid_tiers:
                raise ValueError(f"Invalid tier '{tier}'. Must be one of {valid_tiers}")

        # Ensure Tier A is always included (base features required)
        if 'A' not in self.tiers and len(self.tiers) > 0:
            self.tiers = ['A'] + self.tiers

    def get_mt_codes(self) -> List[int]:
        """
        Get the list of MT codes based on mt_mode.

        Returns:
            List of MT codes to include in selection
        """
        if self.mt_mode == 'reactor_core':
            mt_codes = REACTOR_CORE_MT.copy()
        elif self.mt_mode == 'threshold_only':
            mt_codes = THRESHOLD_MT.copy()
        elif self.mt_mode == 'fission_details':
            mt_codes = FISSION_DETAILS_MT.copy()
        elif self.mt_mode == 'all_physical':
            # CRITICAL FIX: When projectile='neutron', filter to neutron MT codes
            # This enables predicate pushdown at fragment level (90% fewer fragments!)
            if self.projectile == 'neutron':
                mt_codes = sorted(list(NEUTRON_MT_CODES))
            else:
                # All physical codes: 2-8999 (exclude bookkeeping)
                # We'll filter at runtime based on available data
                mt_codes = None  # Signal to not filter by MT in this mode
        elif self.mt_mode == 'custom':
            mt_codes = self.custom_mt_codes.copy()
        else:
            raise ValueError(f"Unknown mt_mode: {self.mt_mode}")

        # Apply exclusions if requested
        if self.exclude_bookkeeping and mt_codes is not None:
            mt_codes = [mt for mt in mt_codes if mt not in BOOKKEEPING_MT and mt < 9000]

        return mt_codes

    def get_projectile_mt_filter(self) -> Optional[Set[int]]:
        """
        Get MT codes associated with selected projectile.

        Returns:
            Set of MT codes for projectile, or None if 'all'
        """
        if self.projectile == 'neutron':
            return NEUTRON_MT_CODES
        else:
            return None  # No filtering

    def should_exclude_isotope(self, z: int, a: int) -> bool:
        """
        Check if an isotope should be excluded (holdout).

        Args:
            z: Atomic number
            a: Mass number

        Returns:
            True if isotope is in holdout list
        """
        if self.holdout_isotopes is None:
            return False
        return (z, a) in self.holdout_isotopes

    def __repr__(self) -> str:
        """Readable representation of selection criteria."""
        lines = [
            "DataSelection(",
            f"  Projectile: {self.projectile}",
            f"  Energy: {self.energy_min:.2e} - {self.energy_max:.2e} eV",
            f"  MT mode: {self.mt_mode}",
        ]

        mt_codes = self.get_mt_codes()
        if mt_codes is not None:
            lines.append(f"  MT codes: {sorted(mt_codes)[:10]}{'...' if len(mt_codes) > 10 else ''} ({len(mt_codes)} total)")
        else:
            lines.append(f"  MT codes: all physical (< 9000)")

        # Show tier selection
        lines.append(f"  Feature tiers: {self.tiers}")

        if self.holdout_isotopes:
            lines.append(f"  Holdout isotopes: {self.holdout_isotopes}")

        lines.append(f"  Drop invalid: {self.drop_invalid}")
        lines.append(")")

        return "\n".join(lines)


# Convenience functions for common use cases

def default_selection() -> DataSelection:
    """Default: reactor physics, neutrons, essential reactions."""
    return DataSelection()


def full_spectrum_selection() -> DataSelection:
    """Full energy range, all physical reactions."""
    return DataSelection(
        energy_min=1e-5,
        energy_max=1e9,  # 1 GeV
        mt_mode='all_physical'
    )


def evaluation_selection(holdout_isotopes: List[Tuple[int, int]]) -> DataSelection:
    """
    Selection for evaluation with holdout isotopes.

    Args:
        holdout_isotopes: List of (Z, A) tuples to exclude from training

    Example:
        >>> # Train on everything except U-235 and Cl-35
        >>> sel = evaluation_selection([(92, 235), (17, 35)])
    """
    return DataSelection(holdout_isotopes=holdout_isotopes)
