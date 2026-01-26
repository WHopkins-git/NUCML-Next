"""
MT Code Reference
==================

Complete mapping of ENDF-6 format MT (reaction type) codes to human-readable names.

Reference: ENDF-102 Data Formats and Procedures for the Evaluated Nuclear Data File ENDF-6
URL: https://www.oecd-nea.org/dbdata/data/manual-endf/endf102.pdf

MT codes define reaction types in nuclear data libraries:
- 1-999: Standard reactions
- 1000+: Special reactions (capture gamma production, etc.)
- 9000+: Lumped covariance data

This module provides comprehensive mappings for visualization and analysis.
"""

# Complete MT code dictionary (standard ENDF-6 codes)
MT_NAMES = {
    # ========================================================================
    # FUNDAMENTAL CROSS SECTIONS (1-5)
    # ========================================================================
    1: "Total",
    2: "Elastic",
    3: "Nonelastic",
    4: "Inelastic",
    5: "Other/Misc",

    # ========================================================================
    # PARTICLE EMISSION (10-45)
    # ========================================================================
    10: "(n,continuum)",
    11: "(n,2nd)",
    16: "(n,2n)",
    17: "(n,3n)",
    18: "Fission",
    19: "(n,f) - 1st chance",
    20: "(n,nf) - 2nd chance",
    21: "(n,2nf) - 3rd chance",
    22: "(n,nα)",
    23: "(n,n3α)",
    24: "(n,2nα)",
    25: "(n,3nα)",
    28: "(n,np)",
    29: "(n,n2α)",
    30: "(n,2n2α)",
    32: "(n,nd)",
    33: "(n,nt)",
    34: "(n,n³He)",
    35: "(n,nd2α)",
    36: "(n,nt2α)",
    37: "(n,4n)",
    38: "(n,3nf) - 4th chance",
    41: "(n,2np)",
    42: "(n,3np)",
    44: "(n,n2p)",
    45: "(n,npa)",

    # ========================================================================
    # INELASTIC LEVELS (50-91)
    # ========================================================================
    50: "(n,n₀) - Ground state",
    51: "(n,n₁) - 1st excited",
    52: "(n,n₂) - 2nd excited",
    53: "(n,n₃) - 3rd excited",
    54: "(n,n₄) - 4th excited",
    55: "(n,n₅) - 5th excited",
    56: "(n,n₆) - 6th excited",
    57: "(n,n₇) - 7th excited",
    58: "(n,n₈) - 8th excited",
    59: "(n,n₉) - 9th excited",
    60: "(n,n₁₀) - 10th excited",
    61: "(n,n₁₁) - 11th excited",
    62: "(n,n₁₂) - 12th excited",
    63: "(n,n₁₃) - 13th excited",
    64: "(n,n₁₄) - 14th excited",
    65: "(n,n₁₅) - 15th excited",
    66: "(n,n₁₆) - 16th excited",
    67: "(n,n₁₇) - 17th excited",
    68: "(n,n₁₈) - 18th excited",
    69: "(n,n₁₉) - 19th excited",
    70: "(n,n₂₀) - 20th excited",
    71: "(n,n₂₁) - 21st excited",
    72: "(n,n₂₂) - 22nd excited",
    73: "(n,n₂₃) - 23rd excited",
    74: "(n,n₂₄) - 24th excited",
    75: "(n,n₂₅) - 25th excited",
    76: "(n,n₂₆) - 26th excited",
    77: "(n,n₂₇) - 27th excited",
    78: "(n,n₂₈) - 28th excited",
    79: "(n,n₂₉) - 29th excited",
    80: "(n,n₃₀) - 30th excited",
    81: "(n,n₃₁) - 31st excited",
    82: "(n,n₃₂) - 32nd excited",
    83: "(n,n₃₃) - 33rd excited",
    84: "(n,n₃₄) - 34th excited",
    85: "(n,n₃₅) - 35th excited",
    86: "(n,n₃₆) - 36th excited",
    87: "(n,n₃₇) - 37th excited",
    88: "(n,n₃₈) - 38th excited",
    89: "(n,n₃₉) - 39th excited",
    90: "(n,n₄₀) - 40th excited",
    91: "(n,n continuum)",

    # ========================================================================
    # CHARGED PARTICLE EMISSION (100-117)
    # ========================================================================
    102: "(n,γ) Capture",
    103: "(n,p)",
    104: "(n,d)",
    105: "(n,t)",
    106: "(n,³He)",
    107: "(n,α)",
    108: "(n,2α)",
    109: "(n,3α)",
    111: "(n,2p)",
    112: "(n,pα)",
    113: "(n,t2α)",
    114: "(n,d2α)",
    115: "(n,pd)",
    116: "(n,pt)",
    117: "(n,dα)",

    # ========================================================================
    # RESIDUAL PRODUCTION (200-299)
    # ========================================================================
    201: "Total absorption",
    202: "Total capture",
    203: "Total (n,Xp)",
    204: "Total (n,Xd)",
    205: "Total (n,Xt)",
    206: "Total (n,X³He)",
    207: "Total (n,Xα)",

    # ========================================================================
    # FISSION PRODUCTS (454-459)
    # ========================================================================
    454: "ν̄ - Avg neutrons/fission",
    455: "ν̄ delayed",
    456: "ν̄ prompt",
    457: "Q̄ - Energy release",
    458: "Energy release (fragments)",
    459: "Energy release (prompt γ)",

    # ========================================================================
    # PHOTON PRODUCTION (500-572)
    # ========================================================================
    500: "Total photon production",
    501: "(n,total) γ",
    502: "(n,γ) γ production",
    504: "(n,n') γ production",
    515: "(n,p) γ production",
    516: "(n,d) γ production",
    517: "(n,t) γ production",
    518: "(n,³He) γ production",
    522: "(n,nα) γ production",
    526: "(n,n2α) γ production",

    # ========================================================================
    # HEATING (301)
    # ========================================================================
    301: "Heating (KERMA)",

    # ========================================================================
    # DAMAGE (444)
    # ========================================================================
    444: "Damage energy",

    # ========================================================================
    # ISOMERIC STATE PRODUCTION (600-849)
    # ========================================================================
    600: "(n,p₀) - ground",
    601: "(n,p₁) - 1st excited",
    649: "(n,p continuum)",
    650: "(n,d₀) - ground",
    699: "(n,d continuum)",
    700: "(n,t₀) - ground",
    749: "(n,t continuum)",
    750: "(n,³He₀) - ground",
    799: "(n,³He continuum)",
    800: "(n,α₀) - ground",
    801: "(n,α₁) - 1st excited",
    849: "(n,α continuum)",

    # ========================================================================
    # COVARIANCE DATA (9000+)
    # ========================================================================
    9000: "Covariance data (lumped)",
    9001: "Covariance: Total",
    9002: "Covariance: Elastic",
    9004: "Covariance: Inelastic",
    9016: "Covariance: (n,2n)",
    9017: "Covariance: (n,3n)",
    9018: "Covariance: Fission",
    9102: "Covariance: (n,γ)",
    9103: "Covariance: (n,p)",
    9104: "Covariance: (n,d)",
    9105: "Covariance: (n,t)",
    9107: "Covariance: (n,α)",
}


# Categorical groupings for analysis
MT_CATEGORIES = {
    "Fundamental": [1, 2, 3, 4, 5],
    "Neutron Emission": [16, 17, 37],
    "Fission": [18, 19, 20, 21, 38],
    "Charged Particle": [103, 104, 105, 106, 107, 108, 109, 111, 112, 115, 116, 117],
    "Capture": [102],
    "Inelastic Levels": list(range(50, 92)),
    "Photon Production": list(range(500, 573)),
    "Covariance": list(range(9000, 9200)),
}


def get_mt_name(mt_code: int, include_code: bool = True) -> str:
    """
    Get human-readable name for an MT code.

    Args:
        mt_code: MT reaction type code
        include_code: If True, prepend "MT {code}: " to name

    Returns:
        Human-readable reaction name

    Example:
        >>> get_mt_name(18)
        'MT 18: Fission'
        >>> get_mt_name(102, include_code=False)
        '(n,γ) Capture'
    """
    name = MT_NAMES.get(mt_code, f"Unknown (MT {mt_code})")

    if include_code and mt_code in MT_NAMES:
        return f"MT {mt_code}: {name}"
    else:
        return name


def get_mt_category(mt_code: int) -> str:
    """
    Get category for an MT code.

    Args:
        mt_code: MT reaction type code

    Returns:
        Category name or "Other"

    Example:
        >>> get_mt_category(18)
        'Fission'
        >>> get_mt_category(103)
        'Charged Particle'
    """
    for category, codes in MT_CATEGORIES.items():
        if mt_code in codes:
            return category
    return "Other"


def get_reactor_critical_mt_codes():
    """
    Get MT codes critical for reactor physics calculations.

    Returns:
        List of (MT code, name) tuples
    """
    critical = [2, 4, 16, 18, 102, 103, 107]
    return [(mt, MT_NAMES[mt]) for mt in critical]


def get_common_mt_codes():
    """
    Get most commonly measured MT codes in EXFOR.

    Returns:
        List of (MT code, name) tuples
    """
    common = [1, 2, 4, 16, 17, 18, 102, 103, 104, 105, 106, 107]
    return [(mt, MT_NAMES[mt]) for mt in common if mt in MT_NAMES]
