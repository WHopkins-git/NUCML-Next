#!/usr/bin/env python
"""
Create synthetic nuclear cross-section data for demonstration.

This generates realistic-looking data with:
- U-235 fission with resonance peaks
- Cl-35 (n,p) with threshold behavior
- Sufficient data points for ML training
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def breit_wigner(E, E_r, Gamma, sigma_0):
    """Breit-Wigner resonance formula."""
    return sigma_0 * Gamma**2 / ((E - E_r)**2 + Gamma**2/4)

def create_u235_fission():
    """Create U-235 fission data with resonances."""
    # Resonance region (0.01 - 1000 eV)
    E_res = np.logspace(-2, 3, 2000)  # 2000 points in resonance region

    # Realistic U-235 fission resonances
    resonances = [
        (0.29, 0.1, 1500),    # Strong resonance at 0.29 eV
        (1.14, 0.15, 800),    # 1.14 eV resonance
        (2.03, 0.12, 600),    # 2.03 eV
        (3.14, 0.18, 700),    # 3.14 eV
        (3.61, 0.14, 500),    # 3.61 eV
        (4.85, 0.16, 550),    # 4.85 eV
        (6.39, 0.20, 600),    # 6.39 eV
        (8.78, 0.22, 650),    # 8.78 eV
        (10.16, 0.24, 700),   # 10.16 eV
        (11.66, 0.26, 750),   # 11.66 eV
        (12.39, 0.25, 800),   # 12.39 eV
        (14.40, 0.28, 850),   # 14.40 eV
        (15.40, 0.30, 900),   # 15.40 eV
        (16.66, 0.32, 950),   # 16.66 eV
        (18.05, 0.34, 1000),  # 18.05 eV
        (19.30, 0.36, 1050),  # 19.30 eV
        (20.60, 0.38, 1100),  # 20.60 eV
        (21.07, 0.35, 1150),  # 21.07 eV
        (23.42, 0.40, 1200),  # 23.42 eV
        (27.80, 0.45, 1300),  # 27.80 eV
        (32.07, 0.50, 1400),  # 32.07 eV
        (35.24, 0.52, 1450),  # 35.24 eV
        (39.43, 0.55, 1500),  # 39.43 eV
        (48.85, 0.60, 1600),  # 48.85 eV
        (66.03, 0.70, 1700),  # 66.03 eV
        (82.38, 0.80, 1800),  # 82.38 eV
        (103.0, 0.90, 1900),  # 103 eV
    ]

    # Calculate cross section with multiple resonances + background
    sigma = np.full_like(E_res, 8.0)  # Background at 8 barns
    for E_r, Gamma, sigma_0 in resonances:
        sigma += breit_wigner(E_res, E_r, Gamma, sigma_0)

    # Add realistic noise (5% uncertainty)
    sigma *= (1 + np.random.normal(0, 0.05, len(sigma)))
    sigma = np.maximum(sigma, 0.1)  # No negative cross sections

    # Fast neutron region (1 keV - 20 MeV)
    E_fast = np.logspace(3, 7, 1000)  # 1000 points in fast region

    # Fast fission cross section (decreases with energy, ~1-2 barns)
    sigma_fast = 1.8 - 0.3 * np.log10(E_fast/1e3)
    sigma_fast += np.random.normal(0, 0.05, len(sigma_fast))
    sigma_fast = np.maximum(sigma_fast, 0.5)

    # Combine regions
    E_all = np.concatenate([E_res, E_fast])
    sigma_all = np.concatenate([sigma, sigma_fast])

    df = pd.DataFrame({
        'Entry': ['DEMO-U235-FIS'] * len(E_all),
        'Z': 92,
        'A': 235,
        'N': 143,
        'MT': 18,  # Fission
        'Energy': E_all,
        'CrossSection': sigma_all,
        'Uncertainty': sigma_all * 0.05,  # 5% uncertainty
    })

    return df

def create_u235_capture():
    """Create U-235 capture data."""
    # Capture decreases with energy (1/v behavior at thermal, resonances)
    E = np.logspace(-2, 7, 2000)

    # 1/v behavior + resonances
    sigma = 680 / np.sqrt(E/0.0253)  # 1/v from thermal

    # Add some resonance structure
    sigma += breit_wigner(E, 0.29, 0.05, 200)
    sigma += breit_wigner(E, 1.14, 0.08, 150)
    sigma += breit_wigner(E, 2.03, 0.06, 100)

    # Decrease in fast region
    mask_fast = E > 1e3
    sigma[mask_fast] = 3.0 - 0.5 * np.log10(E[mask_fast]/1e3)

    # Add noise
    sigma *= (1 + np.random.normal(0, 0.05, len(sigma)))
    sigma = np.maximum(sigma, 0.01)

    df = pd.DataFrame({
        'Entry': ['DEMO-U235-CAP'] * len(E),
        'Z': 92,
        'A': 235,
        'N': 143,
        'MT': 102,  # Capture
        'Energy': E,
        'CrossSection': sigma,
        'Uncertainty': sigma * 0.05,
    })

    return df

def create_cl35_np():
    """Create Cl-35 (n,p) data with threshold behavior."""
    # (n,p) has threshold around 0.6 MeV, peaks around 5-10 MeV
    E = np.logspace(5, 7.3, 600)  # 0.1 MeV to 20 MeV

    E_threshold = 6e5  # 0.6 MeV threshold
    sigma = np.zeros_like(E)

    # Threshold behavior: sigma = 0 below threshold
    mask = E > E_threshold
    E_above = E[mask]

    # Rising from threshold, peaking around 5-10 MeV, then decreasing
    x = (E_above - E_threshold) / 1e6  # MeV above threshold
    sigma[mask] = 0.15 * x * np.exp(-x/10) * (1 + 0.3*np.sin(x*2))

    # Add noise
    sigma[mask] *= (1 + np.random.normal(0, 0.1, mask.sum()))
    sigma = np.maximum(sigma, 0.0)

    df = pd.DataFrame({
        'Entry': ['DEMO-CL35-NP'] * len(E),
        'Z': 17,
        'A': 35,
        'N': 18,
        'MT': 103,  # (n,p)
        'Energy': E,
        'CrossSection': sigma,
        'Uncertainty': np.where(sigma > 0, sigma * 0.1, 0),
    })

    return df

def create_additional_isotopes():
    """Create data for additional isotopes for training diversity."""
    dfs = []

    # Fe-56 elastic (common structural material)
    E = np.logspace(0, 7, 1500)
    sigma = 12 - 2 * np.log10(E/1.0)  # Decreasing with energy
    sigma += np.random.normal(0, 0.5, len(sigma))
    sigma = np.maximum(sigma, 1.0)

    dfs.append(pd.DataFrame({
        'Entry': ['DEMO-FE56-ELAS'] * len(E),
        'Z': 26, 'A': 56, 'N': 30, 'MT': 2, 'Energy': E,
        'CrossSection': sigma, 'Uncertainty': sigma * 0.05,
    }))

    # C-12 elastic (moderator)
    E = np.logspace(0, 7, 1500)
    sigma = 4.8 - 0.3 * np.log10(E/1.0)
    sigma += np.random.normal(0, 0.2, len(sigma))
    sigma = np.maximum(sigma, 2.0)

    dfs.append(pd.DataFrame({
        'Entry': ['DEMO-C12-ELAS'] * len(E),
        'Z': 6, 'A': 12, 'N': 6, 'MT': 2, 'Energy': E,
        'CrossSection': sigma, 'Uncertainty': sigma * 0.05,
    }))

    # O-16 elastic
    E = np.logspace(0, 7, 1500)
    sigma = 3.8 - 0.2 * np.log10(E/1.0)
    sigma += np.random.normal(0, 0.2, len(sigma))
    sigma = np.maximum(sigma, 2.0)

    dfs.append(pd.DataFrame({
        'Entry': ['DEMO-O16-ELAS'] * len(E),
        'Z': 8, 'A': 16, 'N': 8, 'MT': 2, 'Energy': E,
        'CrossSection': sigma, 'Uncertainty': sigma * 0.05,
    }))

    # Pu-239 fission (for diversity)
    E = np.logspace(-2, 7, 2000)
    sigma = 750 / np.sqrt(E/0.0253)  # 1/v at thermal
    sigma[E > 1e3] = 2.0 - 0.2 * np.log10(E[E > 1e3]/1e3)
    sigma += np.random.normal(0, 0.05 * sigma.max(), len(sigma))
    sigma = np.maximum(sigma, 0.5)

    dfs.append(pd.DataFrame({
        'Entry': ['DEMO-PU239-FIS'] * len(E),
        'Z': 94, 'A': 239, 'N': 145, 'MT': 18, 'Energy': E,
        'CrossSection': sigma, 'Uncertainty': sigma * 0.05,
    }))

    return pd.concat(dfs, ignore_index=True)

def main():
    print("="*70)
    print("Creating Demonstration Nuclear Cross-Section Dataset")
    print("="*70)

    # Create datasets
    print("\nGenerating data:")
    print("  - U-235 Fission (resonances)")
    df_u235_fis = create_u235_fission()
    print(f"    ✓ {len(df_u235_fis):,} points")

    print("  - U-235 Capture")
    df_u235_cap = create_u235_capture()
    print(f"    ✓ {len(df_u235_cap):,} points")

    print("  - Cl-35 (n,p)")
    df_cl35 = create_cl35_np()
    print(f"    ✓ {len(df_cl35):,} points")

    print("  - Additional isotopes (Fe, C, O, Pu)")
    df_extra = create_additional_isotopes()
    print(f"    ✓ {len(df_extra):,} points")

    # Combine
    df_all = pd.concat([df_u235_fis, df_u235_cap, df_cl35, df_extra], ignore_index=True)

    print(f"\n✓ Total dataset: {len(df_all):,} measurements")
    print(f"✓ Isotopes: {df_all.groupby(['Z', 'A']).ngroups}")
    print(f"✓ Reactions: {df_all['MT'].nunique()}")

    # Save as partitioned Parquet
    output_path = Path('data/exfor_processed.parquet')
    print(f"\nSaving to {output_path}...")

    # Remove old data
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        print("  Removed old dataset")

    # Convert to PyArrow table and write partitioned
    table = pa.Table.from_pandas(df_all)
    pq.write_to_dataset(
        table,
        root_path=str(output_path),
        partition_cols=['Z', 'A', 'MT'],
    )

    print(f"✓ Saved {len(df_all):,} measurements")
    print("\nDataset ready for ML training!")
    print("="*70)

if __name__ == '__main__':
    main()
