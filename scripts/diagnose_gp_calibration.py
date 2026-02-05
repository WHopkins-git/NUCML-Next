#!/usr/bin/env python3
"""
GP Uncertainty Calibration Diagnostic
======================================

Quantifies the relationship between local data density and GP predictive
uncertainty (gp_std) across multiple reaction test cases.

Expected Behavior:
    - GP uncertainty should EXPAND where data is sparse
    - GP uncertainty should CONTRACT where data is dense
    - Correlation(density, gp_std) should be NEGATIVE

This script analyzes existing gp_std values from the parquet and optionally
re-fits SVGP to extract learned hyperparameters.

Usage:
    python scripts/diagnose_gp_calibration.py [--parquet PATH] [--refit]

Output:
    - Summary table with correlation, hyperparameters, and uncertainty range
    - Root cause diagnosis based on findings
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

# Test cases to analyze
TEST_CASES = [
    # (Z, A, MT, name)
    (79, 197, 102, "Au-197(n,g)"),   # Known issue from visual inspection
    (92, 235, 18, "U-235(n,f)"),     # Dense resonance region, many points
    (26, 56, 2, "Fe-56(n,el)"),      # Structural material, well-measured
    (1, 1, 2, "H-1(n,el)"),          # Simplest nucleus, should be well-behaved
]


def compute_density_per_bin(
    log_E: np.ndarray,
    bin_width: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local data density as points per log-E bin.

    Args:
        log_E: Array of log10(Energy) values
        bin_width: Width of each bin in log-E units

    Returns:
        (bin_centers, point_density) arrays
    """
    e_min, e_max = log_E.min(), log_E.max()
    bins = np.arange(e_min, e_max + bin_width, bin_width)

    counts, bin_edges = np.histogram(log_E, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Density = points per unit log-E
    density = counts / bin_width

    return bin_centers, density


def compute_mean_std_per_bin(
    log_E: np.ndarray,
    gp_std: np.ndarray,
    bin_width: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean gp_std within each log-E bin.

    Args:
        log_E: Array of log10(Energy) values
        gp_std: Array of GP standard deviations
        bin_width: Width of each bin in log-E units

    Returns:
        (bin_centers, mean_gp_std) arrays
    """
    e_min, e_max = log_E.min(), log_E.max()
    bins = np.arange(e_min, e_max + bin_width, bin_width)

    bin_indices = np.digitize(log_E, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    bin_centers = []
    mean_stds = []

    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            mean_stds.append(gp_std[mask].mean())

    return np.array(bin_centers), np.array(mean_stds)


def analyze_density_uncertainty_correlation(
    df_group: pd.DataFrame,
    bin_width: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute correlation between local density and gp_std for a group.

    Args:
        df_group: DataFrame for one (Z, A, MT) group with log_E, gp_std columns
        bin_width: Width of bins for density calculation

    Returns:
        Dictionary with correlation statistics and diagnostics
    """
    log_E = df_group['log_E'].values
    gp_std = df_group['gp_std'].values

    # Compute bin-wise density and mean std
    bin_centers_d, density = compute_density_per_bin(log_E, bin_width)
    bin_centers_s, mean_std = compute_mean_std_per_bin(log_E, gp_std, bin_width)

    # Align bins (use intersection)
    # Both should have same bins, but filter to non-empty
    if len(density) < 3 or len(mean_std) < 3:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'n_bins': 0,
            'density_bins': [],
            'std_bins': [],
        }

    # Compute correlation
    # Use Spearman rank correlation (robust to outliers)
    corr, p_value = stats.spearmanr(density[:len(mean_std)], mean_std)

    return {
        'correlation': corr,
        'p_value': p_value,
        'n_bins': len(mean_std),
        'density_bins': density[:len(mean_std)].tolist(),
        'std_bins': mean_std.tolist(),
    }


def extract_hyperparameters_via_refit(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    n_inducing: int = 50,
    max_epochs: int = 300,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Re-fit SVGP to a group and extract learned hyperparameters.

    Uses the extract_svgp_hyperparameters function from the outlier_detection
    module, which ensures consistency with the actual SVGP implementation.

    Args:
        log_E: log10(Energy) values
        log_sigma: log10(CrossSection) values
        n_inducing: Number of inducing points
        max_epochs: Maximum training epochs
        device: PyTorch device

    Returns:
        Dictionary with lengthscale, outputscale, noise, and inducing points
    """
    try:
        from nucml_next.data.outlier_detection import (
            extract_svgp_hyperparameters,
            SVGPConfig,
        )
    except ImportError:
        return {
            'error': 'nucml_next.data.outlier_detection not available',
            'lengthscale': np.nan,
            'outputscale': np.nan,
            'noise': np.nan,
        }

    config = SVGPConfig(
        n_inducing=n_inducing,
        max_epochs=max_epochs,
        device=device,
    )

    result = extract_svgp_hyperparameters(log_E, log_sigma, config)
    result['n_inducing'] = len(result.get('inducing_points', []))

    return result


def analyze_single_group(
    df_group: pd.DataFrame,
    z: int, a: int, mt: int, name: str,
    refit: bool = False,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Full analysis for a single (Z, A, MT) group.

    Args:
        df_group: DataFrame for the group
        z, a, mt: Isotope and reaction identifiers
        name: Human-readable reaction name
        refit: Whether to re-fit SVGP to extract hyperparameters
        device: PyTorch device for refitting

    Returns:
        Dictionary with all diagnostic metrics
    """
    n = len(df_group)
    log_E = df_group['log_E'].values
    gp_std = df_group['gp_std'].values
    log_sigma = df_group['log_sigma'].values

    # Basic statistics
    result = {
        'name': name,
        'Z': z, 'A': a, 'MT': mt,
        'n_points': n,
        'E_range': (10**log_E.min(), 10**log_E.max()),
        'log_E_range': (log_E.min(), log_E.max()),
    }

    # Uncertainty range
    result['min_gp_std'] = gp_std.min()
    result['max_gp_std'] = gp_std.max()
    result['max_min_ratio'] = gp_std.max() / gp_std.min() if gp_std.min() > 0 else np.inf
    result['mean_gp_std'] = gp_std.mean()
    result['std_gp_std'] = gp_std.std()

    # Density-uncertainty correlation
    corr_result = analyze_density_uncertainty_correlation(df_group, bin_width=0.5)
    result['density_std_correlation'] = corr_result['correlation']
    result['correlation_p_value'] = corr_result['p_value']
    result['n_bins'] = corr_result['n_bins']

    # Hyperparameters (if refitting)
    if refit and n >= 10:
        print(f"  Re-fitting SVGP for {name}...")
        hp = extract_hyperparameters_via_refit(log_E, log_sigma, device=device)
        result['lengthscale'] = hp.get('lengthscale', np.nan)
        result['outputscale'] = hp.get('outputscale', np.nan)
        result['noise'] = hp.get('noise', np.nan)
        result['noise_signal_ratio'] = hp.get('noise_signal_ratio', np.nan)
    else:
        result['lengthscale'] = np.nan
        result['outputscale'] = np.nan
        result['noise'] = np.nan
        result['noise_signal_ratio'] = np.nan

    return result


def print_group_report(result: Dict[str, Any]) -> None:
    """Print detailed report for a single group."""
    print()
    print(f"Test Case: {result['name']} (Z={result['Z']}, A={result['A']}, MT={result['MT']})")
    print(f"  Data points:     {result['n_points']:,}")
    print(f"  Energy range:    {result['E_range'][0]:.2e} to {result['E_range'][1]:.2e} eV")
    print(f"  log10(E) range:  {result['log_E_range'][0]:.2f} to {result['log_E_range'][1]:.2f}")
    print()
    print("  Density-Uncertainty Correlation:")
    print(f"    Bins (0.5 log-E):  {result['n_bins']}")
    corr = result['density_std_correlation']
    corr_str = f"{corr:+.3f}" if not np.isnan(corr) else "N/A"
    p_val = result['correlation_p_value']
    p_str = f"(p={p_val:.3f})" if not np.isnan(p_val) else ""
    expected = "  <-- Should be strongly NEGATIVE if calibrated"
    print(f"    Spearman r(density, std): {corr_str} {p_str}{expected}")
    print()
    print("  Uncertainty Range:")
    print(f"    min(gp_std):       {result['min_gp_std']:.4f}")
    print(f"    max(gp_std):       {result['max_gp_std']:.4f}")
    print(f"    max/min ratio:     {result['max_min_ratio']:.2f}  <-- Should be >> 1 if calibrated")
    print(f"    mean(gp_std):      {result['mean_gp_std']:.4f}")
    print(f"    std(gp_std):       {result['std_gp_std']:.4f}")

    if not np.isnan(result.get('lengthscale', np.nan)):
        print()
        print("  Learned Hyperparameters:")
        print(f"    lengthscale:       {result['lengthscale']:.4f}  <-- Large values cause over-smoothing")
        print(f"    outputscale:       {result['outputscale']:.4f}")
        print(f"    noise:             {result['noise']:.4f}")
        print(f"    noise/signal:      {result['noise_signal_ratio']:.4f}  <-- If >> 1, pred. variance ~ constant")


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print summary table comparing all test cases."""
    print()
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Header
    print(f"{'Reaction':<15} | {'n':<8} | {'r(d,std)':<10} | {'max/min':<10} | {'lengthscale':<12} | {'noise/signal':<12}")
    print("-" * 100)

    for r in results:
        corr = r['density_std_correlation']
        corr_str = f"{corr:+.3f}" if not np.isnan(corr) else "N/A"
        ratio = r['max_min_ratio']
        ratio_str = f"{ratio:.2f}" if not np.isinf(ratio) else "inf"
        ls = r.get('lengthscale', np.nan)
        ls_str = f"{ls:.4f}" if not np.isnan(ls) else "N/A"
        ns = r.get('noise_signal_ratio', np.nan)
        ns_str = f"{ns:.4f}" if not np.isnan(ns) else "N/A"

        print(f"{r['name']:<15} | {r['n_points']:<8,} | {corr_str:<10} | {ratio_str:<10} | {ls_str:<12} | {ns_str:<12}")


def print_diagnosis(results: List[Dict[str, Any]]) -> None:
    """Print root cause diagnosis based on results."""
    print()
    print("=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    # Check correlation
    correlations = [r['density_std_correlation'] for r in results if not np.isnan(r['density_std_correlation'])]
    avg_corr = np.mean(correlations) if correlations else np.nan

    # Check max/min ratio
    ratios = [r['max_min_ratio'] for r in results if not np.isinf(r['max_min_ratio'])]
    avg_ratio = np.mean(ratios) if ratios else np.nan

    # Check lengthscales
    lengthscales = [r['lengthscale'] for r in results if not np.isnan(r.get('lengthscale', np.nan))]
    avg_ls = np.mean(lengthscales) if lengthscales else np.nan

    # Check noise/signal ratios
    ns_ratios = [r['noise_signal_ratio'] for r in results if not np.isnan(r.get('noise_signal_ratio', np.nan))]
    avg_ns = np.mean(ns_ratios) if ns_ratios else np.nan

    findings = []

    # Analyze correlation
    if not np.isnan(avg_corr):
        if avg_corr > -0.3:
            findings.append(
                f"* WEAK density-uncertainty correlation (avg r = {avg_corr:+.3f})\n"
                f"  Expected: strongly negative (< -0.5) if GP is properly calibrated.\n"
                f"  This confirms the visual observation of constant-width uncertainty bands."
            )

    # Analyze max/min ratio
    if not np.isnan(avg_ratio):
        if avg_ratio < 2.0:
            findings.append(
                f"* LOW uncertainty range (avg max/min = {avg_ratio:.2f})\n"
                f"  Expected: ratio >> 2 for well-calibrated GP with variable density.\n"
                f"  The GP is producing nearly constant uncertainty across energy."
            )

    # Analyze lengthscale
    if not np.isnan(avg_ls):
        # Large lengthscale relative to data range causes over-smoothing
        data_ranges = [r['log_E_range'][1] - r['log_E_range'][0] for r in results]
        avg_range = np.mean(data_ranges)
        if avg_ls > avg_range / 3:
            findings.append(
                f"* LARGE lengthscale (avg = {avg_ls:.2f}, data range ~ {avg_range:.1f})\n"
                f"  Lengthscale > 1/3 of data range causes over-smoothing.\n"
                f"  Consider adding a lengthscale prior to prevent this."
            )

    # Analyze noise/signal
    if not np.isnan(avg_ns):
        if avg_ns > 0.3:
            findings.append(
                f"* HIGH noise/signal ratio (avg = {avg_ns:.3f})\n"
                f"  When noise dominates signal, predictive variance ~ noise variance (constant).\n"
                f"  This is likely caused by the homoscedastic GaussianLikelihood."
            )

    if findings:
        print("\nRoot Cause Findings:\n")
        for f in findings:
            print(f)
            print()
    else:
        print("\nNo clear calibration issues detected from the quantitative analysis.")
        print("Consider visual inspection of individual groups.")

    # Recommendations
    print("-" * 100)
    print("RECOMMENDED FIXES (in priority order):")
    print("-" * 100)
    print("""
1. HETEROSCEDASTIC LIKELIHOOD
   Replace GaussianLikelihood with FixedNoiseGaussianLikelihood using measurement uncertainties.
   Nuclear data has known experimental uncertainties that vary with energy.

2. LENGTHSCALE PRIOR
   Add gpytorch.priors.GammaPrior(2.0, 1.0) to the RBF kernel lengthscale.
   Prevents over-smoothing by penalizing very large lengthscales.

3. ADAPTIVE INDUCING POINTS
   Use density-weighted inducing point placement instead of uniform spacing.
   More inducing points in dense regions, fewer in sparse regions.

4. VALIDATION-BASED HYPERPARAMETERS
   Split data into train/validation, optimize hyperparameters by NLL on validation.
   Current approach only maximizes ELBO on training data.
""")


def main():
    parser = argparse.ArgumentParser(description="GP Uncertainty Calibration Diagnostic")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/exfor_processed.parquet",
        help="Path to parquet file with gp_std column"
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Re-fit SVGP to extract learned hyperparameters (slower)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for refitting (cpu or cuda)"
    )
    args = parser.parse_args()

    print("=" * 100)
    print("GP UNCERTAINTY CALIBRATION DIAGNOSTIC")
    print("=" * 100)
    print(f"Parquet: {args.parquet}")
    print(f"Re-fit:  {args.refit}")
    print(f"Device:  {args.device}")

    # Load data
    print("\nLoading parquet...")
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        parquet_path = script_dir / args.parquet

    if not parquet_path.exists():
        print(f"ERROR: Parquet file not found: {args.parquet}")
        return 1

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows")

    # Check required columns
    required = ['Z', 'A', 'MT', 'log_E', 'log_sigma', 'gp_std']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return 1

    # Analyze each test case
    results = []

    for z, a, mt, name in TEST_CASES:
        print(f"\n{'-' * 80}")
        print(f"Analyzing: {name} (Z={z}, A={a}, MT={mt})")

        mask = (df['Z'] == z) & (df['A'] == a) & (df['MT'] == mt)
        df_group = df.loc[mask]

        if len(df_group) == 0:
            print(f"  WARNING: No data found for {name}")
            continue

        result = analyze_single_group(
            df_group, z, a, mt, name,
            refit=args.refit,
            device=args.device,
        )
        results.append(result)
        print_group_report(result)

    if not results:
        print("\nERROR: No test cases could be analyzed")
        return 1

    # Summary and diagnosis
    print_summary_table(results)
    print_diagnosis(results)

    return 0


if __name__ == "__main__":
    exit(main())
