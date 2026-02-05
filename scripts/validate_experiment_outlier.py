#!/usr/bin/env python3
"""
Validation Script for Per-Experiment GP Outlier Detection
==========================================================

Validates the ExperimentOutlierDetector on benchmark reactions to ensure:
1. Reasonable outlier rates (< 15% for most reactions)
2. Proper flagging of discrepant experiments
3. Well-calibrated uncertainty bands

Target reactions:
- U-235(n,f) (Z=92, A=235, MT=18): Standard, should have ~5% experiment outliers
- U-233(n,f) (Z=92, A=233, MT=18): Resonance-rich, was 25% outlier with SVGP
- Au-197(n,gamma) (Z=79, A=197, MT=102): Clean reference standard
- Fe-56(n,el) (Z=26, A=56, MT=2): Structured cross-section
- H-1(n,el) (Z=1, A=1, MT=2): Nearly perfect (< 1% outliers)

Usage:
    python scripts/validate_experiment_outlier.py --data-path data/exfor_full.parquet
    python scripts/validate_experiment_outlier.py --data-path data/exfor_full.parquet --z-threshold 3.0
    python scripts/validate_experiment_outlier.py --data-path data/exfor_full.parquet --save-plots plots/

Output:
- Console summary with outlier statistics
- Optional diagnostic plots (if --save-plots specified)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure nucml_next is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


# Target reactions for validation
TARGET_REACTIONS = [
    {'name': 'U-235(n,f)', 'Z': 92, 'A': 235, 'MT': 18,
     'expected_exp_outlier_rate': 0.10, 'expected_point_outlier_rate': 0.10},
    {'name': 'U-233(n,f)', 'Z': 92, 'A': 233, 'MT': 18,
     'expected_exp_outlier_rate': 0.10, 'expected_point_outlier_rate': 0.15},
    {'name': 'Au-197(n,gamma)', 'Z': 79, 'A': 197, 'MT': 102,
     'expected_exp_outlier_rate': 0.10, 'expected_point_outlier_rate': 0.10},
    {'name': 'Fe-56(n,el)', 'Z': 26, 'A': 56, 'MT': 2,
     'expected_exp_outlier_rate': 0.10, 'expected_point_outlier_rate': 0.10},
    {'name': 'H-1(n,el)', 'Z': 1, 'A': 1, 'MT': 2,
     'expected_exp_outlier_rate': 0.02, 'expected_point_outlier_rate': 0.02},
]


def load_data(data_path: str) -> pd.DataFrame:
    """Load EXFOR data from parquet file."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} data points")
    return df


def filter_reaction(df: pd.DataFrame, Z: int, A: int, MT: int) -> pd.DataFrame:
    """Filter DataFrame to a specific reaction."""
    mask = (df['Z'] == Z) & (df['A'] == A) & (df['MT'] == MT)
    return df[mask].copy()


def run_detector(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    use_wasserstein: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Run ExperimentOutlierDetector on data."""
    from nucml_next.data.experiment_outlier import (
        ExperimentOutlierDetector,
        ExperimentOutlierConfig,
    )

    config = ExperimentOutlierConfig(point_z_threshold=z_threshold)
    config.gp_config.use_wasserstein_calibration = use_wasserstein

    detector = ExperimentOutlierDetector(config)
    result = detector.score_dataframe(df)
    stats = detector.get_statistics()

    return result, stats


def compute_outlier_rates(result: pd.DataFrame) -> Dict:
    """Compute outlier rates from scored DataFrame."""
    n_total = len(result)
    n_experiments = result['experiment_id'].nunique()

    # Point-level outliers
    n_point_outliers = result['point_outlier'].sum()
    point_outlier_rate = n_point_outliers / n_total if n_total > 0 else 0

    # Experiment-level outliers (count unique experiments flagged)
    discrepant_exp = result[result['experiment_outlier']]['experiment_id'].unique()
    n_exp_outliers = len(discrepant_exp)
    exp_outlier_rate = n_exp_outliers / n_experiments if n_experiments > 0 else 0

    return {
        'n_total_points': n_total,
        'n_experiments': n_experiments,
        'n_point_outliers': n_point_outliers,
        'point_outlier_rate': point_outlier_rate,
        'n_exp_outliers': n_exp_outliers,
        'exp_outlier_rate': exp_outlier_rate,
        'discrepant_experiments': list(discrepant_exp),
    }


def compute_calibration_metrics(result: pd.DataFrame) -> Dict:
    """Compute calibration metrics from z-scores."""
    z_scores = result['z_score'].values
    valid_z = z_scores[np.isfinite(z_scores)]

    if len(valid_z) == 0:
        return {'n_valid': 0}

    abs_z = np.abs(valid_z)

    # Fraction within sigma levels (empirical coverage)
    coverage_1sigma = np.mean(abs_z <= 1.0)
    coverage_2sigma = np.mean(abs_z <= 2.0)
    coverage_3sigma = np.mean(abs_z <= 3.0)

    # Theoretical coverage for standard normal
    from scipy.stats import norm
    theo_1sigma = 2 * norm.cdf(1.0) - 1  # ~0.683
    theo_2sigma = 2 * norm.cdf(2.0) - 1  # ~0.954
    theo_3sigma = 2 * norm.cdf(3.0) - 1  # ~0.997

    return {
        'n_valid': len(valid_z),
        'z_mean': np.mean(abs_z),
        'z_std': np.std(valid_z),
        'coverage_1sigma': coverage_1sigma,
        'coverage_2sigma': coverage_2sigma,
        'coverage_3sigma': coverage_3sigma,
        'expected_1sigma': theo_1sigma,
        'expected_2sigma': theo_2sigma,
        'expected_3sigma': theo_3sigma,
    }


def create_diagnostic_plot(
    result: pd.DataFrame,
    reaction_name: str,
    save_path: Path,
) -> None:
    """Create diagnostic plot for a reaction."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{reaction_name} - Experiment Outlier Detection', fontsize=14)

    # Plot 1: Cross-section vs Energy colored by experiment
    ax1 = axes[0, 0]
    for exp_id in result['experiment_id'].unique():
        exp_data = result[result['experiment_id'] == exp_id]
        is_outlier = exp_data['experiment_outlier'].iloc[0]
        color = 'red' if is_outlier else None
        alpha = 1.0 if is_outlier else 0.3
        marker = 'x' if is_outlier else 'o'
        label = f'{exp_id} (outlier)' if is_outlier else None

        ax1.scatter(
            exp_data['Energy'],
            exp_data['CrossSection'],
            s=10, alpha=alpha, c=color, marker=marker, label=label,
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Cross Section (b)')
    ax1.set_title('Cross-Section by Experiment')
    if result['experiment_outlier'].any():
        ax1.legend(loc='upper right', fontsize=8)

    # Plot 2: GP fit with uncertainty bands
    ax2 = axes[0, 1]
    log_E = result['log_E'].values
    log_sigma = result['log_sigma'].values
    gp_mean = result['gp_mean'].values
    gp_std = result['gp_std'].values

    sorted_idx = np.argsort(log_E)
    ax2.scatter(log_E, log_sigma, s=5, alpha=0.3, label='Data')
    ax2.plot(log_E[sorted_idx], gp_mean[sorted_idx], 'r-', lw=1, label='GP Mean')
    ax2.fill_between(
        log_E[sorted_idx],
        gp_mean[sorted_idx] - 2*gp_std[sorted_idx],
        gp_mean[sorted_idx] + 2*gp_std[sorted_idx],
        alpha=0.2, color='red', label='±2σ'
    )
    ax2.set_xlabel('log10(Energy)')
    ax2.set_ylabel('log10(CrossSection)')
    ax2.set_title('GP Fit (log-log space)')
    ax2.legend()

    # Plot 3: Z-score distribution
    ax3 = axes[1, 0]
    valid_z = result['z_score'].values[np.isfinite(result['z_score'].values)]
    ax3.hist(np.abs(valid_z), bins=50, density=True, alpha=0.7, label='Empirical |z|')

    # Theoretical half-normal
    from scipy.stats import halfnorm
    x = np.linspace(0, 6, 100)
    ax3.plot(x, halfnorm.pdf(x), 'r-', lw=2, label='Half-normal')
    ax3.axvline(3.0, color='orange', linestyle='--', label='z=3 threshold')
    ax3.set_xlabel('|z-score|')
    ax3.set_ylabel('Density')
    ax3.set_title('Z-Score Distribution (Calibration Check)')
    ax3.legend()

    # Plot 4: Calibration curve
    ax4 = axes[1, 1]
    sigma_levels = np.linspace(0, 5, 50)
    empirical_coverage = [np.mean(np.abs(valid_z) <= s) for s in sigma_levels]

    from scipy.stats import norm
    theoretical_coverage = 2 * norm.cdf(sigma_levels) - 1

    ax4.plot(sigma_levels, empirical_coverage, 'b-', lw=2, label='Empirical')
    ax4.plot(sigma_levels, theoretical_coverage, 'r--', lw=2, label='Theoretical')
    ax4.fill_between(sigma_levels, 0, theoretical_coverage, alpha=0.1, color='red')
    ax4.set_xlabel('Sigma Level')
    ax4.set_ylabel('Coverage')
    ax4.set_title('Calibration Curve')
    ax4.legend()

    plt.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {save_path}")


def print_summary(
    reaction_name: str,
    outlier_rates: Dict,
    calibration: Dict,
    expected_exp_rate: float,
    expected_point_rate: float,
) -> bool:
    """Print summary and return True if validation passed."""
    print(f"\n{'='*60}")
    print(f"Reaction: {reaction_name}")
    print(f"{'='*60}")

    print(f"\nData Summary:")
    print(f"  Total points: {outlier_rates['n_total_points']:,}")
    print(f"  Experiments: {outlier_rates['n_experiments']}")

    print(f"\nOutlier Rates:")
    exp_rate = outlier_rates['exp_outlier_rate']
    point_rate = outlier_rates['point_outlier_rate']
    print(f"  Experiment outliers: {outlier_rates['n_exp_outliers']} ({exp_rate:.1%})")
    print(f"  Point outliers: {outlier_rates['n_point_outliers']} ({point_rate:.1%})")

    # Check thresholds
    exp_pass = exp_rate <= expected_exp_rate + 0.05  # Allow 5% margin
    point_pass = point_rate <= expected_point_rate + 0.05

    exp_status = "PASS" if exp_pass else "FAIL"
    point_status = "PASS" if point_pass else "FAIL"

    print(f"\n  Experiment outlier rate: {exp_status} (expected <= {expected_exp_rate:.0%})")
    print(f"  Point outlier rate: {point_status} (expected <= {expected_point_rate:.0%})")

    if outlier_rates['discrepant_experiments']:
        print(f"\n  Discrepant experiments: {', '.join(outlier_rates['discrepant_experiments'][:5])}")
        if len(outlier_rates['discrepant_experiments']) > 5:
            print(f"    ... and {len(outlier_rates['discrepant_experiments']) - 5} more")

    print(f"\nCalibration Metrics:")
    if calibration['n_valid'] > 0:
        print(f"  Mean |z|: {calibration['z_mean']:.2f} (expected ~0.8 for half-normal)")
        print(f"  Coverage at 1σ: {calibration['coverage_1sigma']:.1%} (expected {calibration['expected_1sigma']:.1%})")
        print(f"  Coverage at 2σ: {calibration['coverage_2sigma']:.1%} (expected {calibration['expected_2sigma']:.1%})")
        print(f"  Coverage at 3σ: {calibration['coverage_3sigma']:.1%} (expected {calibration['expected_3sigma']:.1%})")
    else:
        print("  No valid z-scores to evaluate")

    return exp_pass and point_pass


def main():
    parser = argparse.ArgumentParser(
        description='Validate ExperimentOutlierDetector on benchmark reactions'
    )
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='Path to EXFOR parquet file'
    )
    parser.add_argument(
        '--z-threshold', type=float, default=3.0,
        help='Z-score threshold for point outliers (default: 3.0)'
    )
    parser.add_argument(
        '--save-plots', type=str, default=None,
        help='Directory to save diagnostic plots (optional)'
    )
    parser.add_argument(
        '--no-wasserstein', action='store_true',
        help='Disable Wasserstein calibration (use marginal likelihood)'
    )
    parser.add_argument(
        '--reactions', type=str, nargs='+', default=None,
        help='Specific reactions to validate (e.g., "U-235(n,f)" "H-1(n,el)")'
    )

    args = parser.parse_args()

    # Load data
    df = load_data(args.data_path)

    # Filter to requested reactions
    if args.reactions:
        target_reactions = [r for r in TARGET_REACTIONS if r['name'] in args.reactions]
        if not target_reactions:
            logger.error(f"No matching reactions found. Available: {[r['name'] for r in TARGET_REACTIONS]}")
            sys.exit(1)
    else:
        target_reactions = TARGET_REACTIONS

    # Run validation
    results = []
    all_passed = True

    for reaction in target_reactions:
        name = reaction['name']
        logger.info(f"\nProcessing {name}...")

        # Filter data
        reaction_df = filter_reaction(df, reaction['Z'], reaction['A'], reaction['MT'])
        if len(reaction_df) == 0:
            logger.warning(f"No data found for {name}")
            continue

        logger.info(f"Found {len(reaction_df):,} points in {reaction_df['Entry'].nunique()} experiments")

        # Run detector
        scored_df, stats = run_detector(
            reaction_df,
            z_threshold=args.z_threshold,
            use_wasserstein=not args.no_wasserstein,
        )

        # Compute metrics
        outlier_rates = compute_outlier_rates(scored_df)
        calibration = compute_calibration_metrics(scored_df)

        # Print summary
        passed = print_summary(
            name,
            outlier_rates,
            calibration,
            reaction['expected_exp_outlier_rate'],
            reaction['expected_point_outlier_rate'],
        )
        all_passed = all_passed and passed

        # Save plot if requested
        if args.save_plots:
            plot_name = name.replace('(', '_').replace(')', '_').replace(',', '')
            plot_path = Path(args.save_plots) / f'{plot_name}_validation.png'
            create_diagnostic_plot(scored_df, name, plot_path)

        results.append({
            'reaction': name,
            'passed': passed,
            'exp_outlier_rate': outlier_rates['exp_outlier_rate'],
            'point_outlier_rate': outlier_rates['point_outlier_rate'],
        })

    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Reactions tested: {len(results)}")
    print(f"Passed: {sum(r['passed'] for r in results)}")
    print(f"Failed: {sum(not r['passed'] for r in results)}")

    if all_passed:
        print("\nAll validations PASSED!")
        sys.exit(0)
    else:
        print("\nSome validations FAILED!")
        for r in results:
            if not r['passed']:
                print(f"  - {r['reaction']}: exp={r['exp_outlier_rate']:.1%}, point={r['point_outlier_rate']:.1%}")
        sys.exit(1)


if __name__ == '__main__':
    main()
