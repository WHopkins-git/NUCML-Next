#!/usr/bin/env python
# =============================================================================
# CRITICAL: Thread limits must be set BEFORE importing numpy/torch/scipy
# These libraries read environment variables at import time
# =============================================================================
import os as _os
import multiprocessing as _mp

# Auto-detect 50% of CPU cores (shared machine etiquette)
_default_threads = max(1, _mp.cpu_count() // 2)
_num_threads = _os.environ.get('NUCML_NUM_THREADS', str(_default_threads))
_os.environ.setdefault('OMP_NUM_THREADS', _num_threads)
_os.environ.setdefault('MKL_NUM_THREADS', _num_threads)
_os.environ.setdefault('OPENBLAS_NUM_THREADS', _num_threads)
_os.environ.setdefault('NUMEXPR_NUM_THREADS', _num_threads)
# =============================================================================

"""
EXFOR Data Ingestion Script - Lean Extraction
==============================================

Ingest X4Pro SQLite database to partitioned Parquet (EXFOR data only).

This script implements the lean ingestion architecture:
- Extracts ONLY EXFOR experimental cross-section measurements
- Produces compact Parquet files without AME data duplication
- AME2020/NUBASE2020 enrichment happens during feature generation

Usage:
    # Standard ingestion (full database, ~13M points)
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

    # Test subset (Uranium + Chlorine only, ~300K points, takes minutes)
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

    # Custom element subset
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --z-filter 79,92,26

    # With smooth mean + local MAD outlier detection (RECOMMENDED)
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method local_mad

    # With legacy SVGP outlier detection
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp

Requirements:
    - X4Pro SQLite database (x4sqlite1.db)
      Download from: https://www-nds.iaea.org/x4/

Note:
    AME2020/NUBASE2020 enrichment is now handled automatically during feature generation.
    Download AME files separately and place in data/ directory:
      - mass_1.mas20.txt, rct1.mas20.txt, rct2_1.mas20.txt
      - nubase_4.mas20.txt, covariance.mas20.txt
    Download: wget https://www-nds.iaea.org/amdc/ame2020/*.mas20.txt

    NucmlDataset will load AME files automatically when tiers=['C'], ['D'], or ['E'] are requested.

Author: NUCML-Next Team
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.ingest import ingest_x4


def main():
    parser = argparse.ArgumentParser(
        description="Ingest X4Pro SQLite database to lean Parquet (EXFOR data only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ingestion (full database)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

  # Test subset (Uranium + Chlorine only, much faster)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset

  # Custom element subset (Gold, Uranium, Iron)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --z-filter 79,92,26

  # With local MAD outlier detection (RECOMMENDED)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method local_mad

  # With legacy SVGP outlier detection
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp

Note:
  AME2020/NUBASE2020 enrichment is now handled during feature generation.
  Place AME files in data/ directory and NucmlDataset will load them automatically.
"""
    )

    parser.add_argument(
        '--x4-db',
        type=str,
        required=True,
        help='Path to X4Pro SQLite database (e.g., x4sqlite1.db)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/exfor_processed.parquet',
        help='Output path for partitioned Parquet dataset (default: data/exfor_processed.parquet)'
    )

    parser.add_argument(
        '--ame2020-dir',
        type=str,
        default=None,
        help='DEPRECATED - This parameter is ignored. AME enrichment now happens during feature generation.'
    )

    # Outlier detection options
    parser.add_argument(
        '--outlier-method',
        type=str,
        default=None,
        choices=['svgp', 'local_mad', None],
        help='Outlier detection method: '
             '"local_mad" (smooth mean + local MAD, recommended), '
             '"svgp" (legacy pooled SVGP). '
             'Default: None (no outlier detection)'
    )
    parser.add_argument(
        '--run-svgp',
        action='store_true',
        default=False,
        help='DEPRECATED: Use --outlier-method svgp instead. '
             'Run SVGP outlier detection (adds z_score column to Parquet)'
    )
    parser.add_argument(
        '--no-svgp',
        action='store_true',
        default=False,
        help='Explicitly skip outlier detection (default behavior)'
    )
    parser.add_argument(
        '--svgp-device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for SVGP computation (default: cpu)'
    )
    parser.add_argument(
        '--max-gpu-points',
        type=int,
        default=40000,
        help='Max points per experiment on GPU; larger experiments use CPU '
             '(default: 40000). Memory: n^2 * 8 bytes, so 40k pts = 12.8GB'
    )
    parser.add_argument(
        '--max-subsample-points',
        type=int,
        default=15000,
        help='Subsample large experiments to this many points for SVGP fitting '
             '(default: 15000). Predictions still made on all points.'
    )
    parser.add_argument(
        '--svgp-checkpoint-dir',
        type=str,
        default=None,
        help='Directory for outlier detection checkpoints (enables resume on interruption)'
    )
    parser.add_argument(
        '--svgp-likelihood',
        type=str,
        default='student_t',
        choices=['student_t', 'heteroscedastic', 'gaussian'],
        help='SVGP likelihood type (default: student_t). Options: '
             'student_t (robust to outliers), '
             'heteroscedastic (uses measurement uncertainties), '
             'gaussian (legacy)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for flagging point outliers (default: 3.0)'
    )
    parser.add_argument(
        '--exp-z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for counting bad points in experiment discrepancy '
             'detection (default: 3.0). Only used with --outlier-method local_mad.'
    )
    parser.add_argument(
        '--exp-fraction-threshold',
        type=float,
        default=0.30,
        help='Fraction of bad points to flag an experiment as discrepant '
             '(default: 0.30 = 30%%). Only used with --outlier-method local_mad.'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=-1,
        help='Parallel workers for outlier detection. '
             '-1 = auto (half CPU cores), 0 = all cores. Default: -1.'
    )

    import multiprocessing
    default_threads = max(1, multiprocessing.cpu_count() // 2)
    parser.add_argument(
        '--num-threads',
        type=int,
        default=default_threads,
        help=f'CPU threads for NumPy/PyTorch linear algebra (default: {default_threads} = 50%% of cores). '
             'Auto-detects half of available cores for shared machine etiquette.'
    )

    # Subset filtering options
    parser.add_argument(
        '--test-subset',
        action='store_true',
        default=False,
        help='Use test subset: Uranium (Z=92) + Chlorine (Z=17) only. '
             'Much faster for development/testing (~300K points instead of 13M)'
    )
    parser.add_argument(
        '--z-filter',
        type=str,
        default=None,
        help='Comma-separated list of atomic numbers (Z) to include. '
             'Example: --z-filter 79,92,26 for Au, U, Fe'
    )

    # Metadata filtering options
    parser.add_argument(
        '--include-non-pure',
        action='store_true',
        default=False,
        help='Include non-pure data (relative, ratio, spectrum-averaged, '
             'non-XS quantities, calculated/derived). '
             'By default these are EXCLUDED because they have different '
             'units or meaning than point-wise absolute cross sections.'
    )
    parser.add_argument(
        '--include-superseded',
        action='store_true',
        default=False,
        help='Include superseded entries (SPSDD flag in SUBENT table). '
             'By default these are EXCLUDED.'
    )

    # Diagnostic mode
    parser.add_argument(
        '--diagnostics',
        action='store_true',
        default=False,
        help='Extract additional metadata (Author, Year, ReactionType, FullCode, NDataPoints) '
             'for interactive inspection. Use with Diagnostics_Interactive_Inspector notebook.'
    )

    args = parser.parse_args()

    # Apply thread limits dynamically (supplements the env vars set at module load)
    # This allows CLI --num-threads to override NUCML_NUM_THREADS env var
    import os
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_threads)

    # Also set torch thread limit if available
    try:
        import torch
        torch.set_num_threads(args.num_threads)
    except ImportError:
        pass

    # Parse z_filter
    z_filter = None
    if args.test_subset:
        z_filter = [17, 92]  # Chlorine and Uranium
    elif args.z_filter:
        try:
            z_filter = [int(z.strip()) for z in args.z_filter.split(',')]
        except ValueError:
            print(f"Error: Invalid --z-filter format. Expected comma-separated integers.")
            print(f"  Got: {args.z_filter}")
            print(f"  Example: --z-filter 79,92,26")
            sys.exit(1)

    # Validate X4 database
    x4_path = Path(args.x4_db)
    if not x4_path.exists():
        print(f"ERROR: X4 database not found: {x4_path}")
        print("\nPlease provide a valid X4Pro SQLite database.")
        print("  - Full database: Download from https://www-nds.iaea.org/x4/")
        print("  - Sample database: Use data/x4sqlite1_sample.db (in repository)")
        sys.exit(1)

    # Warn if --ame2020-dir provided (deprecated parameter)
    if args.ame2020_dir:
        print("\nWARNING: --ame2020-dir is deprecated and will be ignored.")
        print("   AME enrichment now happens during feature generation for better performance.")
        print("   Place AME files in data/ and NucmlDataset will load them automatically.\n")

    # Determine outlier detection method
    outlier_method = args.outlier_method
    if args.run_svgp and not args.no_svgp and outlier_method is None:
        # Legacy --run-svgp flag
        outlier_method = 'svgp'
        print("WARNING: --run-svgp is deprecated. Use --outlier-method svgp instead.\n")

    # Build outlier detector config
    svgp_config = None
    experiment_outlier_config = None

    if outlier_method == 'svgp':
        from nucml_next.data.outlier_detection import SVGPConfig
        svgp_config = SVGPConfig(
            device=args.svgp_device,
            checkpoint_dir=args.svgp_checkpoint_dir,
            likelihood=args.svgp_likelihood,
        )

    elif outlier_method == 'local_mad':
        from nucml_next.data.experiment_outlier import (
            ExperimentOutlierConfig, ExperimentOutlierDetector,
        )

        experiment_outlier_config = ExperimentOutlierConfig(
            point_z_threshold=args.z_threshold,
            exp_z_threshold=args.exp_z_threshold,
            exp_fraction_threshold=args.exp_fraction_threshold,
            n_workers=args.n_workers,  # -1 = auto (half cores), 0 = all, N = exact
        )

    run_svgp = outlier_method == 'svgp'
    run_experiment_outlier = outlier_method == 'local_mad'

    # Run ingestion
    print("\n" + "="*70)
    print("NUCML-Next: X4Pro EXFOR Data Ingestion (Lean Mode)")
    print("="*70)
    print(f"X4 Database:  {args.x4_db}")
    print(f"Output:       {args.output}")
    print(f"Mode:         Lean extraction (EXFOR data only)")
    if z_filter:
        print(f"Z Filter:     {z_filter} (subset mode)")
    if outlier_method == 'svgp':
        print(f"Outlier:      SVGP (legacy) - device={args.svgp_device}, likelihood={args.svgp_likelihood}")
    elif outlier_method == 'local_mad':
        _resolved = ExperimentOutlierDetector._resolve_n_workers(args.n_workers)
        print(f"Outlier:      Smooth mean + local MAD")
        print(f"              Point threshold: z > {args.z_threshold}")
        print(f"              Experiment: >{args.exp_fraction_threshold:.0%} of points with z > {args.exp_z_threshold}")
        print(f"              Workers: {_resolved} (from {multiprocessing.cpu_count()} logical cores)")
    else:
        print(f"Outlier:      Disabled (use --outlier-method to enable)")
    # Metadata filtering status
    exclude_non_pure = not args.include_non_pure
    exclude_superseded = not args.include_superseded
    if exclude_non_pure or exclude_superseded:
        filters = []
        if exclude_non_pure:
            filters.append("non-pure data")
        if exclude_superseded:
            filters.append("superseded entries")
        print(f"Filtering:    Excluding {', '.join(filters)}")
    else:
        print(f"Filtering:    Disabled (all data included)")
    if args.diagnostics:
        print(f"Diagnostics:  ON (Author, Year, ReactionType, FullCode, NDataPoints)")

    print(f"CPU Threads:  {args.num_threads} (50% of {multiprocessing.cpu_count()} cores)")
    print("="*70 + "\n")

    df = ingest_x4(
        x4_db_path=args.x4_db,
        output_path=args.output,
        ame2020_dir=None,  # Always None now (enrichment happens during feature generation)
        run_svgp=run_svgp,
        svgp_config=svgp_config,
        z_filter=z_filter,
        exclude_non_pure=exclude_non_pure,
        exclude_superseded=exclude_superseded,
        diagnostics=args.diagnostics,
    )

    # Run local MAD outlier detection if requested
    if run_experiment_outlier:
        print("\n" + "-"*70)
        print("Running local MAD outlier detection...")
        print("-"*70 + "\n")

        from nucml_next.data.experiment_outlier import ExperimentOutlierDetector

        detector = ExperimentOutlierDetector(experiment_outlier_config)
        df = detector.score_dataframe(df)

        # Save updated dataframe
        # Handle case where output_path is a directory (from X4Ingestor's write_to_dataset)
        output_path = Path(args.output)
        if output_path.is_dir():
            print(f"  Removing existing directory: {output_path}")
            shutil.rmtree(output_path)
        df.to_parquet(args.output, index=False)

    print(f"\n[OK] Ingestion complete!")
    print(f"[OK] Processed {len(df):,} data points")
    print(f"[OK] Saved to: {args.output}")

    if run_svgp and 'z_score' in df.columns:
        n_outliers = (df['z_score'] > args.z_threshold).sum()
        print(f"[OK] SVGP outlier detection: {n_outliers:,} outliers at z>{args.z_threshold} ({100*n_outliers/len(df):.2f}%)")
    elif run_experiment_outlier and 'z_score' in df.columns:
        n_point_outliers = df['point_outlier'].sum() if 'point_outlier' in df.columns else 0
        n_exp_outliers = df[df['experiment_outlier']]['experiment_id'].nunique() if 'experiment_outlier' in df.columns else 0
        print(f"[OK] Local MAD outlier detection:")
        print(f"    Point outliers: {n_point_outliers:,} ({100*n_point_outliers/len(df):.2f}%)")
        print(f"    Experiment outliers: {n_exp_outliers} experiments flagged")
    else:
        print(f"[OK] Lean Parquet contains EXFOR data only (no AME duplication)")
    print(f"\n[NOTE] AME2020/NUBASE2020 enrichment will be added during feature generation")
    print(f"       Place AME *.mas20.txt files in data/ directory")
    print()


if __name__ == '__main__':
    main()
