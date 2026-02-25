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

    # With per-experiment GP outlier detection (RECOMMENDED)
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method experiment

    # With legacy SVGP outlier detection
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --outlier-method svgp

    # Full pipeline: test subset + per-experiment outlier detection
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset \\
        --outlier-method experiment --z-threshold 3.0

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

  # With SVGP outlier detection (Student-t likelihood, default)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --run-svgp

  # With SVGP using heteroscedastic likelihood (uses measurement uncertainties)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --run-svgp --svgp-likelihood heteroscedastic

  # Full pipeline: test subset + SVGP + CUDA acceleration
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --test-subset --run-svgp --svgp-device cuda

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
        choices=['svgp', 'experiment', None],
        help='Outlier detection method: '
             '"svgp" (legacy pooled SVGP), '
             '"experiment" (per-experiment GP with consensus, recommended). '
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
        help='Subsample large experiments to this many points for GP fitting '
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

    # Phase 1–4 GP enhancement flags (only used with --outlier-method experiment)
    parser.add_argument(
        '--smooth-mean',
        type=str,
        default='constant',
        choices=['constant', 'spline'],
        help='Mean function type for per-experiment GP (default: constant). '
             '"spline" fits a data-driven consensus trend before GP fitting.'
    )
    parser.add_argument(
        '--kernel-type',
        type=str,
        default='rbf',
        choices=['rbf', 'gibbs'],
        help='GP kernel type (default: rbf). '
             '"gibbs" uses a physics-informed nonstationary kernel based on '
             'RIPL-3 nuclear level density (requires --ripl-data-path).'
    )
    parser.add_argument(
        '--gibbs-lengthscale-source',
        type=str,
        default='data',
        choices=['data', 'ripl', 'auto'],
        help='Lengthscale source for Gibbs kernel (default: data). '
             '"data" computes from local residual variability (requires --smooth-mean spline). '
             '"ripl" uses RIPL-3 level density (requires --ripl-data-path). '
             '"auto" tries data first, falls back to RIPL-3.'
    )
    parser.add_argument(
        '--ripl-data-path',
        type=str,
        default=None,
        help='Path to RIPL-3 levels-param.data file (required for --kernel-type gibbs '
             'with --gibbs-lengthscale-source ripl). '
             'Without this, Gibbs kernel falls back to RBF.'
    )
    parser.add_argument(
        '--likelihood',
        type=str,
        default='gaussian',
        choices=['gaussian', 'contaminated'],
        help='GP likelihood type (default: gaussian). '
             '"contaminated" uses a contaminated normal mixture for principled '
             'outlier identification, producing per-point outlier_probability.'
    )
    parser.add_argument(
        '--hierarchical-refitting',
        action='store_true',
        default=False,
        help='Enable two-pass hierarchical fitting: Pass 1 fits independently, '
             'Pass 2 re-fits with group-informed constrained bounds and shared '
             'outputscale. Produces more consistent GP fits across experiments.'
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
    elif outlier_method == 'experiment':
        from nucml_next.data.experiment_outlier import ExperimentOutlierConfig
        from nucml_next.data.experiment_gp import ExactGPExperimentConfig

        # Validate CUDA availability before proceeding
        if args.svgp_device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    print(f"WARNING: --svgp-device cuda requested but CUDA is not available.")
                    print(f"         Falling back to CPU. Check your PyTorch installation.")
                    args.svgp_device = 'cpu'
                else:
                    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            except ImportError:
                print(f"WARNING: --svgp-device cuda requested but PyTorch not installed.")
                print(f"         Falling back to CPU.")
                args.svgp_device = 'cpu'

        # Warn about Gibbs kernel requirements
        if args.kernel_type == 'gibbs':
            ls_src = args.gibbs_lengthscale_source
            if ls_src in ('data', 'auto') and args.smooth_mean == 'constant':
                print("WARNING: --gibbs-lengthscale-source data requires --smooth-mean spline. "
                      "Will fall back to RIPL-3 or RBF.")
            if ls_src == 'ripl' and not args.ripl_data_path:
                print("WARNING: --gibbs-lengthscale-source ripl requires --ripl-data-path. "
                      "Gibbs kernel will fall back to RBF.")

        # Phase 1: Smooth mean config
        smooth_mean_config = None
        if args.smooth_mean == 'spline':
            from nucml_next.data.smooth_mean import SmoothMeanConfig
            smooth_mean_config = SmoothMeanConfig(smooth_mean_type='spline')

        # Phase 2: Kernel config
        kernel_config = None
        if args.kernel_type == 'gibbs':
            from nucml_next.data.kernels import KernelConfig
            kernel_config = KernelConfig(kernel_type='gibbs')

        # Phase 3: Likelihood config
        likelihood_config = None
        if args.likelihood == 'contaminated':
            from nucml_next.data.likelihood import LikelihoodConfig
            likelihood_config = LikelihoodConfig(likelihood_type='contaminated')

        # Build GP and detector configs with all phase settings
        gp_config = ExactGPExperimentConfig(
            device=args.svgp_device,
            max_gpu_points=args.max_gpu_points,
            max_subsample_points=args.max_subsample_points,
            smooth_mean_config=smooth_mean_config,
            kernel_config=kernel_config,
            likelihood_config=likelihood_config,
        )
        experiment_outlier_config = ExperimentOutlierConfig(
            gp_config=gp_config,
            point_z_threshold=args.z_threshold,
            checkpoint_dir=args.svgp_checkpoint_dir,
            ripl_data_path=args.ripl_data_path,
            gibbs_lengthscale_source=args.gibbs_lengthscale_source,
            hierarchical_refitting=args.hierarchical_refitting,
        )

    run_svgp = outlier_method == 'svgp'
    run_experiment_outlier = outlier_method == 'experiment'

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
    elif outlier_method == 'experiment':
        features = []
        if args.smooth_mean != 'constant':
            features.append(f"mean={args.smooth_mean}")
        if args.kernel_type != 'rbf':
            features.append(f"kernel={args.kernel_type}({args.gibbs_lengthscale_source})")
        if args.likelihood != 'gaussian':
            features.append(f"likelihood={args.likelihood}")
        if args.hierarchical_refitting:
            features.append("hierarchical")
        feat_str = f", features=[{', '.join(features)}]" if features else ""
        print(f"Outlier:      Per-experiment GP - device={args.svgp_device}, z_threshold={args.z_threshold}{feat_str}")
    else:
        print(f"Outlier:      Disabled (use --outlier-method to enable)")
    if args.svgp_checkpoint_dir:
        print(f"Checkpoints:  {args.svgp_checkpoint_dir}")
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

    # Run per-experiment outlier detection if requested
    if run_experiment_outlier:
        print("\n" + "-"*70)
        print("Running per-experiment GP outlier detection...")
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
        print(f"[OK] Per-experiment outlier detection:")
        print(f"    Point outliers: {n_point_outliers:,} ({100*n_point_outliers/len(df):.2f}%)")
        print(f"    Experiment outliers: {n_exp_outliers} experiments flagged")
    else:
        print(f"[OK] Lean Parquet contains EXFOR data only (no AME duplication)")
    print(f"\n[NOTE] AME2020/NUBASE2020 enrichment will be added during feature generation")
    print(f"       Place AME *.mas20.txt files in data/ directory")
    print()


if __name__ == '__main__':
    main()
