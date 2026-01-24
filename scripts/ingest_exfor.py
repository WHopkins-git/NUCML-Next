#!/usr/bin/env python
"""
EXFOR Data Ingestion Script with Full AME2020/NUBASE2020 Enrichment
====================================================================

Ingest X4Pro SQLite database to partitioned Parquet with ALL tier enrichment.

This script implements the pre-enrichment architecture:
- Loads ALL AME2020/NUBASE2020 files (5 files) during ingestion
- Writes complete enrichment schema to Parquet
- Feature selection becomes simple column selection (no joins needed)

Usage:
    # Basic ingestion (no enrichment)
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

    # Full enrichment (recommended - all tiers)
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --ame2020-dir data/

Requirements:
    - X4Pro SQLite database (x4sqlite1.db)
      Download from: https://www-nds.iaea.org/x4/

    - (Optional but recommended) AME2020/NUBASE2020 data files in directory:
      * mass_1.mas20.txt (Tier B, C)
      * rct1.mas20.txt (Tier C, E)
      * rct2_1.mas20.txt (Tier C, E)
      * nubase_4.mas20.txt (Tier D)
      * covariance.mas20.txt (optional)

      Download: wget https://www-nds.iaea.org/amdc/ame2020/*.mas20.txt

Author: NUCML-Next Team
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.ingest import ingest_x4


def main():
    parser = argparse.ArgumentParser(
        description="Ingest X4Pro SQLite database to Parquet with full AME2020/NUBASE2020 enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion (no enrichment)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

  # With custom output path
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --output data/exfor_enriched.parquet

  # With full AME2020/NUBASE2020 enrichment (recommended - all tiers)
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --ame2020-dir data/

  # The ame2020-dir should contain:
  #   - mass_1.mas20.txt (Tier B, C)
  #   - rct1.mas20.txt (Tier C, E)
  #   - rct2_1.mas20.txt (Tier C, E)
  #   - nubase_4.mas20.txt (Tier D)
  #   - covariance.mas20.txt (optional)
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
        help='Directory containing AME2020/NUBASE2020 *.mas20.txt files (optional, for full tier enrichment)'
    )

    args = parser.parse_args()

    # Validate X4 database
    x4_path = Path(args.x4_db)
    if not x4_path.exists():
        print(f"❌ Error: X4 database not found: {x4_path}")
        print("\nPlease provide a valid X4Pro SQLite database.")
        print("  - Full database: Download from https://www-nds.iaea.org/x4/")
        print("  - Sample database: Use data/x4sqlite1_sample.db (in repository)")
        sys.exit(1)

    # Validate AME2020 directory if provided
    if args.ame2020_dir:
        ame_path = Path(args.ame2020_dir)
        if not ame_path.exists():
            print(f"❌ Error: AME2020 directory not found: {ame_path}")
            sys.exit(1)

    # Run ingestion
    print("\n" + "="*70)
    print("NUCML-Next: X4Pro EXFOR Data Ingestion")
    print("="*70)
    print(f"X4 Database:  {args.x4_db}")
    print(f"Output:       {args.output}")
    print(f"AME2020 Dir:  {args.ame2020_dir or 'None (no enrichment)'}")
    if args.ame2020_dir:
        print("  → Will load all *.mas20.txt files for tier-based enrichment")
    print("="*70 + "\n")

    df = ingest_x4(
        x4_db_path=args.x4_db,
        output_path=args.output,
        ame2020_dir=args.ame2020_dir,
    )

    print(f"\n✓ Ingestion complete!")
    print(f"✓ Processed {len(df):,} data points")
    print(f"✓ Saved to: {args.output}")
    if args.ame2020_dir:
        print(f"✓ Parquet now contains ALL enrichment columns for tier-based feature selection")
    print()


if __name__ == '__main__':
    main()
