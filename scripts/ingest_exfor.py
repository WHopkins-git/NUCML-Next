#!/usr/bin/env python
"""
EXFOR Data Ingestion Script - Lean Extraction
==============================================

Ingest X4Pro SQLite database to partitioned Parquet (EXFOR data only).

This script implements the lean ingestion architecture:
- Extracts ONLY EXFOR experimental cross-section measurements
- Produces compact Parquet files without AME data duplication
- AME2020/NUBASE2020 enrichment happens during feature generation

Usage:
    # Standard ingestion
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

    # With custom output path
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --output data/my_exfor.parquet

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
  # Standard ingestion
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

  # With custom output path
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --output data/exfor.parquet

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

    args = parser.parse_args()

    # Validate X4 database
    x4_path = Path(args.x4_db)
    if not x4_path.exists():
        print(f"❌ Error: X4 database not found: {x4_path}")
        print("\nPlease provide a valid X4Pro SQLite database.")
        print("  - Full database: Download from https://www-nds.iaea.org/x4/")
        print("  - Sample database: Use data/x4sqlite1_sample.db (in repository)")
        sys.exit(1)

    # Warn if --ame2020-dir provided (deprecated parameter)
    if args.ame2020_dir:
        print("\n⚠️  WARNING: --ame2020-dir is deprecated and will be ignored.")
        print("   AME enrichment now happens during feature generation for better performance.")
        print("   Place AME files in data/ and NucmlDataset will load them automatically.\n")

    # Run ingestion
    print("\n" + "="*70)
    print("NUCML-Next: X4Pro EXFOR Data Ingestion (Lean Mode)")
    print("="*70)
    print(f"X4 Database:  {args.x4_db}")
    print(f"Output:       {args.output}")
    print(f"Mode:         Lean extraction (EXFOR data only)")
    print("="*70 + "\n")

    df = ingest_x4(
        x4_db_path=args.x4_db,
        output_path=args.output,
        ame2020_dir=None,  # Always None now (enrichment happens during feature generation)
    )

    print(f"\n✓ Ingestion complete!")
    print(f"✓ Processed {len(df):,} data points")
    print(f"✓ Saved to: {args.output}")
    print(f"✓ Lean Parquet contains EXFOR data only (no AME duplication)")
    print(f"\n💡 AME2020/NUBASE2020 enrichment will be added during feature generation")
    print(f"   Place AME *.mas20.txt files in data/ directory")
    print()


if __name__ == '__main__':
    main()
