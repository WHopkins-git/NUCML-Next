#!/usr/bin/env python
"""
EXFOR Data Ingestion Script
============================

Ingest X4Pro SQLite database to partitioned Parquet for NUCML-Next.

Usage:
    python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --output data/exfor_processed.parquet

Requirements:
    - X4Pro SQLite database (x4sqlite1.db)
    - Download from: https://www-nds.iaea.org/x4/
    - (Optional) AME2020 data for isotopic enrichment

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
        description="Ingest X4Pro SQLite database to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db

  # With custom output path
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --output /tmp/exfor.parquet

  # With AME2020 enrichment
  python scripts/ingest_exfor.py --x4-db data/x4sqlite1.db --ame2020 data/ame2020.txt
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
        '--ame2020',
        type=str,
        default=None,
        help='Path to AME2020 mass file (optional, for isotopic enrichment)'
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

    # Run ingestion
    print("\n" + "="*70)
    print("NUCML-Next: X4Pro EXFOR Data Ingestion")
    print("="*70)
    print(f"X4 Database: {args.x4_db}")
    print(f"Output:      {args.output}")
    print(f"AME2020:     {args.ame2020 or 'Using SEMF approximation'}")
    print("="*70 + "\n")

    df = ingest_x4(
        x4_db_path=args.x4_db,
        output_path=args.output,
        ame2020_path=args.ame2020,
    )

    print(f"\n✓ Ingestion complete!")
    print(f"✓ Processed {len(df):,} data points")
    print(f"✓ Saved to: {args.output}\n")


if __name__ == '__main__':
    main()
