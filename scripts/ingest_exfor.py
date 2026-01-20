#!/usr/bin/env python
"""
EXFOR Data Ingestion Script
============================

This script ingests the IAEA EXFOR-X5json bulk database and creates
a partitioned Parquet dataset for NUCML-Next.

Usage:
    python scripts/ingest_exfor.py --exfor-root ~/data/EXFOR-X5json/ --output data/exfor_processed.parquet

Requirements:
    - Download EXFOR-X5json bulk database from: https://www-nds.iaea.org/exfor/
    - Unzip to a local directory
    - (Optional) Download AME2020 data from: https://www-nds.iaea.org/amdc/

Author: NUCML-Next Team
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.data import ingest_exfor


def main():
    parser = argparse.ArgumentParser(
        description="Ingest EXFOR-X5json database to Parquet"
    )

    parser.add_argument(
        '--exfor-root',
        type=str,
        required=True,
        help='Path to unzipped EXFOR-X5json directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/exfor_processed.parquet',
        help='Output path for partitioned Parquet dataset'
    )

    parser.add_argument(
        '--ame2020',
        type=str,
        default=None,
        help='Path to AME2020 data file (optional)'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )

    args = parser.parse_args()

    # Validate EXFOR root
    exfor_path = Path(args.exfor_root)
    if not exfor_path.exists():
        print(f"❌ Error: EXFOR root not found: {exfor_path}")
        print("\nPlease download EXFOR-X5json from:")
        print("  https://www-nds.iaea.org/exfor/")
        sys.exit(1)

    # Run ingestion
    print("\n" + "="*70)
    print("NUCML-Next: EXFOR Data Ingestion")
    print("="*70)
    print(f"EXFOR root: {args.exfor_root}")
    print(f"Output:     {args.output}")
    print(f"AME2020:    {args.ame2020 or 'Using SEMF approximation'}")
    if args.max_files:
        print(f"Max files:  {args.max_files} (testing mode)")
    print("="*70 + "\n")

    df = ingest_exfor(
        exfor_root=args.exfor_root,
        output_path=args.output,
        ame2020_path=args.ame2020,
        max_files=args.max_files,
    )

    print(f"\n✓ Ingestion complete!")
    print(f"✓ Processed {len(df)} data points")
    print(f"✓ Saved to: {args.output}")

    # Print sample
    print(f"\nSample data:")
    print(df.head(10))


if __name__ == '__main__':
    main()
