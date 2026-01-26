#!/usr/bin/env python3
"""
Quick test to verify that AME enrichment doesn't duplicate rows.

Expected behavior:
- EXFOR data should have ~16.9M rows
- After AME enrichment, should still have ~16.9M rows (no duplication)
- Coverage should be ~89% (measurements with AME data available)
"""

import sys
from pathlib import Path
from nucml_next.data.dataset import NucmlDataset
from nucml_next.data.selection import DataSelection

def test_enrichment_no_duplication():
    """Test that AME enrichment preserves row count."""

    print("=" * 80)
    print("Testing AME Enrichment Fix")
    print("=" * 80)

    # Create selection that requires AME enrichment (Tier C)
    selection = DataSelection(
        projectile='neutron',
        mt_mode='all_physical',
        tiers=['A', 'B', 'C', 'D'],  # Tiers B-D require AME
        drop_invalid=True
    )

    # Load dataset with AME enrichment
    print("\nLoading EXFOR data with AME enrichment...")
    data_path = Path('../data/exfor_processed.parquet')
    if not data_path.exists():
        data_path = Path('data/exfor_processed.parquet')

    if not data_path.exists():
        print("❌ ERROR: EXFOR data not found at data/exfor_processed.parquet")
        print("   Please run from repository root or notebook directory")
        sys.exit(1)

    dataset = NucmlDataset(
        data_path=str(data_path),
        mode='tabular',
        selection=selection
    )

    # Check row count
    num_rows = len(dataset.df)
    print(f"\n✓ Final row count: {num_rows:,}")

    # Expected: ~16.9M rows (no duplication)
    # If duplicated, would be ~28-30M rows
    if num_rows > 20_000_000:
        print(f"❌ FAIL: Row count ({num_rows:,}) is too high - likely duplicated!")
        print("   Expected: ~16.9M rows")
        print("   Got: {num_rows:,} rows")
        sys.exit(1)
    elif num_rows < 10_000_000:
        print(f"⚠️  WARNING: Row count ({num_rows:,}) is lower than expected")
        print("   Expected: ~16.9M rows")
        print("   This might indicate excessive filtering")
    else:
        print(f"✓ PASS: Row count is within expected range (10M - 20M)")

    # Check AME enrichment coverage
    if 'Mass_Excess_keV' in dataset.df.columns:
        n_enriched = dataset.df['Mass_Excess_keV'].notna().sum()
        coverage = 100 * n_enriched / len(dataset.df)
        print(f"✓ AME coverage: {n_enriched:,} / {num_rows:,} ({coverage:.1f}%)")

        if coverage < 80:
            print(f"⚠️  WARNING: AME coverage ({coverage:.1f}%) is lower than expected (~89%)")
    else:
        print("⚠️  WARNING: Mass_Excess_keV column not found - AME enrichment may have failed")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully - no row duplication detected!")
    print("=" * 80)

if __name__ == "__main__":
    test_enrichment_no_duplication()
