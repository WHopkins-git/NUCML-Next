"""
Examples Helper Functions
=========================

Minimal convenience wrappers for common operations in notebooks.

These functions are intentionally simple and should not contain business logic.
"""

from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd


def load_sample_db_path() -> str:
    """
    Get path to sample X4 database in repository.

    Returns:
        Path to data/x4sqlite1_sample.db

    Raises:
        FileNotFoundError: If sample database not found
    """
    sample_path = Path('data/x4sqlite1_sample.db')

    # Try relative to current directory
    if sample_path.exists():
        return str(sample_path)

    # Try relative to repository root
    repo_root = Path(__file__).parent.parent.parent
    sample_path = repo_root / 'data' / 'x4sqlite1_sample.db'

    if sample_path.exists():
        return str(sample_path)

    raise FileNotFoundError(
        "Sample database not found: data/x4sqlite1_sample.db\n"
        "Please ensure the sample database is present in the repository."
    )


def quick_ingest(
    x4_db_path: Optional[str] = None,
    output_path: str = 'data/exfor_processed.parquet',
    ame2020_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quick ingestion with sensible defaults.

    Args:
        x4_db_path: Path to X4 database. If None, uses sample database.
        output_path: Output Parquet path
        ame2020_path: Optional AME2020 file

    Returns:
        Ingested DataFrame

    Example:
        >>> from nucml_next.examples import quick_ingest
        >>> df = quick_ingest()  # Uses sample DB
        >>> print(f"Loaded {len(df)} points")
    """
    from nucml_next.ingest import ingest_x4

    if x4_db_path is None:
        x4_db_path = load_sample_db_path()
        print(f"Using sample database: {x4_db_path}")

    return ingest_x4(
        x4_db_path=x4_db_path,
        output_path=output_path,
        ame2020_path=ame2020_path,
    )


def load_dataset(
    data_path: str = 'data/exfor_processed.parquet',
    mode: str = 'tabular',
    filters: Optional[Dict[str, List]] = None,
) -> 'NucmlDataset':
    """
    Load NucmlDataset with sensible defaults.

    Args:
        data_path: Path to Parquet dataset
        mode: 'tabular' or 'graph'
        filters: Optional filters dict (e.g., {'Z': [92], 'MT': [18]})

    Returns:
        NucmlDataset instance

    Example:
        >>> from nucml_next.examples import load_dataset
        >>> # Load U-235 and Cl-35 data
        >>> dataset = load_dataset(
        ...     filters={'Z': [92, 17], 'A': [235, 35]}
        ... )
    """
    from nucml_next.data import NucmlDataset

    return NucmlDataset(
        data_path=data_path,
        mode=mode,
        filters=filters,
    )


def print_dataset_summary(dataset: 'NucmlDataset'):
    """
    Print a formatted summary of dataset contents.

    Args:
        dataset: NucmlDataset instance

    Example:
        >>> from nucml_next.examples import load_dataset, print_dataset_summary
        >>> dataset = load_dataset()
        >>> print_dataset_summary(dataset)
    """
    df = dataset.df

    print("="*70)
    print("Dataset Summary")
    print("="*70)
    print(f"Total data points:     {len(df):,}")
    print(f"Unique isotopes:       {df[['Z', 'A']].drop_duplicates().shape[0]}")
    print(f"Unique reactions (MT): {df['MT'].nunique()}")
    print(f"Energy range:          {df['Energy'].min():.2e} - {df['Energy'].max():.2e} eV")

    if 'Uncertainty' in df.columns:
        with_unc = df['Uncertainty'].notna().sum()
        print(f"Points with uncertainty: {with_unc:,} ({100*with_unc/len(df):.1f}%)")

    if 'Mass_Excess_keV' in df.columns:
        with_ame = df['Mass_Excess_keV'].notna().sum()
        print(f"AME2020 enriched:      {with_ame:,} ({100*with_ame/len(df):.1f}%)")

    print("\nIsotope breakdown:")
    for (z, a), group in df.groupby(['Z', 'A']):
        # Determine element symbol (simplified)
        element_map = {1: 'H', 6: 'C', 8: 'O', 17: 'Cl', 92: 'U', 94: 'Pu'}
        element = element_map.get(z, '??')
        isotope_name = f"{element}-{a}" if a > 0 else f"{element}-NAT"

        print(f"  {isotope_name:8s} (Z={z:2d}, A={a:3d}): {len(group):>6,} points")

        # Show reactions
        for mt in sorted(group['MT'].unique()):
            mt_name = {2: 'Elastic', 18: 'Fission', 102: 'Capture', 103: '(n,p)', 16: '(n,2n)'}.get(int(mt), f'MT={mt}')
            mt_count = len(group[group['MT'] == mt])
            print(f"    └─ {mt_name:12s}: {mt_count:>6,} points")

    print("="*70)
