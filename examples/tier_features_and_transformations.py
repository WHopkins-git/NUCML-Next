"""
Example: Tier-Based Features with Transformation Pipeline
==========================================================

Demonstrates the complete ML workflow for nuclear cross-section prediction:

1. **Isomer Fallback**: Isomeric states inherit AME properties from ground states
2. **Tier-Based Features**: Hierarchical feature engineering (A → E)
3. **Transformation Pipeline**: Log-scaling and standardization
4. **Inverse Transforms**: Convert predictions back to physical units

Features by Tier:
-----------------
- **Tier A (Core)**: Z, A, N, Energy + 9-feature Particle Vector
  - Particle vector: [out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met]
  - 13 features total

- **Tier B (Geometric)**: Nuclear Radius and kR parameter
  - R = r₀ A^(1/3) where r₀ = 1.25 fm
  - kR = k * R (dimensionless interaction parameter)
  - +2 features (15 total)

- **Tier C (Energetics)**: Mass, Binding, and Separation Energies
  - Mass_Excess_MeV, Binding_Energy_MeV, Binding_Per_Nucleon_MeV
  - S_1n_MeV, S_2n_MeV, S_1p_MeV, S_2p_MeV
  - +7 features (22 total)

- **Tier D (Topological)**: Shell structure and nuclear topology
  - Valence_N, Valence_P: Distance to magic numbers (2, 8, 20, 28, 50, 82, 126)
  - P_Factor: Promiscuity Factor = N_p * N_n / (N_p + N_n)
  - Spin, Parity, Shell_Closure_N, Shell_Closure_P
  - +8 features (30 total)

Transformations:
----------------
1. **Log-scaling**:
   - Cross-section: σ' = log₁₀(σ + 10⁻¹⁰)
   - Energy: E' = log₁₀(E)

2. **Standardization** (Z-score):
   - X' = (X - μ) / σ for Z, A, N and Tier B-D features

3. **Inverse Transforms**:
   - σ = 10^(σ') - 10⁻¹⁰
   - E = 10^(E')
   - X = X' * σ + μ

Usage:
------
    python examples/tier_features_and_transformations.py --data-path data/exfor_processed.parquet
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.data.dataset import NucmlDataset
from nucml_next.data.selection import DataSelection
from nucml_next.data.transformations import TransformationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_tier_features():
    """Demonstrate tier-based feature generation with new Promiscuity Factor."""

    logger.info("\n" + "="*80)
    logger.info("TIER-BASED FEATURE DEMONSTRATION")
    logger.info("="*80)

    # Check if data exists
    data_path = Path('data/exfor_processed.parquet')
    if not data_path.exists():
        logger.error(f"❌ Data not found at {data_path}")
        logger.info("Please run: python scripts/ingest_exfor.py --exfor-root /path/to/EXFOR")
        return None

    # Load dataset with Tier C features (includes A, B, C automatically)
    logger.info("\n📊 Loading dataset with Tier A, B, C features...")
    selection = DataSelection(
        projectile='neutron',
        mt_mode='reactor_core',  # Essential reactions: elastic, fission, capture, etc.
        energy_min=1e-5,  # Thermal energies
        energy_max=2e7,   # Up to 20 MeV
        tiers=['A', 'B', 'C', 'D'],  # Include all tiers through D
        drop_invalid=True
    )

    dataset = NucmlDataset(
        data_path=str(data_path),
        mode='tabular',
        selection=selection
    )

    # Generate tier-based features
    logger.info("\n🔧 Generating tier-based features...")
    df = dataset.to_tabular(mode='tier', tiers=['A', 'B', 'C', 'D'])

    logger.info(f"\n✓ Generated features for {len(df):,} data points")
    logger.info(f"  Total columns: {len(df.columns)}")

    # Show tier breakdown
    logger.info("\n📈 Feature Breakdown by Tier:")

    # Tier A
    tier_a_cols = ['Z', 'A', 'N', 'Energy', 'out_n', 'out_p', 'out_a',
                   'out_g', 'out_f', 'out_t', 'out_h', 'out_d', 'is_met']
    tier_a_present = [col for col in tier_a_cols if col in df.columns]
    logger.info(f"  Tier A (Core): {len(tier_a_present)} features")
    logger.info(f"    {tier_a_present}")

    # Tier B
    tier_b_cols = ['R_fm', 'kR']
    tier_b_present = [col for col in tier_b_cols if col in df.columns]
    logger.info(f"  Tier B (Geometric): {len(tier_b_present)} features")
    logger.info(f"    {tier_b_present}")

    # Tier C
    tier_c_cols = ['Mass_Excess_MeV', 'Binding_Energy_MeV', 'Binding_Per_Nucleon_MeV',
                   'S_1n_MeV', 'S_2n_MeV', 'S_1p_MeV', 'S_2p_MeV']
    tier_c_present = [col for col in tier_c_cols if col in df.columns]
    logger.info(f"  Tier C (Energetics): {len(tier_c_present)} features")
    logger.info(f"    {tier_c_present}")

    # Tier D (including new Promiscuity Factor)
    tier_d_cols = ['Spin', 'Parity', 'Valence_N', 'Valence_P', 'P_Factor',
                   'Shell_Closure_N', 'Shell_Closure_P']
    tier_d_present = [col for col in tier_d_cols if col in df.columns]
    logger.info(f"  Tier D (Topological): {len(tier_d_present)} features")
    logger.info(f"    {tier_d_present}")

    # Demonstrate Promiscuity Factor
    if 'P_Factor' in df.columns:
        logger.info("\n🎯 Promiscuity Factor (P = N_p * N_n / (N_p + N_n)):")
        logger.info(f"  Formula: Measures coupling between proton and neutron valence shells")
        logger.info(f"  Range: [{df['P_Factor'].min():.2f}, {df['P_Factor'].max():.2f}]")
        logger.info(f"  Mean: {df['P_Factor'].mean():.2f}")
        logger.info(f"  Std:  {df['P_Factor'].std():.2f}")

        # Show examples for different nuclei
        sample_nuclei = df[['Z', 'A', 'Valence_N', 'Valence_P', 'P_Factor']].drop_duplicates()
        logger.info("\n  Example nuclei:")
        for _, row in sample_nuclei.head(5).iterrows():
            logger.info(
                f"    Z={int(row['Z']):3d}, A={int(row['A']):3d}: "
                f"Valence_N={int(row['Valence_N']):2d}, "
                f"Valence_P={int(row['Valence_P']):2d}, "
                f"P_Factor={row['P_Factor']:.2f}"
            )

    return df


def demonstrate_transformation_pipeline(df):
    """Demonstrate log-scaling and standardization transformations."""

    logger.info("\n" + "="*80)
    logger.info("TRANSFORMATION PIPELINE DEMONSTRATION")
    logger.info("="*80)

    if df is None:
        logger.error("❌ No data available for transformation demo")
        return

    # Separate features and target
    feature_cols = [col for col in df.columns
                   if col not in ['CrossSection', 'Uncertainty', 'Entry', 'MT']]
    X = df[feature_cols].copy()
    y = df['CrossSection'].copy()
    energy = df['Energy'].copy()

    logger.info(f"\n📊 Dataset: {len(X)} samples, {len(feature_cols)} features")

    # Split into train/test (proper ML workflow)
    logger.info("\n✂️  Splitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    energy_train = X_train['Energy']
    energy_test = X_test['Energy']

    logger.info(f"  Train: {len(X_train):,} samples")
    logger.info(f"  Test:  {len(X_test):,} samples")

    # Create and fit transformation pipeline
    logger.info("\n🔧 Creating transformation pipeline...")
    pipeline = TransformationPipeline()

    # Determine features to standardize (exclude Energy, particle vector indicators)
    standardize_cols = [
        col for col in feature_cols
        if col not in ['Energy', 'out_n', 'out_p', 'out_a', 'out_g', 'out_f',
                      'out_t', 'out_h', 'out_d', 'is_met', 'Entry', 'MT']
    ]

    logger.info(f"  Features to standardize: {len(standardize_cols)}")
    logger.info(f"    {standardize_cols[:5]}...")

    # Fit on training data only (critical for preventing data leakage!)
    logger.info("\n📈 Fitting pipeline on TRAINING data only...")
    X_train_transformed, y_train_transformed = pipeline.fit_transform(
        X_train, y_train, energy_train, feature_columns=standardize_cols
    )

    logger.info(f"  ✓ Fitted pipeline")
    logger.info(f"    Feature mean: {pipeline.feature_mean_[:3]}...")
    logger.info(f"    Feature std:  {pipeline.feature_std_[:3]}...")

    # Transform test data with fitted pipeline (NO refitting!)
    logger.info("\n🧪 Transforming TEST data with fitted pipeline...")
    X_test_transformed = pipeline.transform(X_test, energy_test)
    y_test_transformed = pipeline.transform_target(y_test)

    # Verify transformations
    logger.info("\n✅ Transformation Verification:")

    # 1. Log-scaling
    logger.info("\n  1. Log-Scaling:")
    logger.info(f"     Cross-section (original): [{y_train.min():.2e}, {y_train.max():.2e}] barns")
    logger.info(f"     Cross-section (log):      [{y_train_transformed.min():.2f}, {y_train_transformed.max():.2f}]")
    logger.info(f"     Energy (original):        [{energy_train.min():.2e}, {energy_train.max():.2e}] eV")
    logger.info(f"     Energy (log):             [{X_train_transformed['Energy'].min():.2f}, {X_train_transformed['Energy'].max():.2f}]")

    # 2. Standardization
    logger.info("\n  2. Standardization (Z-score):")
    for col in standardize_cols[:3]:
        if col in X_train.columns:
            orig_mean = X_train[col].mean()
            orig_std = X_train[col].std()
            trans_mean = X_train_transformed[col].mean()
            trans_std = X_train_transformed[col].std()
            logger.info(f"     {col:20s}: μ={orig_mean:8.2f} → {trans_mean:6.3f}, σ={orig_std:8.2f} → {trans_std:6.3f}")

    # 3. Inverse transformations
    logger.info("\n  3. Inverse Transformations:")
    y_train_inverse = pipeline.inverse_transform_target(y_train_transformed)
    X_train_inverse = pipeline.inverse_transform(X_train_transformed, energy_train)

    # Check reconstruction accuracy
    y_reconstruction_error = np.mean(np.abs(y_train - y_train_inverse))
    logger.info(f"     Cross-section reconstruction error: {y_reconstruction_error:.2e} barns (should be ≈0)")

    for col in standardize_cols[:3]:
        if col in X_train.columns:
            col_error = np.mean(np.abs(X_train[col] - X_train_inverse[col]))
            logger.info(f"     {col:20s} reconstruction error: {col_error:.2e}")

    # Save pipeline for deployment
    pipeline_path = Path('models/transformation_pipeline.pkl')
    pipeline_path.parent.mkdir(exist_ok=True)
    pipeline.save(str(pipeline_path))
    logger.info(f"\n💾 Saved pipeline to {pipeline_path}")

    # Demonstrate loading
    logger.info("\n📥 Loading pipeline from disk...")
    loaded_pipeline = TransformationPipeline.load(str(pipeline_path))
    logger.info(f"  ✓ Loaded pipeline: {loaded_pipeline}")

    # Verify loaded pipeline works
    X_test_reloaded = loaded_pipeline.transform(X_test, energy_test)
    transform_match = np.allclose(X_test_transformed[standardize_cols], X_test_reloaded[standardize_cols])
    logger.info(f"  ✓ Transform consistency check: {transform_match}")

    return pipeline


def demonstrate_isomer_fallback():
    """Demonstrate isomer fallback functionality."""

    logger.info("\n" + "="*80)
    logger.info("ISOMER FALLBACK DEMONSTRATION")
    logger.info("="*80)

    # Check if AME data exists
    ame_path = Path('data/mass_1.mas20.txt')
    if not ame_path.exists():
        logger.warning("⚠️  AME data not found - skipping isomer fallback demo")
        logger.info("   Download from: https://www-nds.iaea.org/amdc/ame2020/")
        return

    from nucml_next.data.enrichment import AME2020DataEnricher

    logger.info("\n📊 Loading AME2020 data with isomer support...")
    enricher = AME2020DataEnricher(data_dir='data/')
    enrichment_table = enricher.load_all()

    # Check for isomeric states
    if 'Isomer_Level' in enrichment_table.columns:
        ground_states = enrichment_table[enrichment_table['Isomer_Level'] == 0]
        isomers = enrichment_table[enrichment_table['Isomer_Level'] > 0]

        logger.info(f"\n  ✓ Loaded {len(ground_states):,} ground states")
        logger.info(f"  ✓ Loaded {len(isomers):,} isomeric states")

        if len(isomers) > 0:
            logger.info("\n🎯 Isomer Fallback Examples:")
            logger.info("   (Isomers inherit AME properties from ground state)")

            # Show example isomers with inherited properties
            for _, isomer in isomers.head(5).iterrows():
                z, a, level = int(isomer['Z']), int(isomer['A']), int(isomer['Isomer_Level'])
                ground = enrichment_table[
                    (enrichment_table['Z'] == z) &
                    (enrichment_table['A'] == a) &
                    (enrichment_table['Isomer_Level'] == 0)
                ]

                if len(ground) > 0:
                    logger.info(f"\n   Isotope: Z={z}, A={a}")
                    logger.info(f"     Ground state:  Spin={ground.iloc[0]['Spin']:.1f}, Mass_Excess={ground.iloc[0]['Mass_Excess_keV']:.1f} keV")
                    logger.info(f"     Isomer (m{level}):   Spin={isomer['Spin']:.1f}, Mass_Excess={isomer['Mass_Excess_keV']:.1f} keV (inherited)")
    else:
        logger.info("  ℹ️  Isomer_Level column not found - isomer support may not be enabled")


def main():
    """Run all demonstrations."""

    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "  NUCML-Next: Tier Features & Transformation Pipeline Demo".center(78) + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("╚" + "="*78 + "╝")

    # 1. Demonstrate tier-based features
    df = demonstrate_tier_features()

    # 2. Demonstrate transformation pipeline
    if df is not None:
        pipeline = demonstrate_transformation_pipeline(df)

    # 3. Demonstrate isomer fallback
    demonstrate_isomer_fallback()

    logger.info("\n" + "="*80)
    logger.info("✨ ALL DEMONSTRATIONS COMPLETE")
    logger.info("="*80)
    logger.info("\nKey Takeaways:")
    logger.info("  1. Tier D now includes Promiscuity Factor: P = N_p * N_n / (N_p + N_n)")
    logger.info("  2. Isomeric states inherit AME properties from ground states")
    logger.info("  3. TransformationPipeline handles log-scaling and standardization")
    logger.info("  4. All transformations are reversible (fit/transform/inverse_transform)")
    logger.info("  5. Pipeline can be saved/loaded for deployment\n")


if __name__ == '__main__':
    main()
