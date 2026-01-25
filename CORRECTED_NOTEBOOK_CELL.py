# CORRECTED VERSION OF CELL-3 FOR 00_Baselines_and_Limitations.ipynb
# This fixes three critical issues:
# 1. Uses mode='tier' instead of mode='naive' to get particle vector features
# 2. Adds manual MT filter to exclude bookkeeping codes
# 3. Properly filters energy range

# ============================================================================
# DATA LOADING WITH MANUAL FILTERS (PyArrow bug workarounds)
# ============================================================================

from nucml_next.data import DataSelection

print("Creating physics-aware data selection...")
print("=" * 80)

# Training selection: Reactor physics, neutrons, all physical reactions
training_selection = DataSelection(
    projectile='neutron',
    energy_min=1e-5,               # Thermal neutrons
    energy_max=2e7,                # 20 MeV max (reactor physics)
    mt_mode='all_physical',        # All physical reactions
    exclude_bookkeeping=True,      # Exclude MT 0, 1, ≥9000
    drop_invalid=True,
    tiers=['A', 'C']               # Tier A: particle vector, Tier C: energetics
)

print("Training Selection:")
print(training_selection)
print()

# Load FULL dataset for training
print("=" * 80)
print("Loading training dataset...")
print("=" * 80)
dataset_full = NucmlDataset(
    data_path='../data/exfor_processed.parquet',
    mode='tabular',
    selection=training_selection
)

# ============================================================================
# WORKAROUND: PyArrow Filter Bugs (Apply Manual Filters)
# ============================================================================
# PyArrow predicate pushdown doesn't correctly filter energy and MT codes
# Apply manual filters to ensure correct data quality

print("\n⚠️  Applying manual filters (PyArrow bug workarounds)...")
original_size = len(dataset_full.df)

# Filter 1: Energy range (should be 1e-5 to 2e7, but PyArrow loads up to 1e9)
dataset_full.df = dataset_full.df[
    (dataset_full.df['Energy'] >= training_selection.energy_min) &
    (dataset_full.df['Energy'] <= training_selection.energy_max)
].copy()
after_energy = len(dataset_full.df)

# Filter 2: Exclude bookkeeping codes (MT 0, 1, ≥9000)
# Despite exclude_bookkeeping=True, PyArrow loads them anyway
dataset_full.df = dataset_full.df[
    (dataset_full.df['MT'] != 0) &      # Undefined reactions
    (dataset_full.df['MT'] != 1) &      # Total XS (sum of partials)
    (dataset_full.df['MT'] < 9000)      # Covariance data
].copy()
final_size = len(dataset_full.df)

print(f"  Energy filter:     {original_size:,} → {after_energy:,} ({after_energy/original_size*100:.1f}%)")
print(f"  MT filter:         {after_energy:,} → {final_size:,} ({final_size/after_energy*100:.1f}%)")
print(f"  Total filtered:    {original_size:,} → {final_size:,} ({final_size/original_size*100:.1f}%)")

# ============================================================================
# Project to tabular format with TIER-BASED features (particle vector)
# ============================================================================
print("\n" + "=" * 80)
print("Projecting to tabular format (tier mode with particle vector)...")
print("=" * 80)

# IMPORTANT: Use mode='tier' to get particle vector transformation
# mode='naive' uses MT one-hot encoding (intentionally bad for comparison)
# mode='tier' uses FeatureGenerator which transforms MT → particle features
df_tier = dataset_full.to_tabular(mode='tier', tiers=['A', 'C'])

print(f"\n✓ Training dataset: {df_tier.shape}")
print(f"  Energy range: {df_tier['Energy'].min():.2e} to {df_tier['Energy'].max():.2e} eV")
print(f"\n📋 Features Available:")
print(f"  Tier A features: Z, A, N, Energy + 9-feature particle vector")
print(f"    Particle vector: out_n, out_p, out_a, out_g, out_f, out_t, out_h, out_d, is_met")
print(f"  Tier C features: Mass excess, binding, separation energies")
print(f"  Total features: {len(df_tier.columns)}")
print(f"\nFeature names:")
print(df_tier.columns.tolist())

# Show isotope distribution
print("\n📊 Training Data Distribution (Top 10 Isotopes):")
isotope_counts = dataset_full.df.groupby(['Z', 'A']).size().sort_values(ascending=False).head(10)
for (z, a), count in isotope_counts.items():
    element_map = {92: 'U', 17: 'Cl', 94: 'Pu', 26: 'Fe', 8: 'O', 1: 'H',
                   82: 'Pb', 6: 'C', 13: 'Al', 7: 'N', 11: 'Na', 79: 'Au'}
    elem = element_map.get(z, f'Z{z}')
    print(f"  {elem}-{a:3d}: {count:>8,} measurements")

print(f"\n✓ Total isotopes: {dataset_full.df.groupby(['Z', 'A']).ngroups} unique Z/A combinations")
print(f"✓ Total reaction types: {dataset_full.df['MT'].nunique()} unique MT codes")
print(f"✓ Total measurements: {len(dataset_full.df):,}")

# Show MT distribution
print("\n📊 Top 10 Reaction Types (MT codes):")
mt_counts = dataset_full.df['MT'].value_counts().head(10)
mt_names = {18: 'Fission', 102: '(n,γ) Capture', 103: '(n,p)', 2: 'Elastic',
            16: '(n,2n)', 17: '(n,3n)', 4: 'Inelastic', 107: '(n,α)'}
for mt, count in mt_counts.items():
    name = mt_names.get(mt, f'MT-{mt}')
    print(f"  MT {mt:3d} {name:15s}: {count:>8,} measurements")

print(f"\n✓ Training on neutron-induced reactions (reactor energies 0.01 eV - 20 MeV)")
print(f"✓ Excluded bookkeeping codes (MT 0, 1, 9000+)")
print(f"✓ Using Tier A + C features with particle vector transformation")

# ============================================================================
# Load evaluation targets (U-235 and Cl-35)
# ============================================================================
print("\n" + "="*70)
print("Loading evaluation targets (U-235 and Cl-35)...")
print(f"NOTE: Using same energy range: {training_selection.energy_min:.2e} to {training_selection.energy_max:.2e} eV")
print("="*70)

eval_selection = DataSelection(
    projectile='neutron',
    energy_min=training_selection.energy_min,
    energy_max=training_selection.energy_max,
    mt_mode='custom',
    custom_mt_codes=[18, 102, 103],  # Fission, capture, (n,p)
    exclude_bookkeeping=True,
    drop_invalid=True,
    tiers=['A', 'C']
)

dataset_eval = NucmlDataset(
    data_path='../data/exfor_processed.parquet',
    mode='tabular',
    selection=eval_selection
)

# Apply same manual filters
dataset_eval.df = dataset_eval.df[
    (dataset_eval.df['Energy'] >= eval_selection.energy_min) &
    (dataset_eval.df['Energy'] <= eval_selection.energy_max) &
    (dataset_eval.df['MT'] != 0) &
    (dataset_eval.df['MT'] != 1) &
    (dataset_eval.df['MT'] < 9000)
].copy()

# Filter to U-235 and Cl-35 only
dataset_eval.df = dataset_eval.df[
    ((dataset_eval.df['Z'] == 92) & (dataset_eval.df['A'] == 235)) |
    ((dataset_eval.df['Z'] == 17) & (dataset_eval.df['A'] == 35))
].copy()

print(f"✓ Evaluation dataset: {len(dataset_eval.df)} measurements")
print(f"  Energy range: {dataset_eval.df['Energy'].min():.2e} to {dataset_eval.df['Energy'].max():.2e} eV")
print("\n📊 Evaluation Isotopes:")
for (z, a), group in dataset_eval.df.groupby(['Z', 'A']):
    isotope = f"{'U' if z==92 else 'Cl'}-{a}"
    e_min = group['Energy'].min()
    e_max = group['Energy'].max()
    print(f"  {isotope:8s}: {len(group):>6,} measurements (Energy: {e_min:.2e} to {e_max:.2e} eV)")
print("="*70)

# Display first few rows
df_tier.head()
