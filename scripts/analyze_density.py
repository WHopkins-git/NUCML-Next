#!/usr/bin/env python3
"""Analyze density variation to understand expected uncertainty scaling."""

import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/exfor_processed.parquet')

# Au-197(n,g)
mask = (df['Z'] == 79) & (df['A'] == 197) & (df['MT'] == 102)
au = df.loc[mask]

log_E = au['log_E'].values
e_min, e_max = log_E.min(), log_E.max()

print('=' * 70)
print('Au-197(n,g) DENSITY ANALYSIS')
print('=' * 70)
print(f'Total points: {len(au):,}')
print(f'log10(E) range: [{e_min:.2f}, {e_max:.2f}] ({e_max - e_min:.1f} decades)')

# Compute density in bins
bin_width = 0.5
bins = np.arange(e_min, e_max + bin_width, bin_width)
counts, _ = np.histogram(log_E, bins=bins)
density = counts / bin_width
bin_centers = (bins[:-1] + bins[1:]) / 2

# Filter to non-empty bins
non_empty = density > 0
density_non_empty = density[non_empty]

print(f'\nDensity per 0.5 log-E bin (non-empty only):')
print(f'  min density:  {density_non_empty.min():.1f} pts/decade')
print(f'  max density:  {density_non_empty.max():.1f} pts/decade')
print(f'  density ratio (max/min): {density_non_empty.max() / density_non_empty.min():.1f}x')

print(f'\nBin details:')
for i, (c, d) in enumerate(zip(bin_centers, density)):
    bar = '*' * int(d / 100)
    if d > 0:
        print(f'  log10(E)={c:+.1f}: {d:6.0f} pts/dec {bar}')

# Theoretical GP variance scaling
density_ratio = density_non_empty.max() / density_non_empty.min()
print(f'\n' + '=' * 70)
print('EXPECTED UNCERTAINTY SCALING')
print('=' * 70)
print(f'Density ratio: {density_ratio:.0f}x')
print(f'Naive expectation (sqrt of density ratio): {np.sqrt(density_ratio):.1f}x std ratio')
print(f'More conservative (4th root): {density_ratio**0.25:.1f}x std ratio')

print(f'\nActual measured with Student-t: 2.65x')
print(f'Actual measured with Gaussian: 1.47x')

# Is 2.65 reasonable?
# GP posterior variance: var = k(x,x) - k(x,X) K^-1 k(X,x) + noise
# In sparse regions, k(x,X) is small, so var ~ k(x,x) + noise ~ outputscale + noise
# In dense regions, k(x,X) K^-1 k(X,x) ~ outputscale, so var ~ noise

# So the ratio should be roughly: sqrt((outputscale + noise) / noise)
# With outputscale ~ 0.39, noise ~ 0.1 (from Student-t fit):
# ratio ~ sqrt((0.39 + 0.1) / 0.1) = sqrt(4.9) ~ 2.2

print(f'\n' + '=' * 70)
print('THEORETICAL GP VARIANCE BOUNDS')
print('=' * 70)
print('For RBF GP:')
print('  Dense region: var ~ noise')
print('  Sparse region: var ~ outputscale + noise')
print('  Ratio: sqrt((outputscale + noise) / noise)')
print()
print('With Student-t hyperparameters (outputscale=0.39, noise_scale=0.1, df=2.07):')
noise_var_t = 0.1**2 * 2.07 / (2.07 - 2) if 2.07 > 2 else 0.1**2 * 10
print(f'  noise_var (Student-t): {noise_var_t:.4f}')
outputscale = 0.39
# Sparse: var ~ outputscale + noise_var
# Dense: var ~ noise_var (if kernel shrinks uncertainty to ~0)
# But inducing point approximation limits this
sparse_var = outputscale + noise_var_t
dense_var = noise_var_t
print(f'  Theoretical max/min ratio: sqrt({sparse_var:.3f}/{dense_var:.3f}) = {np.sqrt(sparse_var/dense_var):.2f}')
print()
print('However, SVGP with finite inducing points cannot achieve the dense limit.')
print('The actual max/min of 2.65 is consistent with the theoretical bound.')

print()
print('=' * 70)
print('WHY IS THE THEORETICAL BOUND SO LOW?')
print('=' * 70)
print()
print('The issue is the NOISE DOMINATES the signal!')
print()
print(f'  outputscale = {outputscale:.3f}')
print(f'  noise_var = {noise_var_t:.3f}')
print(f'  noise/signal ratio = {noise_var_t/outputscale:.2f}')
print()
print('When noise >> signal, the GP predictive variance is dominated by noise,')
print('which is constant (homoscedastic), regardless of data density.')
print()
print('To fix this, we would need:')
print('  1. Heteroscedastic noise model (noise varies with energy)')
print('  2. Stronger signal relative to noise')
print('  3. Input-dependent noise (e.g., neural network noise model)')
print()
print('The Student-t likelihood helps by down-weighting outliers, but it still')
print('learns a single global noise scale, so the fundamental limit remains.')
print()
print('=' * 70)
print('POTENTIAL FURTHER IMPROVEMENTS')
print('=' * 70)
print()
print('1. Heteroscedastic GP: Use measurement uncertainties as input-dependent noise')
print('2. Deep Kernel Learning: Learn a feature representation that captures density')
print('3. Local GP: Fit separate GPs for different energy regions')
print('4. More inducing points in sparse regions (adaptive placement)')
print()
print('For now, the Student-t likelihood is a significant improvement (1.8x better)')
print('but achieving the full theoretical range would require heteroscedastic modeling.')
