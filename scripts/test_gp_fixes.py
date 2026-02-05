#!/usr/bin/env python3
"""
Test GP calibration fixes on a single (Z, A, MT) group.

Compares four approaches:
1. Baseline: Current SVGP with homoscedastic GaussianLikelihood
2. Fix A: Heteroscedastic likelihood (FixedNoiseGaussianLikelihood)
3. Fix B: Lengthscale prior (GammaPrior)
4. Fix C: Student-t likelihood (robust to outliers, heavier tails)

Usage:
    python scripts/test_gp_fixes.py [--parquet PATH] [--z Z] [--a A] [--mt MT]
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy import stats


def load_group(parquet_path: str, z: int, a: int, mt: int) -> pd.DataFrame:
    """Load a single (Z, A, MT) group from parquet."""
    df = pd.read_parquet(parquet_path)
    mask = (df['Z'] == z) & (df['A'] == a) & (df['MT'] == mt)
    return df.loc[mask].copy()


def compute_diagnostics(
    log_E: np.ndarray,
    gp_std: np.ndarray,
    bin_width: float = 0.5,
) -> Dict[str, float]:
    """Compute density-uncertainty correlation and range metrics."""
    # Bin the data
    e_min, e_max = log_E.min(), log_E.max()
    bins = np.arange(e_min, e_max + bin_width, bin_width)

    # Density per bin
    counts, _ = np.histogram(log_E, bins=bins)
    density = counts / bin_width

    # Mean std per bin
    bin_indices = np.digitize(log_E, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    mean_stds = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_stds.append(gp_std[mask].mean())
        else:
            mean_stds.append(np.nan)

    mean_stds = np.array(mean_stds)
    valid = ~np.isnan(mean_stds) & (density > 0)

    if valid.sum() >= 3:
        corr, p_val = stats.spearmanr(density[valid], mean_stds[valid])
    else:
        corr, p_val = np.nan, np.nan

    return {
        'correlation': corr,
        'p_value': p_val,
        'min_std': gp_std.min(),
        'max_std': gp_std.max(),
        'max_min_ratio': gp_std.max() / gp_std.min() if gp_std.min() > 0 else np.inf,
        'mean_std': gp_std.mean(),
    }


class SVGPModel(gpytorch.models.ApproximateGP):
    """Standard SVGP model with RBF kernel."""

    def __init__(self, inducing_points, use_lengthscale_prior: bool = False):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        base_kernel = gpytorch.kernels.RBFKernel()
        if use_lengthscale_prior:
            # Gamma(2, 1) prior: mode at 1.0, discourages very large lengthscales
            base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(2.0, 1.0)

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_baseline(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    n_inducing: int = 50,
    max_epochs: int = 300,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Baseline: Homoscedastic GaussianLikelihood."""
    device = torch.device('cpu')
    train_x = torch.tensor(log_E, dtype=torch.float32, device=device).unsqueeze(-1)
    train_y = torch.tensor(log_sigma, dtype=torch.float32, device=device)

    n_ind = min(n_inducing, len(log_E))
    inducing_x = torch.linspace(
        train_x.min().item(), train_x.max().item(), n_ind, device=device
    ).unsqueeze(-1)

    model = SVGPModel(inducing_x, use_lengthscale_prior=False).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    for _ in range(max_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(train_x))
        gp_mean = pred.mean.cpu().numpy()
        gp_std = pred.stddev.cpu().numpy()

    hyperparams = {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'noise': likelihood.noise.item(),
    }

    return gp_mean, gp_std, hyperparams


def fit_heteroscedastic(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    uncertainties: np.ndarray,
    n_inducing: int = 50,
    max_epochs: int = 300,
    learn_additional_noise: bool = True,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fix A: Heteroscedastic FixedNoiseGaussianLikelihood.

    Args:
        learn_additional_noise: If True, learn residual noise on top of fixed noise.
        scale_factor: Multiply uncertainties by this factor (to test if they're underestimated).
    """
    device = torch.device('cpu')
    train_x = torch.tensor(log_E, dtype=torch.float32, device=device).unsqueeze(-1)
    train_y = torch.tensor(log_sigma, dtype=torch.float32, device=device)

    # Convert measurement uncertainties to variance in log-space
    # Apply scale factor to test if uncertainties are underestimated
    scaled_unc = uncertainties * scale_factor
    noise_var = torch.tensor(scaled_unc**2, dtype=torch.float32, device=device)

    n_ind = min(n_inducing, len(log_E))
    inducing_x = torch.linspace(
        train_x.min().item(), train_x.max().item(), n_ind, device=device
    ).unsqueeze(-1)

    model = SVGPModel(inducing_x, use_lengthscale_prior=False).to(device)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise_var, learn_additional_noise=learn_additional_noise
    ).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    for _ in range(max_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # For heteroscedastic, compute GP variance + per-point noise
        f_dist = model(train_x)
        gp_mean = f_dist.mean.cpu().numpy()
        gp_var = f_dist.variance.cpu().numpy()

        # Per-point noise + any additional learned noise
        total_noise_var = scaled_unc**2
        if learn_additional_noise and hasattr(likelihood, 'second_noise'):
            total_noise_var = total_noise_var + likelihood.second_noise.item()

        gp_std = np.sqrt(gp_var + total_noise_var)

    # Get additional noise if learned
    add_noise = 0.0
    if learn_additional_noise and hasattr(likelihood, 'second_noise'):
        sn = likelihood.second_noise
        add_noise = sn.item() if hasattr(sn, 'item') else float(sn)

    hyperparams = {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'additional_noise': add_noise,
        'scale_factor': scale_factor,
    }

    return gp_mean, gp_std, hyperparams


def fit_with_prior(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    n_inducing: int = 50,
    max_epochs: int = 300,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fix B: Lengthscale prior to prevent over-smoothing."""
    device = torch.device('cpu')
    train_x = torch.tensor(log_E, dtype=torch.float32, device=device).unsqueeze(-1)
    train_y = torch.tensor(log_sigma, dtype=torch.float32, device=device)

    n_ind = min(n_inducing, len(log_E))
    inducing_x = torch.linspace(
        train_x.min().item(), train_x.max().item(), n_ind, device=device
    ).unsqueeze(-1)

    model = SVGPModel(inducing_x, use_lengthscale_prior=True).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    for _ in range(max_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(train_x))
        gp_mean = pred.mean.cpu().numpy()
        gp_std = pred.stddev.cpu().numpy()

    hyperparams = {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'noise': likelihood.noise.item(),
    }

    return gp_mean, gp_std, hyperparams


def fit_student_t(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    n_inducing: int = 50,
    max_epochs: int = 300,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fix C: Student-t likelihood for robustness to outliers."""
    device = torch.device('cpu')
    train_x = torch.tensor(log_E, dtype=torch.float32, device=device).unsqueeze(-1)
    train_y = torch.tensor(log_sigma, dtype=torch.float32, device=device)

    n_ind = min(n_inducing, len(log_E))
    inducing_x = torch.linspace(
        train_x.min().item(), train_x.max().item(), n_ind, device=device
    ).unsqueeze(-1)

    model = SVGPModel(inducing_x, use_lengthscale_prior=False).to(device)
    # Student-t with learnable degrees of freedom
    likelihood = gpytorch.likelihoods.StudentTLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    for _ in range(max_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    # For Student-t, we need to compute mean and std differently
    # The likelihood's forward returns a StudentT distribution
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model(train_x)  # GP posterior (MultivariateNormal)
        gp_mean = f_dist.mean.cpu().numpy()
        # For predictive uncertainty, combine GP variance with likelihood variance
        # Student-t scale parameter relates to variance as var = scale^2 * df/(df-2) for df > 2
        gp_var = f_dist.variance.cpu().numpy()
        noise_scale = likelihood.noise.item() if hasattr(likelihood, 'noise') else 0.1
        df = likelihood.deg_free.item() if hasattr(likelihood, 'deg_free') else 4.0
        # Total variance = GP variance + noise variance
        # For Student-t: noise_var ~ noise_scale^2 * df/(df-2) if df > 2
        if df > 2:
            noise_var = noise_scale**2 * df / (df - 2)
        else:
            noise_var = noise_scale**2 * 10  # Approximate for small df
        gp_std = np.sqrt(gp_var + noise_var)

    hyperparams = {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'noise_scale': noise_scale,
        'deg_freedom': df,
    }

    return gp_mean, gp_std, hyperparams


def main():
    parser = argparse.ArgumentParser(description="Test GP calibration fixes")
    parser.add_argument("--parquet", type=str, default="data/exfor_processed.parquet")
    parser.add_argument("--z", type=int, default=79, help="Atomic number (default: Au=79)")
    parser.add_argument("--a", type=int, default=197, help="Mass number (default: 197)")
    parser.add_argument("--mt", type=int, default=102, help="Reaction type (default: n,g=102)")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    args = parser.parse_args()

    # Load data
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        parquet_path = Path(__file__).parent.parent / args.parquet

    if not parquet_path.exists():
        print(f"ERROR: Parquet not found: {args.parquet}")
        return 1

    print("=" * 80)
    print("GP CALIBRATION FIX COMPARISON")
    print("=" * 80)
    print(f"Target: Z={args.z}, A={args.a}, MT={args.mt}")
    print(f"Parquet: {parquet_path}")

    df_group = load_group(str(parquet_path), args.z, args.a, args.mt)
    if len(df_group) == 0:
        print(f"ERROR: No data found for Z={args.z}, A={args.a}, MT={args.mt}")
        return 1

    print(f"Data points: {len(df_group):,}")

    log_E = df_group['log_E'].values
    log_sigma = df_group['log_sigma'].values

    # Get measurement uncertainties if available
    log_uncertainties = None
    if 'Uncertainty' in df_group.columns and 'CrossSection' in df_group.columns:
        unc = df_group['Uncertainty'].values
        xs = df_group['CrossSection'].values
        # Check for valid uncertainties
        valid_unc = (unc > 0) & (xs > 0) & np.isfinite(unc) & np.isfinite(xs)
        if valid_unc.sum() > len(df_group) * 0.5:  # At least 50% valid
            rel_unc = np.where(valid_unc, unc / xs, 0.1)  # 10% fallback for invalid
            rel_unc = np.clip(rel_unc, 0.01, 1.0)
            # In log10 space: sigma_log10 ~ 0.434 * sigma_rel
            log_uncertainties = 0.434 * rel_unc
            print(f"\nMeasurement uncertainty range: [{log_uncertainties.min():.3f}, {log_uncertainties.max():.3f}] (log10)")
        else:
            print(f"\nMeasurement uncertainties: {valid_unc.sum()}/{len(df_group)} valid - using synthetic")

    if log_uncertainties is None:
        # Create synthetic heteroscedastic noise based on energy
        # Higher uncertainty at high energies (sparse data), lower at thermal
        log_E_norm = (log_E - log_E.min()) / (log_E.max() - log_E.min() + 1e-10)
        log_uncertainties = 0.05 + 0.15 * log_E_norm  # 5-20% depending on energy
        print(f"\nUsing synthetic uncertainties: [{log_uncertainties.min():.3f}, {log_uncertainties.max():.3f}] (log10)")

    # Run each approach
    results = {}

    print("\n" + "-" * 80)
    print("1. BASELINE (Homoscedastic GaussianLikelihood)")
    print("-" * 80)
    gp_mean, gp_std, hp = fit_baseline(log_E, log_sigma, max_epochs=args.epochs)
    diag = compute_diagnostics(log_E, gp_std)
    results['baseline'] = {'diagnostics': diag, 'hyperparams': hp}
    print(f"  Lengthscale:    {hp['lengthscale']:.4f}")
    print(f"  Outputscale:    {hp['outputscale']:.4f}")
    print(f"  Noise:          {hp['noise']:.4f}")
    print(f"  Corr(rho,std):  {diag['correlation']:+.3f}")
    print(f"  max/min ratio:  {diag['max_min_ratio']:.2f}")

    print("\n" + "-" * 80)
    print("2. FIX A: Heteroscedastic (with learned additional noise)")
    print("-" * 80)
    gp_mean, gp_std, hp = fit_heteroscedastic(log_E, log_sigma, log_uncertainties, max_epochs=args.epochs, learn_additional_noise=True)
    diag = compute_diagnostics(log_E, gp_std)
    results['heteroscedastic'] = {'diagnostics': diag, 'hyperparams': hp}
    print(f"  Lengthscale:    {hp['lengthscale']:.4f}")
    print(f"  Outputscale:    {hp['outputscale']:.4f}")
    print(f"  Add. noise:     {hp['additional_noise']:.4f}")
    print(f"  Corr(rho,std):  {diag['correlation']:+.3f}")
    print(f"  max/min ratio:  {diag['max_min_ratio']:.2f}")

    print("\n" + "-" * 80)
    print("2b. FIX A: Heteroscedastic (NO additional noise - strict)")
    print("-" * 80)
    gp_mean, gp_std, hp = fit_heteroscedastic(log_E, log_sigma, log_uncertainties, max_epochs=args.epochs, learn_additional_noise=False)
    diag = compute_diagnostics(log_E, gp_std)
    results['hetero_strict'] = {'diagnostics': diag, 'hyperparams': hp}
    print(f"  Lengthscale:    {hp['lengthscale']:.4f}")
    print(f"  Outputscale:    {hp['outputscale']:.4f}")
    print(f"  Corr(rho,std):  {diag['correlation']:+.3f}")
    print(f"  max/min ratio:  {diag['max_min_ratio']:.2f}")

    print("\n" + "-" * 80)
    print("3. FIX B: Lengthscale Prior (GammaPrior(2,1))")
    print("-" * 80)
    gp_mean, gp_std, hp = fit_with_prior(log_E, log_sigma, max_epochs=args.epochs)
    diag = compute_diagnostics(log_E, gp_std)
    results['lengthscale_prior'] = {'diagnostics': diag, 'hyperparams': hp}
    print(f"  Lengthscale:    {hp['lengthscale']:.4f}")
    print(f"  Outputscale:    {hp['outputscale']:.4f}")
    print(f"  Noise:          {hp['noise']:.4f}")
    print(f"  Corr(rho,std):  {diag['correlation']:+.3f}")
    print(f"  max/min ratio:  {diag['max_min_ratio']:.2f}")

    print("\n" + "-" * 80)
    print("4. FIX C: Student-t Likelihood (heavier tails)")
    print("-" * 80)
    try:
        gp_mean, gp_std, hp = fit_student_t(log_E, log_sigma, max_epochs=args.epochs)
        diag = compute_diagnostics(log_E, gp_std)
        results['student_t'] = {'diagnostics': diag, 'hyperparams': hp}
        print(f"  Lengthscale:    {hp['lengthscale']:.4f}")
        print(f"  Outputscale:    {hp['outputscale']:.4f}")
        print(f"  Deg. freedom:   {hp['deg_freedom']:.2f}")
        print(f"  Corr(rho,std):  {diag['correlation']:+.3f}")
        print(f"  max/min ratio:  {diag['max_min_ratio']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['student_t'] = None

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} | {'Corr(rho,std)':<14} | {'max/min':<10} | {'Lengthscale':<12}")
    print("-" * 80)

    for name, data in results.items():
        if data is None:
            continue
        d = data['diagnostics']
        h = data['hyperparams']
        corr_str = f"{d['correlation']:+.3f}" if not np.isnan(d['correlation']) else "N/A"
        ratio_str = f"{d['max_min_ratio']:.2f}"
        ls_str = f"{h['lengthscale']:.4f}"
        print(f"{name:<25} | {corr_str:<14} | {ratio_str:<10} | {ls_str:<12}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline_ratio = results['baseline']['diagnostics']['max_min_ratio']

    improvements = []
    for name, data in results.items():
        if data is None or name == 'baseline':
            continue
        ratio = data['diagnostics']['max_min_ratio']
        if ratio > baseline_ratio * 1.1:  # >10% improvement
            improvements.append((name, ratio / baseline_ratio))

    if improvements:
        print("\nImprovements over baseline (max/min ratio):")
        for name, factor in sorted(improvements, key=lambda x: -x[1]):
            print(f"  {name}: {factor:.2f}x better")
    else:
        print("\nNo significant improvements detected.")
        print("The constant-uncertainty issue may require a different approach:")
        print("  - Adaptive inducing point placement")
        print("  - Different kernel (e.g., Matern, periodic)")
        print("  - Input warping to handle non-stationarity")

    # Student-t analysis
    if results.get('student_t'):
        df_val = results['student_t']['hyperparams'].get('deg_freedom', np.nan)
        if df_val is not None and not np.isnan(df_val):
            print(f"\nStudent-t degrees of freedom: {df_val:.2f}")
            if df_val < 10:
                print("  -> Heavy tails detected (df < 10), suggesting outliers")
                print("  -> Student-t may be more appropriate than Gaussian")
            elif df_val > 30:
                print("  -> Approaching Gaussian (df > 30)")
                print("  -> Student-t offers no advantage over Gaussian")
            else:
                print("  -> Moderate tails (10 < df < 30)")

    return 0


if __name__ == "__main__":
    exit(main())
