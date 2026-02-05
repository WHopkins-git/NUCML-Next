"""
SVGP Outlier Detection for EXFOR Cross-Section Data
====================================================

Stochastic Variational Gaussian Process (SVGP) outlier detection that fits a
GP per (Z, A, MT) group in log-log space and computes per-point z-scores.

Architecture:
    - Input: log10(Energy) -> Target: log10(CrossSection)
    - RBF kernel with automatic lengthscale
    - ~50 inducing points per group (variational approximation)
    - Z-score: |log_sigma - gp_mean| / gp_std

Edge Cases:
    - >= min_group_size_svgp points: Full SVGP fit
    - 2 to min_group_size_svgp - 1 points: MAD (Median Absolute Deviation) fallback
    - 1 point: z_score = 0 (cannot assess outlier status)

Usage:
    >>> from nucml_next.data.outlier_detection import SVGPOutlierDetector, SVGPConfig
    >>> config = SVGPConfig(device='cuda', checkpoint_dir='data/checkpoints/')
    >>> detector = SVGPOutlierDetector(config)
    >>> df_scored = detector.score_dataframe(df)
    >>> # df_scored now has columns: log_E, log_sigma, gp_mean, gp_std, z_score
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SVGPConfig:
    """Configuration for SVGP outlier detection.

    Attributes:
        n_inducing: Number of inducing points for variational approximation.
            More points = better fit but slower. 50 is good for most groups.
        max_epochs: Maximum training epochs per group.
        lr: Learning rate for Adam optimizer.
        convergence_tol: Stop early if loss change < this value.
        patience: Number of epochs with no improvement before early stopping.
        min_group_size_svgp: Minimum points to use SVGP (below this, use MAD).
        device: PyTorch device ('cpu' or 'cuda').
        checkpoint_dir: Directory for saving checkpoints (None = no checkpointing).
        checkpoint_interval: Save checkpoint every N groups processed.
        likelihood: Likelihood type:
            - 'student_t' (default): Robust to outliers, learns degrees of freedom.
              Recommended for nuclear data with heavy-tailed measurement errors.
            - 'heteroscedastic': Uses measurement uncertainties from 'Uncertainty'
              column for per-point noise variance. Best for calibrated uncertainty.
            - 'gaussian': Standard homoscedastic Gaussian noise (legacy).
        learn_additional_noise: For heteroscedastic likelihood only. If True,
            learn residual noise on top of measurement uncertainties. If False,
            use measurement uncertainties strictly. Default False for maximum
            uncertainty variation.
    """
    n_inducing: int = 50
    max_epochs: int = 300
    lr: float = 0.05
    convergence_tol: float = 1e-3
    patience: int = 10
    min_group_size_svgp: int = 10
    device: str = 'cpu'
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 1000
    likelihood: str = 'student_t'  # 'student_t', 'heteroscedastic', or 'gaussian'
    learn_additional_noise: bool = False  # For heteroscedastic only


class SVGPOutlierDetector:
    """SVGP-based outlier detector for nuclear cross-section data.

    Fits a Gaussian Process per (Z, A, MT) group in log-log space and
    computes z-scores measuring how far each point deviates from the
    smooth GP trend.

    Args:
        config: SVGPConfig with hyperparameters and device settings.

    Example:
        >>> detector = SVGPOutlierDetector(SVGPConfig(max_epochs=100))
        >>> df_scored = detector.score_dataframe(df)
        >>> outliers = df_scored[df_scored['z_score'] > 3.0]
    """

    def __init__(self, config: SVGPConfig = None):
        if config is None:
            config = SVGPConfig()
        self.config = config

        # Validate that gpytorch is available (fail fast, not silently)
        try:
            import gpytorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"SVGP outlier detection requires gpytorch and torch. "
                f"Install with: pip install gpytorch\n"
                f"Original error: {e}"
            ) from e

        # Statistics tracking
        self._stats = {
            'svgp_groups': 0,
            'mad_groups': 0,
            'single_groups': 0,
            'svgp_fallback_to_mad': 0,
            'total_points': 0,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all data points with SVGP z-scores.

        Groups data by (Z, A, MT) and fits a GP per group. Adds columns:
        - log_E: log10(Energy)
        - log_sigma: log10(CrossSection)
        - gp_mean: GP predicted mean in log10 space
        - gp_std: GP predicted std in log10 space
        - z_score: |log_sigma - gp_mean| / gp_std

        Args:
            df: DataFrame with columns: Z, A, MT, Energy, CrossSection

        Returns:
            DataFrame with additional columns: log_E, log_sigma, gp_mean, gp_std, z_score
        """
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # Validate required columns
        required = ['Z', 'A', 'MT', 'Energy', 'CrossSection']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Pre-compute log values
        result = df.copy()
        result['log_E'] = np.log10(result['Energy'].clip(lower=1e-30))
        result['log_sigma'] = np.log10(result['CrossSection'].clip(lower=1e-30))

        # Initialize output columns
        result['gp_mean'] = np.nan
        result['gp_std'] = np.nan
        result['z_score'] = np.nan

        # Group by (Z, A, MT)
        groups = list(result.groupby(['Z', 'A', 'MT']))
        n_groups = len(groups)
        logger.info(f"SVGP outlier detection: {n_groups:,} groups, {len(df):,} points")

        # Check for checkpoint to resume from
        start_idx = 0
        partial_results: Dict[Tuple, pd.DataFrame] = {}
        if self.config.checkpoint_dir:
            start_idx, partial_results = self._load_checkpoint()
            if start_idx > 0:
                logger.info(f"Resuming from checkpoint: group {start_idx}/{n_groups}")

        # Process groups
        iterator = enumerate(groups)
        if has_tqdm:
            iterator = tqdm(iterator, total=n_groups, desc="SVGP scoring",
                           initial=start_idx)

        for i, ((z, a, mt), group_df) in iterator:
            if i < start_idx:
                continue

            group_key = (z, a, mt)

            # Check if already processed (from checkpoint)
            if group_key in partial_results:
                scored = partial_results[group_key]
            else:
                scored = self._score_group(group_df)
                partial_results[group_key] = scored

            # Update result DataFrame
            result.loc[scored.index, 'gp_mean'] = scored['gp_mean'].values
            result.loc[scored.index, 'gp_std'] = scored['gp_std'].values
            result.loc[scored.index, 'z_score'] = scored['z_score'].values

            # Checkpoint
            if (self.config.checkpoint_dir and
                    (i + 1) % self.config.checkpoint_interval == 0):
                self._save_checkpoint(i + 1, partial_results)

            # Progress logging (every 10%)
            if not has_tqdm and n_groups >= 10 and (i + 1) % max(1, n_groups // 10) == 0:
                pct = 100 * (i + 1) / n_groups
                logger.info(
                    f"  Progress: {pct:.0f}% ({i+1}/{n_groups} groups) | "
                    f"SVGP: {self._stats['svgp_groups']}, "
                    f"MAD: {self._stats['mad_groups']}, "
                    f"Single: {self._stats['single_groups']}"
                )

        # Final checkpoint
        if self.config.checkpoint_dir:
            self._save_checkpoint(n_groups, partial_results)

        # Log summary
        logger.info(
            f"SVGP scoring complete: "
            f"{self._stats['svgp_groups']} SVGP, "
            f"{self._stats['mad_groups']} MAD fallback, "
            f"{self._stats['single_groups']} single-point, "
            f"{self._stats['svgp_fallback_to_mad']} SVGP failures"
        )

        # Verify no NaN z_scores remain
        nan_count = result['z_score'].isna().sum()
        if nan_count > 0:
            warnings.warn(f"{nan_count} points have NaN z_scores (unexpected)")

        return result

    def _score_group(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """Score a single (Z, A, MT) group.

        Routes to SVGP, MAD fallback, or single-point handler based on size.

        Args:
            df_group: DataFrame for one (Z, A, MT) group

        Returns:
            DataFrame with gp_mean, gp_std, z_score columns filled
        """
        n = len(df_group)
        result = df_group.copy()

        if n == 1:
            # Single point: cannot assess outlier status
            self._stats['single_groups'] += 1
            result['gp_mean'] = result['log_sigma'].values[0]
            result['gp_std'] = 1.0
            result['z_score'] = 0.0
            return result

        log_E = df_group['log_E'].values
        log_sigma = df_group['log_sigma'].values

        # Extract uncertainties for heteroscedastic likelihood
        log_uncertainties = None
        if self.config.likelihood == 'heteroscedastic':
            log_uncertainties = self._extract_log_uncertainties(df_group)

        if n < self.config.min_group_size_svgp:
            # Small group: MAD fallback
            self._stats['mad_groups'] += 1
            gp_mean, gp_std = self._mad_fallback(log_sigma)
        else:
            # Full SVGP fit
            try:
                gp_mean, gp_std = self._fit_svgp(log_E, log_sigma, log_uncertainties)
                self._stats['svgp_groups'] += 1
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                # Numerical/convergence failure, fall back to MAD
                logger.warning(
                    f"SVGP fit failed for group "
                    f"(Z={df_group['Z'].iloc[0]}, A={df_group['A'].iloc[0]}, "
                    f"MT={df_group['MT'].iloc[0]}, n={n}): "
                    f"{type(e).__name__}: {e}. Using MAD fallback."
                )
                self._stats['svgp_fallback_to_mad'] += 1
                gp_mean, gp_std = self._mad_fallback(log_sigma)

        # Compute z-scores
        z_scores = np.abs(log_sigma - gp_mean) / np.clip(gp_std, 1e-10, None)

        result['gp_mean'] = gp_mean
        result['gp_std'] = gp_std
        result['z_score'] = z_scores

        return result

    def _extract_log_uncertainties(
        self, df_group: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Extract and transform measurement uncertainties to log-space.

        For heteroscedastic likelihood, converts relative uncertainties to
        log-space standard deviations. Missing/invalid uncertainties are
        filled with the maximum valid uncertainty in the group (conservative).

        Args:
            df_group: DataFrame with 'Uncertainty' and 'CrossSection' columns

        Returns:
            Array of log-space uncertainties, or None if no valid uncertainties
        """
        if 'Uncertainty' not in df_group.columns or 'CrossSection' not in df_group.columns:
            return None

        unc = df_group['Uncertainty'].values
        xs = df_group['CrossSection'].values

        # Identify valid uncertainties
        valid = (unc > 0) & (xs > 0) & np.isfinite(unc) & np.isfinite(xs)

        if valid.sum() == 0:
            # No valid uncertainties - cannot use heteroscedastic
            return None

        # Compute relative uncertainty
        rel_unc = np.where(valid, unc / xs, np.nan)

        # Fill missing with max uncertainty in group (conservative approach)
        max_rel_unc = np.nanmax(rel_unc)
        rel_unc = np.where(np.isnan(rel_unc), max_rel_unc, rel_unc)

        # Clamp to reasonable range [1%, 100%]
        rel_unc = np.clip(rel_unc, 0.01, 1.0)

        # Convert to log10-space uncertainty
        # For y = log10(x), dy = dx / (x * ln(10)) = (dx/x) / ln(10) = rel_unc / 2.303
        # Simplified: sigma_log10 ~ 0.434 * sigma_rel
        log_uncertainties = 0.434 * rel_unc

        return log_uncertainties

    def _fit_svgp(
        self, log_E: np.ndarray, log_sigma: np.ndarray,
        log_uncertainties: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit SVGP model and return predictions.

        Uses GPyTorch ApproximateGP with variational inference.
        Input: log10(Energy), Target: log10(CrossSection).

        The likelihood type is controlled by config.likelihood:
        - 'student_t': Robust to outliers, learns degrees of freedom (default)
        - 'heteroscedastic': Uses per-point noise from log_uncertainties
        - 'gaussian': Standard homoscedastic Gaussian noise (legacy)

        Args:
            log_E: log10(Energy) values, shape (N,)
            log_sigma: log10(CrossSection) values, shape (N,)
            log_uncertainties: Per-point uncertainties in log-space for
                heteroscedastic likelihood. Shape (N,). Ignored for other likelihoods.

        Returns:
            Tuple of (gp_mean, gp_std) arrays, each shape (N,)
        """
        import torch
        import gpytorch

        device = torch.device(self.config.device)

        # Convert to tensors
        train_x = torch.tensor(log_E, dtype=torch.float32, device=device)
        train_y = torch.tensor(log_sigma, dtype=torch.float32, device=device)

        # Inducing points: evenly spaced in log-E range
        n_inducing = min(self.config.n_inducing, len(log_E))
        inducing_x = torch.linspace(
            train_x.min().item(), train_x.max().item(), n_inducing,
            device=device
        ).unsqueeze(-1)

        # Define SVGP model
        class _SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
                )
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Create model and likelihood
        model = _SVGPModel(inducing_x).to(device)

        # Choose likelihood based on config
        use_student_t = self.config.likelihood == 'student_t'
        use_heteroscedastic = (
            self.config.likelihood == 'heteroscedastic'
            and log_uncertainties is not None
        )

        if use_heteroscedastic:
            # Per-point noise variance from measurement uncertainties
            noise_var = torch.tensor(
                log_uncertainties**2, dtype=torch.float32, device=device
            )
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noise_var,
                learn_additional_noise=self.config.learn_additional_noise
            ).to(device)
        elif use_student_t:
            likelihood = gpytorch.likelihoods.StudentTLikelihood().to(device)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Training
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=self.config.lr)

        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=train_y.size(0)
        )

        train_x_2d = train_x.unsqueeze(-1)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            optimizer.zero_grad()
            output = model(train_x_2d)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            # Early stopping
            if best_loss - loss_val > self.config.convergence_tol:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        # Prediction
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if use_heteroscedastic:
                # For heteroscedastic, combine GP variance with per-point noise
                f_dist = model(train_x_2d)
                gp_mean = f_dist.mean.cpu().numpy()
                gp_var = f_dist.variance.cpu().numpy()

                # Per-point noise variance from input + learned additional noise (if enabled)
                noise_var = log_uncertainties**2
                if self.config.learn_additional_noise and hasattr(likelihood, 'second_noise'):
                    sn = likelihood.second_noise
                    add_noise = sn.item() if hasattr(sn, 'item') else float(sn)
                    noise_var = noise_var + add_noise

                gp_std = np.sqrt(gp_var + noise_var)
            elif use_student_t:
                # For Student-t, compute predictive uncertainty from GP + noise
                f_dist = model(train_x_2d)
                gp_mean = f_dist.mean.cpu().numpy()
                gp_var = f_dist.variance.cpu().numpy()

                # Get noise scale and degrees of freedom
                noise_scale = likelihood.noise.item() if hasattr(likelihood, 'noise') else 0.1
                df = likelihood.deg_free.item() if hasattr(likelihood, 'deg_free') else 4.0

                # Student-t variance: scale^2 * df/(df-2) for df > 2
                if df > 2:
                    noise_var = noise_scale**2 * df / (df - 2)
                else:
                    # For df <= 2, variance is infinite; use a large finite approx
                    noise_var = noise_scale**2 * 10

                gp_std = np.sqrt(gp_var + noise_var)
            else:
                # Standard Gaussian: use likelihood's predictive distribution
                pred = likelihood(model(train_x_2d))
                gp_mean = pred.mean.cpu().numpy()
                gp_std = pred.stddev.cpu().numpy()

        # Ensure std is positive
        gp_std = np.clip(gp_std, 1e-10, None)

        return gp_mean, gp_std

    def _mad_fallback(
        self, log_sigma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute outlier scores using Median Absolute Deviation.

        For small groups where SVGP is not reliable. Uses the MAD as a
        robust estimator of scale:
            scale = MAD * 1.4826 (consistent estimator for normal distribution)

        Args:
            log_sigma: log10(CrossSection) values, shape (N,)

        Returns:
            Tuple of (center, scale) arrays, each shape (N,) broadcast
        """
        center = np.median(log_sigma)
        mad = np.median(np.abs(log_sigma - center))

        # Convert MAD to standard deviation estimate
        # 1.4826 is the consistency constant for normal distribution
        scale = mad * 1.4826

        # Handle case where all values are identical (MAD = 0)
        if scale < 1e-10:
            scale = 1e-6  # Tiny scale → all z_scores ~ 0

        gp_mean = np.full_like(log_sigma, center)
        gp_std = np.full_like(log_sigma, scale)

        return gp_mean, gp_std

    def _save_checkpoint(
        self, group_idx: int, results: Dict[Tuple, pd.DataFrame]
    ) -> None:
        """Save processing checkpoint for resume capability.

        Args:
            group_idx: Index of last completed group
            results: Dictionary mapping (Z, A, MT) -> scored DataFrame
        """
        if not self.config.checkpoint_dir:
            return

        import torch

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'svgp_checkpoint.pt'

        # Serialize results as dict of {(Z,A,MT): {col: values}}
        serializable_results = {}
        for key, df in results.items():
            serializable_results[key] = {
                'index': df.index.tolist(),
                'gp_mean': df['gp_mean'].values.tolist(),
                'gp_std': df['gp_std'].values.tolist(),
                'z_score': df['z_score'].values.tolist(),
            }

        torch.save({
            'group_idx': group_idx,
            'results': serializable_results,
            'stats': self._stats.copy(),
        }, str(checkpoint_path))

        logger.info(f"Checkpoint saved: group {group_idx} -> {checkpoint_path}")

    def _load_checkpoint(self) -> Tuple[int, Dict[Tuple, pd.DataFrame]]:
        """Load checkpoint if available.

        Returns:
            Tuple of (start_group_idx, partial_results_dict)
        """
        if not self.config.checkpoint_dir:
            return 0, {}

        checkpoint_path = Path(self.config.checkpoint_dir) / 'svgp_checkpoint.pt'
        if not checkpoint_path.exists():
            return 0, {}

        try:
            import torch
            checkpoint = torch.load(str(checkpoint_path), weights_only=False)

            group_idx = checkpoint['group_idx']
            self._stats = checkpoint.get('stats', self._stats)

            # Deserialize results
            partial_results = {}
            for key, data in checkpoint['results'].items():
                # Reconstruct a minimal DataFrame with just the scored columns
                scored_df = pd.DataFrame({
                    'gp_mean': data['gp_mean'],
                    'gp_std': data['gp_std'],
                    'z_score': data['z_score'],
                }, index=data['index'])
                partial_results[key] = scored_df

            logger.info(f"Loaded checkpoint: group {group_idx}")
            return group_idx, partial_results

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
            return 0, {}


def extract_svgp_hyperparameters(
    log_E: np.ndarray,
    log_sigma: np.ndarray,
    config: SVGPConfig = None,
) -> Dict[str, Any]:
    """Fit SVGP and extract learned hyperparameters for diagnostics.

    This function is useful for understanding why uncertainty may appear
    constant-width. Key diagnostics:
      - Large lengthscale -> over-smoothing, constant uncertainty
      - High noise/signal ratio -> predictive variance dominated by noise
      - Inducing point placement -> whether they capture data distribution

    Args:
        log_E: log10(Energy) values, shape (N,)
        log_sigma: log10(CrossSection) values, shape (N,)
        config: SVGPConfig with hyperparameters. Uses defaults if None.

    Returns:
        Dictionary with:
          - lengthscale: RBF kernel lengthscale (larger = smoother)
          - outputscale: Signal variance (kernel amplitude)
          - noise: Likelihood noise variance
          - noise_signal_ratio: noise / outputscale (if > 1, noise dominates)
          - inducing_points: Final inducing point locations
          - n_epochs: Number of training epochs
          - final_loss: Final ELBO loss value

    Example:
        >>> params = extract_svgp_hyperparameters(log_E, log_sigma)
        >>> print(f"Lengthscale: {params['lengthscale']:.3f}")
        >>> print(f"Noise/Signal: {params['noise_signal_ratio']:.3f}")
    """
    import torch
    import gpytorch

    if config is None:
        config = SVGPConfig()

    device = torch.device(config.device)

    # Convert to tensors
    train_x = torch.tensor(log_E, dtype=torch.float32, device=device)
    train_y = torch.tensor(log_sigma, dtype=torch.float32, device=device)

    # Inducing points: evenly spaced in log-E range
    n_inducing = min(config.n_inducing, len(log_E))
    inducing_x = torch.linspace(
        train_x.min().item(), train_x.max().item(), n_inducing,
        device=device
    ).unsqueeze(-1)

    # Define SVGP model (same as in _fit_svgp)
    class _SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Create model and likelihood
    model = _SVGPModel(inducing_x).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # Training
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=config.lr)

    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=train_y.size(0)
    )

    train_x_2d = train_x.unsqueeze(-1)
    best_loss = float('inf')
    patience_counter = 0
    n_epochs = 0
    final_loss = float('inf')

    for epoch in range(config.max_epochs):
        optimizer.zero_grad()
        output = model(train_x_2d)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        n_epochs = epoch + 1
        final_loss = loss_val

        # Early stopping
        if best_loss - loss_val > config.convergence_tol:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # Extract hyperparameters
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        noise = likelihood.noise.item()
        inducing_pts = model.variational_strategy.inducing_points.cpu().numpy().flatten()

    return {
        'lengthscale': lengthscale,
        'outputscale': outputscale,
        'noise': noise,
        'noise_signal_ratio': noise / max(outputscale, 1e-10),
        'inducing_points': inducing_pts,
        'n_epochs': n_epochs,
        'final_loss': final_loss,
    }
