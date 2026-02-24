"""
Contaminated Normal Likelihood for Robust GP Outlier Detection
==============================================================

Provides a contaminated normal mixture likelihood that identifies
outlier points via EM and downweights them, producing a continuous
``outlier_probability ∈ [0, 1]`` per point.

Model:
    p(yᵢ | fᵢ) = (1-ε)·N(yᵢ; fᵢ, σᵢ²) + ε·N(yᵢ; fᵢ, κ·σᵢ²)

where ε is the contamination fraction and κ is the noise inflation
factor for the outlier component.

Key Components:
    LikelihoodConfig: Configuration dataclass
    run_contaminated_em: NumPy EM algorithm
    run_contaminated_em_torch: PyTorch EM algorithm (GPU-accelerated)
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LikelihoodConfig:
    """Configuration for the GP likelihood model.

    Attributes:
        likelihood_type: ``'gaussian'`` (default, no EM) or ``'contaminated'``
            (run EM to identify outliers).
        contamination_fraction: ε — prior probability that any point is an
            outlier.  Fixed (not optimized).
        contamination_scale: κ — noise inflation factor for the outlier
            component.  An outlier's effective noise is κ·σᵢ².  Fixed.
        max_em_iterations: Maximum EM iterations before stopping.
        em_convergence_tol: Convergence tolerance on max change in outlier
            probabilities between iterations.
    """
    likelihood_type: str = 'gaussian'
    contamination_fraction: float = 0.05
    contamination_scale: float = 10.0
    max_em_iterations: int = 10
    em_convergence_tol: float = 1e-3


def run_contaminated_em(
    K_kernel: np.ndarray,
    y: np.ndarray,
    noise_variance: np.ndarray,
    mean_value: Union[float, np.ndarray],
    config: LikelihoodConfig,
    jitter: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run contaminated normal EM algorithm (NumPy).

    After convergence, the returned Cholesky factor ``L`` and effective
    noise ``sigma_eff_sq`` incorporate outlier downweighting.  The caller
    can recompute ``alpha = L⁻ᵀ L⁻¹ residuals`` to get a robust GP
    posterior.

    Args:
        K_kernel: (n, n) kernel matrix **without** noise (pure signal
            covariance).
        y: (n,) training targets (log_sigma).
        noise_variance: (n,) per-point measurement noise σᵢ².
        mean_value: Mean function values at training points (scalar or
            array of shape (n,)).
        config: Likelihood configuration.
        jitter: Diagonal jitter for numerical stability.

    Returns:
        (sigma_eff_sq, outlier_prob, L):
            - sigma_eff_sq: (n,) effective noise variance after EM.
            - outlier_prob: (n,) posterior outlier probability per point.
            - L: (n, n) Cholesky factor of (K_kernel + diag(sigma_eff_sq) + jitter·I).
    """
    n = len(y)
    eps = config.contamination_fraction
    kappa = config.contamination_scale
    residuals = y - np.asarray(mean_value)

    # Log prior ratio: log(ε / (1-ε))
    log_prior_ratio = np.log(eps) - np.log(1.0 - eps)

    # Initialize outlier weights at prior
    w = np.full(n, eps)

    for iteration in range(config.max_em_iterations):
        w_old = w.copy()

        # --- M-step: update effective noise from current weights ---
        sigma_eff_sq = noise_variance * (1.0 + w * (kappa - 1.0))

        # Build noisy kernel matrix and Cholesky
        K_noisy = K_kernel + np.diag(sigma_eff_sq) + jitter * np.eye(n)
        L = np.linalg.cholesky(K_noisy)

        # Solve for alpha = K_noisy⁻¹ @ residuals
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, residuals))

        # --- E-step: compute posterior outlier probability ---
        # Posterior residual: eᵢ = sigma_eff_sq[i] * alpha[i]
        e = sigma_eff_sq * alpha

        # Log-likelihood ratio: inlier vs outlier component
        # log p(y|outlier) - log p(y|inlier)
        #   = 0.5*log(σ²/(κσ²)) + 0.5*e²*(1/σ² - 1/(κσ²))
        #   = -0.5*log(κ) + 0.5*e²*(1/σ²)*(1 - 1/κ)
        log_r = (
            -0.5 * np.log(kappa)
            + 0.5 * e**2 / noise_variance * (1.0 - 1.0 / kappa)
        )

        # Posterior: w = sigmoid(log_prior_ratio + log_r)
        logit = np.clip(log_prior_ratio + log_r, -500.0, 500.0)
        w = 1.0 / (1.0 + np.exp(-logit))

        # Check convergence
        max_change = np.max(np.abs(w - w_old))
        if max_change < config.em_convergence_tol:
            logger.debug(
                f"Contaminated EM converged at iteration {iteration + 1} "
                f"(max_change={max_change:.6f})"
            )
            break
    else:
        logger.debug(
            f"Contaminated EM reached max iterations ({config.max_em_iterations}), "
            f"max_change={max_change:.6f}"
        )

    # Final M-step with converged weights (ensure L matches final sigma_eff_sq)
    sigma_eff_sq = noise_variance * (1.0 + w * (kappa - 1.0))
    K_noisy = K_kernel + np.diag(sigma_eff_sq) + jitter * np.eye(n)
    L = np.linalg.cholesky(K_noisy)

    return sigma_eff_sq, w, L


def run_contaminated_em_torch(
    K_kernel,  # torch.Tensor: (n, n) kernel matrix WITHOUT noise
    y: np.ndarray,
    noise_variance: np.ndarray,
    mean_value: Union[float, np.ndarray],
    config: LikelihoodConfig,
    jitter: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run contaminated normal EM algorithm (PyTorch, GPU-accelerated).

    Same algorithm as :func:`run_contaminated_em` but uses PyTorch for
    Cholesky and triangular solves on GPU.  Returns NumPy arrays.

    Args:
        K_kernel: (n, n) torch.Tensor kernel matrix **without** noise,
            already on the target device.
        y: (n,) training targets (NumPy).
        noise_variance: (n,) per-point measurement noise σᵢ² (NumPy).
        mean_value: Mean function values (scalar or NumPy array).
        config: Likelihood configuration.
        jitter: Diagonal jitter for numerical stability.

    Returns:
        (sigma_eff_sq, outlier_prob, L) — all NumPy arrays.
    """
    import torch

    device = K_kernel.device
    dtype = K_kernel.dtype
    n = len(y)

    eps = config.contamination_fraction
    kappa = config.contamination_scale

    residuals_np = y - np.asarray(mean_value)
    residuals_t = torch.tensor(residuals_np, dtype=dtype, device=device)
    noise_var_t = torch.tensor(noise_variance, dtype=dtype, device=device)
    eye_t = torch.eye(n, dtype=dtype, device=device)

    log_prior_ratio = np.log(eps) - np.log(1.0 - eps)

    # Initialize outlier weights at prior
    w_t = torch.full((n,), eps, dtype=dtype, device=device)

    for iteration in range(config.max_em_iterations):
        w_old_t = w_t.clone()

        # --- M-step ---
        sigma_eff_sq_t = noise_var_t * (1.0 + w_t * (kappa - 1.0))
        K_noisy = K_kernel + torch.diag(sigma_eff_sq_t) + jitter * eye_t
        L_t = torch.linalg.cholesky(K_noisy)

        # Solve for alpha
        z = torch.linalg.solve_triangular(
            L_t, residuals_t.unsqueeze(1), upper=False
        ).squeeze(1)
        alpha_t = torch.linalg.solve_triangular(
            L_t.T, z.unsqueeze(1), upper=True
        ).squeeze(1)

        # --- E-step ---
        e_t = sigma_eff_sq_t * alpha_t
        log_r_t = (
            -0.5 * np.log(kappa)
            + 0.5 * e_t**2 / noise_var_t * (1.0 - 1.0 / kappa)
        )

        logit_t = torch.clamp(log_prior_ratio + log_r_t, -500.0, 500.0)
        w_t = torch.sigmoid(logit_t)

        # Check convergence
        max_change = torch.max(torch.abs(w_t - w_old_t)).item()
        if max_change < config.em_convergence_tol:
            logger.debug(
                f"Contaminated EM (torch) converged at iteration {iteration + 1} "
                f"(max_change={max_change:.6f})"
            )
            break
    else:
        logger.debug(
            f"Contaminated EM (torch) reached max iterations "
            f"({config.max_em_iterations}), max_change={max_change:.6f}"
        )

    # Final M-step with converged weights
    sigma_eff_sq_t = noise_var_t * (1.0 + w_t * (kappa - 1.0))
    K_noisy = K_kernel + torch.diag(sigma_eff_sq_t) + jitter * eye_t
    L_t = torch.linalg.cholesky(K_noisy)

    # Convert to NumPy
    sigma_eff_sq = sigma_eff_sq_t.cpu().numpy()
    outlier_prob = w_t.cpu().numpy()
    L = L_t.cpu().numpy()

    return sigma_eff_sq, outlier_prob, L
