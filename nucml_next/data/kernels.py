"""
Kernel Abstraction for GP Outlier Detection
============================================

Provides a kernel abstraction layer so that the GP code in
``experiment_gp.py`` and ``calibration.py`` can use any kernel
(RBF, Gibbs, etc.) without hardcoding the kernel formula.

Key Classes:
    KernelConfig: Configuration dataclass for all kernel types.
    RBFKernel: Stationary RBF (squared exponential) kernel.
    GibbsKernel: Nonstationary Gibbs kernel with physics-informed
        energy-dependent lengthscale from RIPL-3 level density data.

Key Functions:
    build_kernel: Factory that constructs the right kernel from config.

Design Decisions
----------------
- **Outputscale is NOT optimised.** It is estimated from
  ``Var(residuals) - mean(noise)`` in ``experiment_gp.py`` and injected
  into the kernel.  Only the kernel-specific parameters are optimised
  via Wasserstein calibration.

- **Kernel does NO file I/O.** The RIPL-3 interpolator is injected
  from outside (by ``experiment_outlier.py`` which knows (Z, A)).

- **Dual NumPy/PyTorch paths** mirror the existing codebase pattern.

- ``n_optimizable_params`` separates the Wasserstein search space from
  the total parameter count:
    - RBF: 1 (lengthscale)
    - Gibbs: 2 (a₀, a₁ correction terms)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KernelConfig:
    """Configuration for GP kernels.

    Attributes
    ----------
    kernel_type : str
        ``'rbf'`` (default) or ``'gibbs'``.
    outputscale : float
        Signal variance σ².  Estimated from data, never optimised.
    lengthscale : float
        Lengthscale for RBF kernel.  Optimised via Wasserstein.
    gibbs_correction_a0 : float
        Additive correction to log D(E) in Gibbs lengthscale function.
    gibbs_correction_a1 : float
        Slope correction × log_E in Gibbs lengthscale function.
    gibbs_lengthscale_bounds : tuple
        (min, max) bounds for the effective lengthscale after softplus.
    ripl_log_D_interpolator : callable or None
        Injected callable: ``log₁₀(E [eV]) → log₁₀(D [eV])``.
        Required for Gibbs kernel.  When None and kernel_type='gibbs',
        ``build_kernel()`` falls back to RBF with a warning.
    """
    kernel_type: str = 'rbf'
    outputscale: float = 1.0
    lengthscale: float = 1.0
    gibbs_correction_a0: float = 0.0
    gibbs_correction_a1: float = 0.0
    gibbs_lengthscale_bounds: Tuple[float, float] = (0.01, 10.0)
    ripl_log_D_interpolator: Optional[Callable] = None


class Kernel(ABC):
    """Abstract base class for GP kernels.

    All kernels must provide:
    - ``compute_matrix(x1, x2)`` — NumPy kernel matrix
    - ``compute_matrix_torch(x1, x2, device)`` — PyTorch kernel matrix
    - ``n_optimizable_params`` — how many params the Wasserstein optimizer
      searches over (excludes outputscale)
    - ``get_optimizable_params()`` / ``set_optimizable_params()``
    """

    def __init__(self, config: KernelConfig):
        self.config = config

    @abstractmethod
    def compute_matrix(
        self,
        x1: np.ndarray,
        x2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute kernel matrix K(x1, x2) using NumPy.

        Parameters
        ----------
        x1 : ndarray, shape (n1,)
        x2 : ndarray, shape (n2,), or None (same as x1)

        Returns
        -------
        K : ndarray, shape (n1, n2)
        """
        ...

    @abstractmethod
    def compute_matrix_torch(
        self,
        x1,  # torch.Tensor
        x2,  # torch.Tensor or None
        device: str = 'cpu',
    ):
        """Compute kernel matrix using PyTorch.

        Parameters
        ----------
        x1 : Tensor, shape (n1,)
        x2 : Tensor, shape (n2,), or None (same as x1)
        device : str

        Returns
        -------
        K : Tensor, shape (n1, n2)
        """
        ...

    @abstractmethod
    def n_optimizable_params(self) -> int:
        """Number of parameters optimised by Wasserstein calibration."""
        ...

    @abstractmethod
    def get_optimizable_params(self) -> np.ndarray:
        """Return current optimisable parameters as a flat array."""
        ...

    @abstractmethod
    def set_optimizable_params(self, params: np.ndarray) -> None:
        """Set optimisable parameters from a flat array."""
        ...

    def get_all_params(self) -> Dict[str, float]:
        """Return all kernel parameters (including non-optimisable)."""
        return {
            'kernel_type': self.config.kernel_type,
            'outputscale': self.config.outputscale,
        }

    def prior_variance(self) -> float:
        """Prior variance at a single point: K(x, x) = outputscale."""
        return self.config.outputscale


class RBFKernel(Kernel):
    """Stationary RBF (squared exponential) kernel.

    K(xᵢ, xⱼ) = σ² · exp(-0.5 · (xᵢ − xⱼ)² / ℓ²)

    Wraps the exact same math previously hardcoded in
    ``experiment_gp._compute_kernel_matrix()`` and the inline PyTorch
    code in ``_build_prediction_cache_torch()``.

    ``n_optimizable_params = 1`` (lengthscale only).
    """

    def compute_matrix(
        self,
        x1: np.ndarray,
        x2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if x2 is None:
            x2 = x1

        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()

        diff = x1[:, None] - x2[None, :]
        K = self.config.outputscale * np.exp(
            -0.5 * diff ** 2 / self.config.lengthscale ** 2
        )
        return K

    def compute_matrix_torch(self, x1, x2=None, device='cpu'):
        import torch

        if x2 is None:
            x2 = x1

        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        K = self.config.outputscale * torch.exp(
            -0.5 * diff.pow(2) / (self.config.lengthscale ** 2)
        )
        return K

    def n_optimizable_params(self) -> int:
        return 1

    def get_optimizable_params(self) -> np.ndarray:
        return np.array([self.config.lengthscale])

    def set_optimizable_params(self, params: np.ndarray) -> None:
        self.config.lengthscale = float(params[0])

    def get_all_params(self) -> Dict[str, float]:
        d = super().get_all_params()
        d['lengthscale'] = self.config.lengthscale
        return d


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _softplus_torch(x):
    """PyTorch softplus."""
    import torch
    return torch.nn.functional.softplus(x)


class GibbsKernel(Kernel):
    """Nonstationary Gibbs kernel with physics-informed lengthscale.

    K(xᵢ, xⱼ) = σ² · √(2ℓᵢℓⱼ / (ℓᵢ² + ℓⱼ²)) · exp(−(xᵢ−xⱼ)² / (ℓᵢ²+ℓⱼ²))

    where the energy-dependent lengthscale is:

        ℓ(log_E) = softplus(log_D(log_E) + a₀ + a₁ · log_E)

    and ``log_D`` is the RIPL-3 mean level spacing interpolator injected
    via ``config.ripl_log_D_interpolator``.

    ``n_optimizable_params = 2`` (a₀, a₁).
    """

    def _compute_lengthscales(self, x: np.ndarray) -> np.ndarray:
        """Compute ℓ(x) for each point using RIPL-3 + corrections.

        Parameters
        ----------
        x : ndarray, shape (n,)
            log₁₀(E [eV]) values.

        Returns
        -------
        ell : ndarray, shape (n,)
            Lengthscale at each point.
        """
        log_D = self.config.ripl_log_D_interpolator(x)
        raw = log_D + self.config.gibbs_correction_a0 + self.config.gibbs_correction_a1 * x
        ell = _softplus(raw)
        # Apply bounds
        lb, ub = self.config.gibbs_lengthscale_bounds
        ell = np.clip(ell, lb, ub)
        return ell

    def _compute_lengthscales_torch(self, x):
        """PyTorch version of lengthscale computation."""
        import torch

        # Evaluate RIPL-3 interpolator on CPU numpy, then convert
        x_np = x.detach().cpu().numpy()
        log_D_np = self.config.ripl_log_D_interpolator(x_np)
        log_D = torch.tensor(log_D_np, dtype=x.dtype, device=x.device)

        raw = log_D + self.config.gibbs_correction_a0 + self.config.gibbs_correction_a1 * x
        ell = _softplus_torch(raw)
        lb, ub = self.config.gibbs_lengthscale_bounds
        ell = torch.clamp(ell, lb, ub)
        return ell

    def compute_matrix(
        self,
        x1: np.ndarray,
        x2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if x2 is None:
            x2 = x1

        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()

        ell1 = self._compute_lengthscales(x1)  # (n1,)
        ell2 = self._compute_lengthscales(x2)  # (n2,)

        # Broadcast: ell1[:, None] vs ell2[None, :]
        li = ell1[:, None]  # (n1, 1)
        lj = ell2[None, :]  # (1, n2)

        li2 = li ** 2
        lj2 = lj ** 2
        sum_l2 = li2 + lj2

        # Gibbs kernel formula
        # Normalisation factor: sqrt(2 * li * lj / (li² + lj²))
        norm = np.sqrt(2.0 * li * lj / sum_l2)

        # Squared distance
        diff = x1[:, None] - x2[None, :]

        # Exponential factor: exp(-(xi - xj)² / (li² + lj²))
        K = self.config.outputscale * norm * np.exp(-diff ** 2 / sum_l2)

        return K

    def compute_matrix_torch(self, x1, x2=None, device='cpu'):
        import torch

        if x2 is None:
            x2 = x1

        ell1 = self._compute_lengthscales_torch(x1)  # (n1,)
        ell2 = self._compute_lengthscales_torch(x2)  # (n2,)

        li = ell1.unsqueeze(1)  # (n1, 1)
        lj = ell2.unsqueeze(0)  # (1, n2)

        li2 = li ** 2
        lj2 = lj ** 2
        sum_l2 = li2 + lj2

        norm = torch.sqrt(2.0 * li * lj / sum_l2)

        diff = x1.unsqueeze(1) - x2.unsqueeze(0)

        K = self.config.outputscale * norm * torch.exp(-diff ** 2 / sum_l2)

        return K

    def n_optimizable_params(self) -> int:
        return 2

    def get_optimizable_params(self) -> np.ndarray:
        return np.array([
            self.config.gibbs_correction_a0,
            self.config.gibbs_correction_a1,
        ])

    def set_optimizable_params(self, params: np.ndarray) -> None:
        self.config.gibbs_correction_a0 = float(params[0])
        self.config.gibbs_correction_a1 = float(params[1])

    def get_all_params(self) -> Dict[str, float]:
        d = super().get_all_params()
        d['gibbs_correction_a0'] = self.config.gibbs_correction_a0
        d['gibbs_correction_a1'] = self.config.gibbs_correction_a1
        return d

    def prior_variance(self) -> float:
        """Prior variance K(x, x) = outputscale (norm factor = 1 on diagonal)."""
        return self.config.outputscale


def build_kernel(config: Optional[KernelConfig] = None) -> Kernel:
    """Factory: construct the appropriate kernel from config.

    Parameters
    ----------
    config : KernelConfig or None
        If None, returns a default RBFKernel.

    Returns
    -------
    Kernel
        RBFKernel or GibbsKernel instance.
    """
    if config is None:
        return RBFKernel(KernelConfig())

    if config.kernel_type == 'rbf':
        return RBFKernel(config)

    if config.kernel_type == 'gibbs':
        if config.ripl_log_D_interpolator is None:
            logger.warning(
                "Gibbs kernel requested but no RIPL-3 interpolator provided; "
                "falling back to RBF kernel"
            )
            return RBFKernel(config)
        return GibbsKernel(config)

    raise ValueError(
        f"Unknown kernel_type: {config.kernel_type!r}. "
        f"Expected 'rbf' or 'gibbs'."
    )
