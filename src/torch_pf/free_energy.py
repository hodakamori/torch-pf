"""Free energy models and functionals for phase-field simulations.

Local free energy densities
---------------------------
``FloryHuggins`` -- Regularized Flory-Huggins for polymer blends.
``DoubleWell``   -- Polynomial double-well f(φ) = W φ²(1−φ)².

Free energy functional
----------------------
``FreeEnergyFunctional`` -- Bundles the local free energy density f(φ),
the gradient coefficient κ, and the optional long-range (Ohta-Kawasaki)
interaction strength α into a single thermodynamic description:

    F[φ] = ∫ [f(φ) + (κ/2)|∇φ|²] dr + (α/2) ∫∫ G(φ−φ̄)(φ−φ̄) dr dr'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FreeEnergy(ABC):
    """Abstract base class for local free energy densities f(φ)."""

    @abstractmethod
    def free_energy_density(self, phi: Tensor) -> Tensor: ...

    @abstractmethod
    def chemical_potential(self, phi: Tensor) -> Tensor: ...

    @abstractmethod
    def second_derivative(self, phi: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Regularized Flory-Huggins
# ---------------------------------------------------------------------------


def _safe_log(x: Tensor, eps: float = 0.01) -> Tensor:
    """Logarithm with smooth quadratic extension for x < eps.

    For x >= eps:  returns log(x)
    For x <  eps:  returns log(eps) + (x - eps)/eps - 0.5*((x - eps)/eps)^2
    """
    safe = x.clamp(min=eps)
    log_val = safe.log()
    # Quadratic extension below eps
    mask = x < eps
    if mask.any():
        t = (x[mask] - eps) / eps
        log_val = log_val.clone()
        log_val[mask] = eps.log() if isinstance(eps, Tensor) else torch.tensor(eps).log().item()
        log_val[mask] = log_val[mask] + t - 0.5 * t * t
    return log_val


def _safe_log_deriv(x: Tensor, eps: float = 0.01) -> Tensor:
    """Derivative of _safe_log: 1/x for x >= eps, linear for x < eps."""
    safe = x.clamp(min=eps)
    val = 1.0 / safe
    mask = x < eps
    if mask.any():
        val = val.clone()
        val[mask] = (1.0 / eps) * (1.0 - (x[mask] - eps) / eps)
    return val


class FloryHuggins(FreeEnergy):
    """Regularized Flory-Huggins free energy for binary polymer blends.

    f(φ) = (φ/N_A) ln(φ) + ((1−φ)/N_B) ln(1−φ) + χ φ(1−φ)

    The logarithms are smoothly extended outside [ε, 1−ε] to avoid
    numerical divergence, making the model safe for spectral solvers.

    Parameters
    ----------
    chi : float
        Flory-Huggins interaction parameter.
    n_a : float
        Degree of polymerization of polymer A.
    n_b : float
        Degree of polymerization of polymer B.
    eps : float
        Regularization width for the logarithm (default 0.01).
    """

    def __init__(
        self,
        chi: float,
        n_a: float = 100.0,
        n_b: float = 100.0,
        eps: float = 0.01,
    ) -> None:
        self.chi = chi
        self.n_a = n_a
        self.n_b = n_b
        self.eps = eps

    @property
    def chi_critical(self) -> float:
        """Critical χ for the blend."""
        return 0.5 * (1.0 / self.n_a**0.5 + 1.0 / self.n_b**0.5) ** 2

    @property
    def phi_critical(self) -> float:
        """Critical composition."""
        na_sqrt = self.n_a**0.5
        nb_sqrt = self.n_b**0.5
        return nb_sqrt / (na_sqrt + nb_sqrt)

    def free_energy_density(self, phi: Tensor) -> Tensor:
        eps = self.eps
        return (
            (phi / self.n_a) * _safe_log(phi, eps)
            + ((1.0 - phi) / self.n_b) * _safe_log(1.0 - phi, eps)
            + self.chi * phi * (1.0 - phi)
        )

    def chemical_potential(self, phi: Tensor) -> Tensor:
        eps = self.eps
        return (
            (1.0 / self.n_a) * (_safe_log(phi, eps) + 1.0)
            - (1.0 / self.n_b) * (_safe_log(1.0 - phi, eps) + 1.0)
            + self.chi * (1.0 - 2.0 * phi)
        )

    def second_derivative(self, phi: Tensor) -> Tensor:
        """d²f/dφ² (for spinodal analysis and stabilization estimate)."""
        eps = self.eps
        return (
            (1.0 / self.n_a) * _safe_log_deriv(phi, eps)
            + (1.0 / self.n_b) * _safe_log_deriv(1.0 - phi, eps)
            - 2.0 * self.chi
        )


# ---------------------------------------------------------------------------
# Polynomial double-well
# ---------------------------------------------------------------------------


class DoubleWell(FreeEnergy):
    """Double-well free energy f(φ) = W φ²(1−φ)².

    A simple and numerically robust model commonly used in Cahn-Hilliard
    simulations.  The wells are at φ = 0 and φ = 1 with barrier height W/16.

    Parameters
    ----------
    W : float
        Barrier height parameter (controls the driving force for separation).
    """

    def __init__(self, W: float = 1.0) -> None:
        self.W = W

    def free_energy_density(self, phi: Tensor) -> Tensor:
        return self.W * phi**2 * (1.0 - phi) ** 2

    def chemical_potential(self, phi: Tensor) -> Tensor:
        # f'(φ) = 2W φ(1−φ)(1−2φ)
        return 2.0 * self.W * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

    def second_derivative(self, phi: Tensor) -> Tensor:
        # f''(φ) = 2W(1 − 6φ + 6φ²)
        return 2.0 * self.W * (1.0 - 6.0 * phi + 6.0 * phi**2)


# ---------------------------------------------------------------------------
# Free energy functional  F[φ]
# ---------------------------------------------------------------------------


@dataclass
class FreeEnergyFunctional:
    """Complete free energy functional F[φ].

    F[φ] = ∫ [f(φ) + (κ/2)|∇φ|²] dr  +  (α/2) ∫∫ G(φ−φ̄)(φ−φ̄) dr dr'

    Parameters
    ----------
    local : FreeEnergy
        Local free energy density f(φ).
    kappa : float
        Gradient energy coefficient κ  (controls interface width / energy).
    alpha : float
        Long-range interaction strength (Ohta-Kawasaki).
        Set to 0 for standard Cahn-Hilliard behaviour.
    """

    local: FreeEnergy
    kappa: float
    alpha: float = 0.0
