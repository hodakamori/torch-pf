"""Wall boundary conditions for phase-field simulations.

Provides a pluggable wall condition that contributes to the chemical
potential and free energy via surface energy coupling.

SurfaceEnergyWall
    Surface energy (wetting) method: μ_wall = −γ |∇Ω|.
    The parameter γ = σ cos(θ) directly controls the contact angle θ,
    where σ is the interfacial tension between the two bulk phases.

    γ > 0  →  wall attracts φ = 1 phase  (θ < 90°)
    γ < 0  →  wall attracts φ = 0 phase  (θ > 90°)
    γ = 0  →  neutral wetting             (θ = 90°)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class WallCondition(ABC):
    """Abstract base class for wall boundary conditions."""

    @abstractmethod
    def to(self, device: torch.device | str) -> WallCondition:
        """Return a copy with all tensors on *device*."""
        ...

    @abstractmethod
    def chemical_potential_contribution(self, phi: Tensor) -> Tensor:
        """Extra term added to the chemical potential due to the wall."""
        ...

    @abstractmethod
    def energy_contribution(self, phi: Tensor, dV: float) -> float:
        """Wall contribution to the total free energy."""
        ...

    @abstractmethod
    def stabilization_estimate(self) -> float:
        """Additional stabilisation constant needed for this wall type."""
        ...


class SurfaceEnergyWall(WallCondition):
    """Surface-energy wetting wall condition.

    Adds a surface free energy at the wall–fluid interface:

        F_surface = −γ ∫ φ |∇Ω| dr

    whose functional derivative gives a chemical potential contribution:

        μ_surface = −γ |∇Ω|

    Here |∇Ω| acts as a surface delta function, concentrating the effect
    at the wall–fluid interface.

    The parameter γ = σ cos(θ) directly encodes the contact angle θ:

    * γ > 0: wall attracts the φ = 1 phase (contact angle < 90°)
    * γ < 0: wall attracts the φ = 0 phase (contact angle > 90°)
    * γ = 0: neutral wetting (contact angle = 90°)

    Parameters
    ----------
    mask : Tensor
        Smooth wall mask Ω(r): 0 = fluid, 1 = wall.
    gamma : float
        Surface energy coupling strength γ = σ cos(θ).
    dx : float
        Grid spacing (must match the solver's ``GridParams.dx``).
    """

    def __init__(
        self,
        mask: Tensor,
        gamma: float,
        dx: float = 1.0,
    ) -> None:
        self.mask = mask
        self.gamma = gamma
        self.dx = dx
        self._surface_delta = self._compute_surface_delta(mask, dx)

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _compute_surface_delta(mask: Tensor, dx: float) -> Tensor:
        """Compute |∇Ω| via spectral differentiation."""
        k_1d = [
            torch.fft.fftfreq(n, d=dx / (2.0 * torch.pi)).to(mask.device)
            for n in mask.shape
        ]
        k_grids = torch.meshgrid(*k_1d, indexing="ij")

        mask_hat = torch.fft.fftn(mask)
        grad_sq: Tensor = sum(  # type: ignore[assignment]
            torch.fft.ifftn(1j * kg * mask_hat).real ** 2 for kg in k_grids
        )
        return grad_sq.sqrt()

    # -- WallCondition interface -------------------------------------------

    def to(self, device: torch.device | str) -> SurfaceEnergyWall:
        wall = SurfaceEnergyWall.__new__(SurfaceEnergyWall)
        wall.mask = self.mask.to(device)
        wall.gamma = self.gamma
        wall.dx = self.dx
        wall._surface_delta = self._surface_delta.to(device)
        return wall

    def chemical_potential_contribution(self, phi: Tensor) -> Tensor:
        return -self.gamma * self._surface_delta

    def energy_contribution(self, phi: Tensor, dV: float) -> float:
        return -self.gamma * (phi * self._surface_delta).sum().item() * dV

    def stabilization_estimate(self) -> float:
        # Linear coupling h(φ) = −γφ: h''(φ) = 0, no extra stabilisation.
        return 0.0
