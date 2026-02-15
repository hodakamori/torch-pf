"""Wall boundary conditions for phase-field simulations.

Provides pluggable wall conditions that contribute to the chemical
potential and free energy.  Two implementations are available:

VolumePenaltyWall
    Original penalty method: μ_wall = λ Ω(r)(φ − φ_wall).
    Simple but the contact angle depends indirectly on λ and φ_wall.

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


# ---------------------------------------------------------------------------
# Volume-penalty wall (original method)
# ---------------------------------------------------------------------------


class VolumePenaltyWall(WallCondition):
    """Volume-penalty wall confinement.

    Adds a penalty term to the chemical potential inside the wall region:

        μ_wall = λ Ω(r) (φ − φ_wall)

    and a corresponding energy:

        F_wall = (λ/2) ∫ Ω(r) (φ − φ_wall)² dr

    Parameters
    ----------
    mask : Tensor
        Smooth wall mask Ω(r): 0 = fluid, 1 = wall.
    phi_wall : float
        Preferred composition inside the wall.
    penalty : float
        Penalty strength λ.
    """

    def __init__(
        self,
        mask: Tensor,
        phi_wall: float = 0.5,
        penalty: float = 10.0,
    ) -> None:
        self.mask = mask
        self.phi_wall = phi_wall
        self.penalty = penalty

    def to(self, device: torch.device | str) -> VolumePenaltyWall:
        return VolumePenaltyWall(
            self.mask.to(device),
            phi_wall=self.phi_wall,
            penalty=self.penalty,
        )

    def chemical_potential_contribution(self, phi: Tensor) -> Tensor:
        return self.penalty * self.mask * (phi - self.phi_wall)

    def energy_contribution(self, phi: Tensor, dV: float) -> float:
        return (
            0.5
            * self.penalty
            * (self.mask * (phi - self.phi_wall) ** 2).sum().item()
            * dV
        )

    def stabilization_estimate(self) -> float:
        return self.penalty


# ---------------------------------------------------------------------------
# Surface-energy (wetting) wall
# ---------------------------------------------------------------------------


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
