"""Ohta-Kawasaki solver for block-copolymer microphase separation.

Extends the Cahn-Hilliard equation with a long-range interaction term:

    ∂φ/∂t = M ∇²μ
    μ     = f'(φ) − κ ∇²φ + α (−∇²)⁻¹(φ − φ̄)

The additional long-range term penalises macroscopic phase separation,
favouring periodic microstructures (lamellae, cylinders, spheres, …).

In Fourier space the semi-implicit update becomes:

    φ̂ᵏ⁺¹ = (φ̂ᵏ − Δt M k² ĝᵏ) / (1 + Δt M (k²(C + κk²) + α))   [k ≠ 0]
    φ̂₀ⁿ⁺¹ = φ̂₀ⁿ                                                    [k = 0, mass conservation]

where g = f'(φ) − Cφ  (stabilised explicit part).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .cahn_hilliard import CahnHilliardSolver, SimulationParams
from .free_energy import FreeEnergy


class OhtaKawasakiSolver(CahnHilliardSolver):
    """Spectral solver for the Ohta-Kawasaki model (2D / 3D).

    Inherits the stabilised semi-implicit FFT scheme from
    :class:`CahnHilliardSolver` and adds the long-range term
    α(−∇²)⁻¹(φ − φ̄) to the chemical potential.

    Parameters
    ----------
    params : SimulationParams
        Simulation parameters.
    free_energy : FreeEnergy
        Free energy model (e.g. :class:`DoubleWell`).
    alpha : float
        Long-range interaction strength.  Larger values favour
        finer microstructures.
    device : torch.device | str
        Computation device.
    """

    def __init__(
        self,
        params: SimulationParams,
        free_energy: FreeEnergy,
        alpha: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.alpha = alpha
        super().__init__(params, free_energy, device=device)

        # Add the implicit long-range contribution to the denominator.
        # For k ≠ 0:  denom += Δt·M·α
        # For k = 0:  unchanged  (the long-range term vanishes by mass conservation)
        k2_nonzero = (self._k2 > 0).to(self._denom.dtype)
        self._denom = self._denom + params.dt * params.mobility * alpha * k2_nonzero

    # step() is inherited unchanged — only the denominator differs.

    def compute_total_free_energy(self, phi: Tensor) -> float:
        """Compute total free energy including the long-range term.

        F = F_CH + (α/2) ∫ (φ−φ̄)(−∇²)⁻¹(φ−φ̄) dr

        The long-range contribution in Fourier space is:

            F_lr = (α/2) · (dV/N^d) · Σ_{k≠0} |ψ̂_k|² / k²
        """
        fe_ch = super().compute_total_free_energy(phi)

        psi = phi - phi.mean()
        psi_hat = torch.fft.fftn(psi)

        k2_inv = torch.zeros_like(self._k2)
        nonzero = self._k2 > 0
        k2_inv[nonzero] = 1.0 / self._k2[nonzero]

        lr_sum = (psi_hat.abs() ** 2 * k2_inv).sum()

        p = self.params
        dV = p.dx ** p.ndim
        n_total = phi.numel()
        fe_lr = 0.5 * self.alpha * lr_sum.item() * dV / n_total

        return fe_ch + fe_lr
