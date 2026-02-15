"""Spectral solver for phase-field equations (Cahn-Hilliard / Ohta-Kawasaki).

Solves conserved-dynamics (Model B) equations of the form

    ∂φ/∂t = M ∇² (δF/δφ)

where the free energy functional F[φ] is described by a
``FreeEnergyFunctional`` object.

Standard Cahn-Hilliard (α = 0)
------------------------------
    δF/δφ = f'(φ) − κ ∇²φ

Ohta-Kawasaki (α > 0)
---------------------
    δF/δφ = f'(φ) − κ ∇²φ + α (−∇²)⁻¹(φ − φ̄)

Wall confinement (volume penalty)
---------------------------------
When a wall mask Ω(r) is provided, a penalty term is added to the
chemical potential:

    μ_eff = μ + λ Ω(r) (φ − φ_wall)

This preserves mass conservation (∂φ/∂t = ∇²μ_eff) and allows
control of the wetting angle via φ_wall.

A stabilised semi-implicit spectral scheme is used:

    φ̂^{n+1} = (φ̂^n − Δt M k² ĝ^n) / D_k
    g^n      = f'(φ^n) + λ Ω (φ^n − φ_wall) − C φ^n

where the denominator D_k accounts for both gradient and long-range terms:

    D_k = 1 + Δt M k² (C + κ k²)        [α = 0]
    D_k = 1 + Δt M (k² (C + κ k²) + α)  [α > 0, k ≠ 0]

The stabilisation constant C is automatically chosen from max |f''|
(plus the wall penalty λ when a wall is present).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .free_energy import FreeEnergyFunctional


@dataclass
class GridParams:
    """Numerical grid and time-stepping parameters.

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions, e.g. (128, 128) for 2D or (64, 64, 64) for 3D.
    dx : float
        Grid spacing (uniform in all directions).
    dt : float
        Time step size.
    """

    shape: tuple[int, ...] = (128, 128)
    dx: float = 1.0
    dt: float = 0.5

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (2 or 3)."""
        return len(self.shape)


class SpectralSolver:
    """Unified spectral solver for Cahn-Hilliard and Ohta-Kawasaki (2D / 3D).

    Works with any ``FreeEnergyFunctional``.  When ``functional.alpha == 0``
    the solver reduces to standard Cahn-Hilliard; otherwise the Ohta-Kawasaki
    long-range term is included automatically.

    Parameters
    ----------
    functional : FreeEnergyFunctional
        Complete thermodynamic description (f(φ), κ, α).
    grid : GridParams
        Numerical grid and time-stepping parameters.
    mobility : float
        Mobility coefficient M (kinetic parameter).
    device : torch.device | str
        Computation device ('cpu' or 'cuda').
    stabilization : float | None
        Stabilisation constant C.  If None, estimated from max |f''|.
    wall : Tensor | None
        Smooth wall mask Ω(r): 0 = fluid, 1 = wall.
        When provided, a volume-penalty term is added to the chemical
        potential to confine φ inside the wall.
    wall_phi : float
        Preferred composition at the wall surface (controls wetting).
        0.5 = neutral; 0 or 1 = preferential wetting by one phase.
    wall_penalty : float
        Penalty strength λ.  Larger values enforce the wall more
        strictly but require a smaller time step for stability.
    """

    def __init__(
        self,
        functional: FreeEnergyFunctional,
        grid: GridParams,
        mobility: float = 1.0,
        device: torch.device | str = "cpu",
        stabilization: float | None = None,
        wall: Tensor | None = None,
        wall_phi: float = 0.5,
        wall_penalty: float = 10.0,
    ) -> None:
        self.functional = functional
        self.grid = grid
        self.mobility = mobility
        self.device = torch.device(device)

        # Wall confinement
        self._wall: Tensor | None = wall.to(self.device) if wall is not None else None
        self._wall_phi = wall_phi
        self._wall_penalty = wall_penalty

        # Stabilisation constant
        if stabilization is not None:
            self._C = stabilization
        else:
            self._C = self._estimate_stabilization()

        # Precompute wavenumber grids
        self._k_grids = self._build_k_grids()
        self._k2: Tensor = sum(kg**2 for kg in self._k_grids)  # type: ignore[assignment]

        # Denominator: standard CH part
        self._denom = (
            1.0
            + grid.dt * mobility * self._k2 * (self._C + functional.kappa * self._k2)
        )

        # Long-range (OK) contribution for k ≠ 0
        if functional.alpha > 0:
            k2_nonzero = (self._k2 > 0).to(self._denom.dtype)
            self._denom = self._denom + grid.dt * mobility * functional.alpha * k2_nonzero

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_stabilization(self) -> float:
        """Estimate stabilisation constant from max |f''| in [0.01, 0.99]."""
        phi_test = torch.linspace(0.01, 0.99, 2000)
        d2f = self.functional.local.second_derivative(phi_test)
        C = max(float(d2f.abs().max()) * 1.1, 1.0)
        # Account for wall penalty in stabilisation
        if self._wall is not None:
            C += self._wall_penalty
        return C

    def _build_k_grids(self) -> list[Tensor]:
        """Build wavenumber component grids for each spatial dimension."""
        g = self.grid
        k_1d = [
            torch.fft.fftfreq(n, d=g.dx / (2.0 * torch.pi)).to(self.device)
            for n in g.shape
        ]
        grids = torch.meshgrid(*k_1d, indexing="ij")
        return list(grids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, phi: Tensor) -> Tensor:
        """Advance the field by one time step.

        Parameters
        ----------
        phi : Tensor
            Current volume fraction field, shape matching ``grid.shape``.

        Returns
        -------
        Tensor
            Updated volume fraction field.
        """
        g = self.grid
        mu = self.functional.local.chemical_potential(phi)
        # Wall penalty: λ Ω(r) (φ − φ_wall)
        if self._wall is not None:
            mu = mu + self._wall_penalty * self._wall * (phi - self._wall_phi)
        g_explicit = mu - self._C * phi
        g_hat = torch.fft.fftn(g_explicit)
        phi_hat = torch.fft.fftn(phi)

        phi_hat_new = (phi_hat - g.dt * self.mobility * self._k2 * g_hat) / self._denom
        return torch.fft.ifftn(phi_hat_new).real

    def run(
        self,
        phi0: Tensor,
        n_steps: int,
        save_interval: int = 100,
    ) -> list[tuple[int, Tensor]]:
        """Run the simulation for multiple steps.

        Parameters
        ----------
        phi0 : Tensor
            Initial volume fraction field.
        n_steps : int
            Total number of time steps.
        save_interval : int
            Save the field every this many steps.

        Returns
        -------
        list of (step, Tensor)
            Snapshots of the field at saved time steps.
        """
        phi = phi0.to(self.device)
        snapshots: list[tuple[int, Tensor]] = [(0, phi.clone().cpu())]

        for step in range(1, n_steps + 1):
            phi = self.step(phi)
            if step % save_interval == 0 or step == n_steps:
                snapshots.append((step, phi.clone().cpu()))

        return snapshots

    def compute_total_free_energy(self, phi: Tensor) -> float:
        """Compute the total free energy F[φ].

        Includes bulk, gradient, long-range (if α > 0), and wall penalty
        contributions.

        Parameters
        ----------
        phi : Tensor
            Volume fraction field.

        Returns
        -------
        float
            Total free energy.
        """
        f = self.functional
        g = self.grid

        # Bulk: ∫ f(φ) dr
        f_bulk = f.local.free_energy_density(phi)

        # Gradient: (κ/2) ∫ |∇φ|² dr
        phi_hat = torch.fft.fftn(phi)
        grad_sq = sum(
            torch.fft.ifftn(1j * kg * phi_hat).real ** 2
            for kg in self._k_grids
        )
        grad_energy = 0.5 * f.kappa * grad_sq

        dV = g.dx ** g.ndim
        result = (f_bulk + grad_energy).sum().item() * dV

        # Long-range (OK): (α/2) ∫ (φ−φ̄)(−∇²)⁻¹(φ−φ̄) dr
        if f.alpha > 0:
            psi = phi - phi.mean()
            psi_hat = torch.fft.fftn(psi)

            k2_inv = torch.zeros_like(self._k2)
            nonzero = self._k2 > 0
            k2_inv[nonzero] = 1.0 / self._k2[nonzero]

            lr_sum = (psi_hat.abs() ** 2 * k2_inv).sum()
            result += 0.5 * f.alpha * lr_sum.item() * dV / phi.numel()

        # Wall penalty: (λ/2) ∫ Ω(r) (φ − φ_wall)² dr
        if self._wall is not None:
            wall_energy = (
                0.5 * self._wall_penalty
                * (self._wall * (phi - self._wall_phi) ** 2).sum().item()
                * dV
            )
            result += wall_energy

        return result
