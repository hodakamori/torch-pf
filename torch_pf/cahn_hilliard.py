"""Cahn-Hilliard solver using the spectral (FFT) method.

Solves the Cahn-Hilliard equation in 2D or 3D:

    ∂φ/∂t = M ∇²μ
    μ     = f'(φ) - κ ∇²φ

A stabilized semi-implicit spectral scheme is used.  The chemical
potential is split as μ = [f'(φ) − Cφ] + [Cφ − κ∇²φ], where the
first bracket is treated explicitly and the second implicitly:

    φ̂^{n+1} = (φ̂^n − Δt M k² ĝ^n) / (1 + Δt M k² (C + κ k²))
    g^n      = f'(φ^n) − C φ^n

The stabilization constant C is automatically chosen to ensure stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .free_energy import FreeEnergy


@dataclass
class SimulationParams:
    """Parameters for the Cahn-Hilliard simulation.

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions, e.g. (128, 128) for 2D or (64, 64, 64) for 3D.
    dx : float
        Grid spacing (uniform in all directions).
    dt : float
        Time step size.
    mobility : float
        Mobility coefficient M.
    kappa : float
        Gradient energy coefficient κ.
    stabilization : float | None
        Stabilization constant C for the semi-implicit scheme.
        If None, it is estimated automatically from the free energy.
    """

    shape: tuple[int, ...] = (128, 128)
    dx: float = 1.0
    dt: float = 0.5
    mobility: float = 1.0
    kappa: float = 0.5
    stabilization: float | None = None

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (2 or 3)."""
        return len(self.shape)


class CahnHilliardSolver:
    """Spectral solver for the Cahn-Hilliard equation (2D / 3D).

    Works with any ``FreeEnergy`` model (FloryHuggins, DoubleWell, etc.).

    Parameters
    ----------
    params : SimulationParams
        Simulation parameters.
    free_energy : FreeEnergy
        Free energy model.
    device : torch.device | str
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        params: SimulationParams,
        free_energy: FreeEnergy,
        device: torch.device | str = "cpu",
    ) -> None:
        self.params = params
        self.free_energy = free_energy
        self.device = torch.device(device)

        # Determine stabilization constant
        if params.stabilization is not None:
            self._C = params.stabilization
        else:
            self._C = self._estimate_stabilization()

        # Precompute wavenumber grids
        self._k_grids = self._build_k_grids()  # list of 1-D k arrays broadcast to N-D
        self._k2 = sum(kg**2 for kg in self._k_grids)  # |k|²
        self._denom = (
            1.0
            + params.dt * params.mobility * self._k2 * (self._C + params.kappa * self._k2)
        )

    def _estimate_stabilization(self) -> float:
        """Estimate stabilization constant from max |f''| in [0.01, 0.99]."""
        phi_test = torch.linspace(0.01, 0.99, 2000)
        d2f = self.free_energy.second_derivative(phi_test)
        return max(float(d2f.abs().max()) * 1.1, 1.0)

    def _build_k_grids(self) -> list[Tensor]:
        """Build wavenumber component grids for each spatial dimension."""
        p = self.params
        k_1d = [
            torch.fft.fftfreq(n, d=p.dx / (2.0 * torch.pi)).to(self.device)
            for n in p.shape
        ]
        grids = torch.meshgrid(*k_1d, indexing="ij")
        return list(grids)

    def step(self, phi: Tensor) -> Tensor:
        """Advance the field by one time step.

        Parameters
        ----------
        phi : Tensor
            Current volume fraction field, shape matching ``params.shape``.

        Returns
        -------
        Tensor
            Updated volume fraction field.
        """
        p = self.params
        g = self.free_energy.chemical_potential(phi) - self._C * phi
        g_hat = torch.fft.fftn(g)
        phi_hat = torch.fft.fftn(phi)

        phi_hat_new = (phi_hat - p.dt * p.mobility * self._k2 * g_hat) / self._denom
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
        """Compute the total free energy F = integral[f(phi) + (kappa/2)|grad(phi)|^2] dr.

        Parameters
        ----------
        phi : Tensor
            Volume fraction field.

        Returns
        -------
        float
            Total free energy.
        """
        p = self.params
        f_bulk = self.free_energy.free_energy_density(phi)

        # |nabla phi|^2 via spectral derivatives
        phi_hat = torch.fft.fftn(phi)
        grad_sq = sum(
            torch.fft.ifftn(1j * kg * phi_hat).real ** 2
            for kg in self._k_grids
        )
        grad_energy = 0.5 * p.kappa * grad_sq

        dV = p.dx ** p.ndim
        total = (f_bulk + grad_energy).sum() * dV
        return total.item()
