"""Cahn-Hilliard solver using the spectral (FFT) method.

Solves the Cahn-Hilliard equation:

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

from dataclasses import dataclass

import torch
from torch import Tensor

from .free_energy import FreeEnergy


@dataclass
class SimulationParams:
    """Parameters for the Cahn-Hilliard simulation.

    Parameters
    ----------
    nx : int
        Number of grid points in x.
    ny : int
        Number of grid points in y.
    dx : float
        Grid spacing.
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

    nx: int = 128
    ny: int = 128
    dx: float = 1.0
    dt: float = 0.5
    mobility: float = 1.0
    kappa: float = 0.5
    stabilization: float | None = None


class CahnHilliardSolver:
    """Spectral solver for the Cahn-Hilliard equation.

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

        # Precompute wavenumbers and denominator
        self._k2, self._k4 = self._build_wavenumbers()
        self._denom = (
            1.0
            + params.dt * params.mobility * self._k2 * (self._C + params.kappa * self._k2)
        )

        # Precompute spectral gradient operators for free energy computation
        kx = torch.fft.fftfreq(params.nx, d=params.dx / (2.0 * torch.pi)).to(self.device)
        ky = torch.fft.fftfreq(params.ny, d=params.dx / (2.0 * torch.pi)).to(self.device)
        self._kx_grid, self._ky_grid = torch.meshgrid(kx, ky, indexing="ij")

    def _estimate_stabilization(self) -> float:
        """Estimate stabilization constant from max |f''| in [0.01, 0.99]."""
        phi_test = torch.linspace(0.01, 0.99, 2000)
        d2f = self.free_energy.second_derivative(phi_test)
        return max(float(d2f.abs().max()) * 1.1, 1.0)

    def _build_wavenumbers(self) -> tuple[Tensor, Tensor]:
        """Build squared wavenumber arrays k² and k⁴."""
        p = self.params
        kx = torch.fft.fftfreq(p.nx, d=p.dx / (2.0 * torch.pi)).to(self.device)
        ky = torch.fft.fftfreq(p.ny, d=p.dx / (2.0 * torch.pi)).to(self.device)
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
        k2 = kx_grid**2 + ky_grid**2
        k4 = k2**2
        return k2, k4

    def step(self, phi: Tensor) -> Tensor:
        """Advance the field by one time step.

        Parameters
        ----------
        phi : Tensor
            Current volume fraction field, shape (nx, ny).

        Returns
        -------
        Tensor
            Updated volume fraction field.
        """
        p = self.params
        # Explicit nonlinear part: g = f'(φ) - C*φ
        g = self.free_energy.chemical_potential(phi) - self._C * phi
        g_hat = torch.fft.fft2(g)
        phi_hat = torch.fft.fft2(phi)

        phi_hat_new = (phi_hat - p.dt * p.mobility * self._k2 * g_hat) / self._denom
        return torch.fft.ifft2(phi_hat_new).real

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
            Initial volume fraction field, shape (nx, ny).
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
        phi_hat = torch.fft.fft2(phi)
        dphi_dx = torch.fft.ifft2(1j * self._kx_grid * phi_hat).real
        dphi_dy = torch.fft.ifft2(1j * self._ky_grid * phi_hat).real
        grad_energy = 0.5 * p.kappa * (dphi_dx**2 + dphi_dy**2)

        total = (f_bulk + grad_energy).sum() * p.dx**2
        return total.item()
