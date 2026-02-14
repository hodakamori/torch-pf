"""Initial condition generators for phase-field simulations (2D / 3D)."""

from __future__ import annotations

import torch
from torch import Tensor


def random_uniform(
    *shape: int,
    phi_mean: float = 0.5,
    noise_amplitude: float = 0.05,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a uniform field with small random perturbations.

    Suitable for simulating spinodal decomposition from a nearly
    homogeneous state.

    Parameters
    ----------
    *shape : int
        Grid dimensions, e.g. ``(128, 128)`` for 2D or ``(64, 64, 64)`` for 3D.
    phi_mean : float
        Mean volume fraction.
    noise_amplitude : float
        Amplitude of random perturbations.
    seed : int or None
        Random seed for reproducibility.
    device : torch.device or str
        Computation device.

    Returns
    -------
    Tensor
        Initial volume fraction field.
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = (torch.rand(*shape, generator=gen, device=device) - 0.5) * 2.0
    phi = phi_mean + noise_amplitude * noise
    return phi.clamp(1e-8, 1.0 - 1e-8)


def droplet(
    *shape: int,
    phi_inside: float = 0.8,
    phi_outside: float = 0.2,
    radius: float | None = None,
    interface_width: float = 4.0,
    dx: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a spherical (3D) or circular (2D) droplet initial condition.

    Parameters
    ----------
    *shape : int
        Grid dimensions.
    phi_inside : float
        Volume fraction inside the droplet.
    phi_outside : float
        Volume fraction outside the droplet.
    radius : float or None
        Droplet radius in physical units. Defaults to L/4 where L = shape[0]*dx.
    interface_width : float
        Width of the diffuse interface.
    dx : float
        Grid spacing.
    device : torch.device or str
        Computation device.

    Returns
    -------
    Tensor
        Initial volume fraction field.
    """
    if radius is None:
        radius = shape[0] * dx / 4.0

    coords = [
        (torch.arange(n, device=device) - n / 2.0) * dx
        for n in shape
    ]
    grids = torch.meshgrid(*coords, indexing="ij")
    r = sum(g**2 for g in grids).sqrt()

    phi = phi_outside + (phi_inside - phi_outside) * 0.5 * (
        1.0 - torch.tanh((r - radius) / interface_width)
    )
    return phi.clamp(1e-8, 1.0 - 1e-8)
