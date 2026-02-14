"""Initial condition generators for phase-field simulations."""

from __future__ import annotations

import torch
from torch import Tensor


def random_uniform(
    nx: int,
    ny: int,
    phi_mean: float = 0.5,
    noise_amplitude: float = 0.05,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a uniform field with small random perturbations.

    Suitable for simulating spinodal decomposition from a nearly
    homogeneous state.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
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
        Initial volume fraction field, shape (nx, ny).
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = (torch.rand(nx, ny, generator=gen, device=device) - 0.5) * 2.0
    phi = phi_mean + noise_amplitude * noise
    return phi.clamp(1e-8, 1.0 - 1e-8)


def circular_droplet(
    nx: int,
    ny: int,
    phi_inside: float = 0.8,
    phi_outside: float = 0.2,
    radius: float | None = None,
    interface_width: float = 4.0,
    dx: float = 1.0,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a circular droplet initial condition.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    phi_inside : float
        Volume fraction inside the droplet.
    phi_outside : float
        Volume fraction outside the droplet.
    radius : float or None
        Droplet radius in grid units. Defaults to nx/4.
    interface_width : float
        Width of the diffuse interface.
    dx : float
        Grid spacing.
    device : torch.device or str
        Computation device.

    Returns
    -------
    Tensor
        Initial volume fraction field, shape (nx, ny).
    """
    if radius is None:
        radius = nx * dx / 4.0

    x = (torch.arange(nx, device=device) - nx / 2.0) * dx
    y = (torch.arange(ny, device=device) - ny / 2.0) * dx
    xg, yg = torch.meshgrid(x, y, indexing="ij")
    r = (xg**2 + yg**2).sqrt()

    phi = phi_outside + (phi_inside - phi_outside) * 0.5 * (
        1.0 - torch.tanh((r - radius) / interface_width)
    )
    return phi.clamp(1e-8, 1.0 - 1e-8)
