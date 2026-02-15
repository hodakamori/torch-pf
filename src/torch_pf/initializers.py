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


def random_nuclei(
    *shape: int,
    n_nuclei: int = 10,
    phi_background: float = 0.05,
    phi_nucleus: float = 0.9,
    radius: float = 10.0,
    interface_width: float = 4.0,
    dx: float = 1.0,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Place multiple nuclei at random positions on a periodic domain.

    Each nucleus is a tanh-profiled droplet.  Overlapping regions are
    clamped to [0, 1].  Distances respect periodic boundary conditions.

    Parameters
    ----------
    *shape : int
        Grid dimensions, e.g. ``(128, 128)`` for 2D.
    n_nuclei : int
        Number of nuclei to place.
    phi_background : float
        Volume fraction of the metastable background.
    phi_nucleus : float
        Volume fraction at the centre of each nucleus.
    radius : float
        Radius of each nucleus (physical units).
    interface_width : float
        Width of the diffuse interface around each nucleus.
    dx : float
        Grid spacing.
    seed : int or None
        Random seed for reproducibility.
    device : torch.device or str
        Computation device.

    Returns
    -------
    Tensor
        Initial volume fraction field.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Domain lengths in physical units
    L = [n * dx for n in shape]

    # Grid coordinates: [0, L)
    coords = [torch.arange(n, device=device) * dx for n in shape]
    grids = torch.meshgrid(*coords, indexing="ij")

    # Random centre positions  (n_nuclei × ndim)
    centres = [torch.rand(n_nuclei, generator=gen) * Li for Li in L]

    phi = torch.full(shape, phi_background, device=device, dtype=torch.float32)
    for i in range(n_nuclei):
        # Minimum-image distance (periodic)
        dist_sq = torch.zeros(shape, device=device)
        for d, g in enumerate(grids):
            diff = g - centres[d][i].to(device)
            diff = diff - L[d] * torch.round(diff / L[d])
            dist_sq = dist_sq + diff**2
        r = dist_sq.sqrt()

        bump = (phi_nucleus - phi_background) * 0.5 * (
            1.0 - torch.tanh((r - radius) / interface_width)
        )
        phi = phi + bump

    return phi.clamp(1e-8, 1.0 - 1e-8)
