"""Spinodal decomposition in a channel with wetting walls (2D).

Demonstrates the surface-energy method for wall wetting:

- A channel is created with smooth walls on the top and bottom
  boundaries (along axis 1).
- The wall preferentially wets one phase via surface energy coupling
  γ > 0, producing wetting layers at both surfaces.
- The bulk undergoes standard spinodal decomposition.

The surface energy term F_s = −γ ∫ φ |∇Ω| dr adds a chemical potential
contribution μ_surface = −γ |∇Ω|, where |∇Ω| acts as a surface delta
function at the wall–fluid interface.
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    DoubleWell,
    FreeEnergyFunctional,
    GridParams,
    SpectralSolver,
    SurfaceEnergyWall,
    channel_walls,
    random_uniform,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Physics ---
    functional = FreeEnergyFunctional(local=DoubleWell(W=1.0), kappa=0.5)
    grid = GridParams(shape=(256, 64), dx=1.0, dt=0.1)

    # --- Walls on top & bottom (axis=1) ---
    mask = channel_walls(*grid.shape, wall_thickness=5, axis=1, device=device)
    # γ > 0 attracts the φ=1 (A-rich) phase to the wall
    wall = SurfaceEnergyWall(mask, gamma=0.5, dx=grid.dx)

    solver = SpectralSolver(
        functional, grid, mobility=1.0, device=device, wall=wall,
    )

    # Initial condition: random in the fluid region
    phi0 = random_uniform(*grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42, device=device)

    n_steps = 30000
    save_interval = 6000
    print(f"Running {n_steps} steps ...")

    snapshots = solver.run(phi0, n_steps=n_steps, save_interval=save_interval)

    # --- Plot ---
    n_cols = len(snapshots)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    for ax, (step, phi) in zip(axes, snapshots):
        im = ax.imshow(
            phi.numpy().T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1,
            aspect="auto",
        )
        ax.set_title(f"t = {step * grid.dt:.0f}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes, label=r"$\phi$", shrink=0.8)
    fig.suptitle(
        rf"Spinodal Decomposition in a Channel "
        rf"(surface energy, $\gamma={wall.gamma}$)",
        fontsize=13,
    )
    plt.savefig("examples/results/wetting.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/wetting.png")

    # Free energy evolution
    print("\nFree energy evolution:")
    for step, phi in snapshots:
        fe = solver.compute_total_free_energy(phi.to(device))
        print(f"  step {step:6d}: F = {fe:.4f}")


if __name__ == "__main__":
    main()
