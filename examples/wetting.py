"""Effect of surface energy γ on wetting behaviour.

Demonstrates how the surface energy coupling γ controls the
wall–fluid interaction in a channel geometry:

  γ > 0  →  wall attracts the φ = 1 phase  (hydrophilic for A)
  γ = 0  →  neutral wetting  (90° contact angle)
  γ < 0  →  wall attracts the φ = 0 phase  (hydrophilic for B)

For each γ value, the simulation runs spinodal decomposition inside
a channel and plots:

  - Top row: 2D snapshot of the final field
  - Bottom row: wall-normal profile ⟨φ⟩_x averaged over x
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
    grid = GridParams(shape=(128, 64), dx=1.0, dt=0.1)

    # --- Channel walls (top & bottom) ---
    mask = channel_walls(*grid.shape, wall_thickness=5, axis=1, device=device)

    # --- Sweep over γ: A-wetting → neutral → B-wetting ---
    gammas = [0.3, 0.1, 0.0, -0.1, -0.3]

    n_steps = 20000

    # Same initial condition for all runs
    phi0 = random_uniform(
        *grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42, device=device
    )

    n_cols = len(gammas)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 7))

    wall_1d = mask[0, :].cpu().numpy()
    y = torch.arange(grid.shape[1]).numpy()

    for col, gamma in enumerate(gammas):
        label = f"γ = {gamma:+.1f}"
        print(f"Running {label} ...")

        wall = SurfaceEnergyWall(mask, gamma=gamma, dx=grid.dx)
        solver = SpectralSolver(
            functional, grid, mobility=1.0, device=device, wall=wall,
        )
        snapshots = solver.run(phi0.clone(), n_steps=n_steps, save_interval=n_steps)
        phi_final = snapshots[-1][1]

        # --- Top row: 2D snapshot ---
        ax_img = axes[0, col]
        ax_img.imshow(
            phi_final.numpy().T, origin="lower", cmap="RdBu_r",
            vmin=0, vmax=1, aspect="auto",
        )
        ax_img.set_title(label, fontsize=12)
        ax_img.set_xlabel("x")
        if col == 0:
            ax_img.set_ylabel("y")

        # --- Bottom row: wall-normal profile ---
        profile = phi_final.mean(dim=0).numpy()

        ax_prof = axes[1, col]
        ax_prof.plot(y, profile, "b-", lw=1.5, label=r"$\langle\phi\rangle_x$")
        ax_prof.fill_between(
            y, 0, wall_1d, alpha=0.15, color="gray", label="wall",
        )
        ax_prof.axhline(0.5, color="k", ls=":", lw=0.5)
        ax_prof.set_xlabel("y")
        ax_prof.set_ylim(-0.05, 1.05)
        ax_prof.set_title(label, fontsize=12)
        if col == 0:
            ax_prof.set_ylabel(r"$\phi$")
        ax_prof.legend(fontsize=8, loc="center right")

        fe = solver.compute_total_free_energy(phi_final.to(device))
        print(f"  {label}  F = {fe:.2f}")

    fig.suptitle(
        f"Surface energy wetting: effect of γ  (t = {n_steps * grid.dt:.0f})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("examples/results/wetting.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/wetting.png")


if __name__ == "__main__":
    main()
