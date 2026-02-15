"""Compare wall-normal profiles for different penalty strengths λ.

Shows how the adsorption layer changes when wall_penalty is reduced
from the default value of 10.0.
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    DoubleWell,
    FreeEnergyFunctional,
    GridParams,
    SpectralSolver,
    channel_walls,
    random_uniform,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    functional = FreeEnergyFunctional(local=DoubleWell(W=1.0), kappa=0.5)
    grid = GridParams(shape=(128, 64), dx=1.0, dt=0.1)

    wall = channel_walls(*grid.shape, wall_thickness=5, axis=1, device=device)
    wall_phi = 1.0

    penalties = [10.0, 2.0, 0.5, 0.1]
    n_steps = 20000

    # Use the same initial condition for all runs
    phi0 = random_uniform(
        *grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42, device=device
    )

    fig, axes = plt.subplots(2, len(penalties), figsize=(4 * len(penalties), 7))

    for col, lam in enumerate(penalties):
        print(f"Running λ = {lam} ...")
        solver = SpectralSolver(
            functional, grid, mobility=1.0, device=device,
            wall=wall, wall_phi=wall_phi, wall_penalty=lam,
        )
        snapshots = solver.run(phi0.clone(), n_steps=n_steps, save_interval=n_steps)
        phi_final = snapshots[-1][1]

        # 2D snapshot
        ax_img = axes[0, col]
        ax_img.imshow(
            phi_final.numpy().T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1,
            aspect="auto",
        )
        ax_img.set_title(rf"$\lambda = {lam}$", fontsize=12)
        ax_img.set_xlabel("x")
        if col == 0:
            ax_img.set_ylabel("y")

        # Wall-normal profile (x-averaged)
        profile = phi_final.mean(dim=0).numpy()  # average over x
        y = torch.arange(grid.shape[1]).numpy()
        wall_profile = wall[0, :].cpu().numpy()

        ax_prof = axes[1, col]
        ax_prof.plot(y, profile, "b-", label=r"$\langle\phi\rangle_x$")
        ax_prof.fill_between(y, 0, wall_profile, alpha=0.2, color="gray", label="wall mask")
        ax_prof.set_xlabel("y")
        ax_prof.set_ylim(-0.1, 1.1)
        ax_prof.set_title(rf"$\lambda = {lam}$", fontsize=12)
        if col == 0:
            ax_prof.set_ylabel(r"$\phi$")
        ax_prof.legend(fontsize=8)

    fig.suptitle(
        rf"Effect of penalty strength $\lambda$ ($\phi_{{wall}}={wall_phi}$, "
        f"t={n_steps * grid.dt:.0f})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("examples/results/wall_penalty_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/wall_penalty_comparison.png")


if __name__ == "__main__":
    main()
