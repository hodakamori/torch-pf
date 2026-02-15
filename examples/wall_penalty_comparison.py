"""Compare SurfaceEnergyWall vs VolumePenaltyWall methods.

Runs the same spinodal decomposition in a channel using both wall
methods and compares the resulting morphologies and wall-normal
profiles side-by-side.

Left columns:  SurfaceEnergyWall with different γ values
Right column:  VolumePenaltyWall (legacy method) for reference
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    DoubleWell,
    FreeEnergyFunctional,
    GridParams,
    SpectralSolver,
    SurfaceEnergyWall,
    VolumePenaltyWall,
    channel_walls,
    random_uniform,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    functional = FreeEnergyFunctional(local=DoubleWell(W=1.0), kappa=0.5)
    grid = GridParams(shape=(128, 64), dx=1.0, dt=0.1)

    mask = channel_walls(*grid.shape, wall_thickness=5, axis=1, device=device)

    walls = [
        ("Surface γ=0.5", SurfaceEnergyWall(mask, gamma=0.5, dx=grid.dx)),
        ("Surface γ=1.0", SurfaceEnergyWall(mask, gamma=1.0, dx=grid.dx)),
        ("Penalty λ=2", VolumePenaltyWall(mask, phi_wall=1.0, penalty=2.0)),
        ("Penalty λ=10", VolumePenaltyWall(mask, phi_wall=1.0, penalty=10.0)),
    ]

    n_steps = 20000

    phi0 = random_uniform(
        *grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42, device=device
    )

    n_cols = len(walls)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7))

    wall_1d = mask[0, :].cpu().numpy()
    y = torch.arange(grid.shape[1]).numpy()

    for col, (label, wall) in enumerate(walls):
        print(f"Running {label} ...")
        solver = SpectralSolver(
            functional, grid, mobility=1.0, device=device, wall=wall,
        )
        snapshots = solver.run(phi0.clone(), n_steps=n_steps, save_interval=n_steps)
        phi_final = snapshots[-1][1]

        # 2D snapshot
        ax_img = axes[0, col]
        ax_img.imshow(
            phi_final.numpy().T, origin="lower", cmap="RdBu_r",
            vmin=0, vmax=1, aspect="auto",
        )
        ax_img.set_title(label, fontsize=12)
        ax_img.set_xlabel("x")
        if col == 0:
            ax_img.set_ylabel("y")

        # Wall-normal profile (x-averaged)
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
        f"Surface energy vs volume penalty  (t = {n_steps * grid.dt:.0f})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        "examples/results/wall_penalty_comparison.png", dpi=150, bbox_inches="tight",
    )
    print("\nSaved examples/results/wall_penalty_comparison.png")


if __name__ == "__main__":
    main()
