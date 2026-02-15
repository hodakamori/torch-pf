"""Spinodal decomposition of a symmetric polymer blend (2D).

This example simulates the spinodal decomposition of a symmetric
polymer blend (N_A = N_B = 100) with χ well above the critical value.
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    FloryHuggins,
    FreeEnergyFunctional,
    GridParams,
    SpectralSolver,
    random_uniform,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    fe = FloryHuggins(chi=0.1, n_a=100, n_b=100)
    print(f"chi = {fe.chi}, chi_c = {fe.chi_critical:.4f}")

    functional = FreeEnergyFunctional(local=fe, kappa=0.5)
    grid = GridParams(shape=(128, 128), dx=1.0, dt=0.5)
    solver = SpectralSolver(functional, grid, mobility=1.0, device=device)

    phi0 = random_uniform(
        *grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42, device=device
    )

    n_steps = 20000
    save_interval = 4000
    print(f"Running {n_steps} steps...")

    snapshots = solver.run(phi0, n_steps=n_steps, save_interval=save_interval)

    print("\nFree energy evolution:")
    for step, phi in snapshots:
        phi_dev = phi.to(device)
        fe_val = solver.compute_total_free_energy(phi_dev)
        print(f"  step {step:6d}: F = {fe_val:.4f}")

    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]

    for ax, (step, phi) in zip(axes, snapshots):
        im = ax.imshow(phi.numpy().T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_title(f"t = {step * grid.dt:.0f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes, label=r"$\phi$", shrink=0.8)
    fig.suptitle("Spinodal Decomposition (2D)", fontsize=14)
    plt.savefig("examples/results/spinodal_2d.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/spinodal_2d.png")


if __name__ == "__main__":
    main()
