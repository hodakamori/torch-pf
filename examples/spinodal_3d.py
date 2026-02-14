"""Spinodal decomposition of a symmetric polymer blend (3D).

This example simulates 3D spinodal decomposition and saves cross-section
snapshots at z = Nz/2.
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    CahnHilliardSolver,
    FloryHuggins,
    SimulationParams,
    random_uniform,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    free_energy = FloryHuggins(chi=0.1, n_a=100, n_b=100)
    print(f"chi = {free_energy.chi}, chi_c = {free_energy.chi_critical:.4f}")

    N = 64  # 64^3 grid
    params = SimulationParams(
        shape=(N, N, N),
        dx=1.0,
        dt=0.5,
        mobility=1.0,
        kappa=0.5,
    )
    print(f"Grid: {params.shape} ({params.ndim}D)")

    solver = CahnHilliardSolver(params, free_energy, device=device)

    phi0 = random_uniform(
        *params.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42, device=device
    )

    n_steps = 5000
    save_interval = 1000
    print(f"Running {n_steps} steps...")

    snapshots = solver.run(phi0, n_steps=n_steps, save_interval=save_interval)

    print("\nFree energy evolution:")
    for step, phi in snapshots:
        phi_dev = phi.to(device)
        fe = solver.compute_total_free_energy(phi_dev)
        print(f"  step {step:6d}: F = {fe:.2f}")

    # Plot cross-sections at z = N/2
    z_mid = N // 2
    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]

    for ax, (step, phi) in zip(axes, snapshots):
        cross = phi[:, :, z_mid].numpy().T
        im = ax.imshow(cross, origin="lower", cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_title(f"t = {step * params.dt:.0f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes, label=r"$\phi$", shrink=0.8)
    fig.suptitle(f"3D Spinodal Decomposition (z = {z_mid} cross-section)", fontsize=14)
    plt.tight_layout()
    plt.savefig("spinodal_3d.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved spinodal_3d.png")


if __name__ == "__main__":
    main()
