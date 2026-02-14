"""Spinodal decomposition of a symmetric polymer blend.

This example simulates the spinodal decomposition of a symmetric
polymer blend (N_A = N_B = 100) with χ above the critical value.
Starting from a nearly homogeneous state with small random perturbations,
the system phase-separates into A-rich and B-rich domains.
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

    # Symmetric blend: N_A = N_B = 100
    # Critical χ = 2/N = 0.02 for symmetric blend
    # Use χ = 0.03 (above critical) to trigger spinodal decomposition
    free_energy = FloryHuggins(chi=0.03, n_a=100, n_b=100)
    print(f"χ = {free_energy.chi}, χ_c = {free_energy.chi_critical:.4f}")

    params = SimulationParams(
        nx=128,
        ny=128,
        dx=1.0,
        dt=0.5,
        mobility=1.0,
        kappa=0.5,
    )

    solver = CahnHilliardSolver(params, free_energy, device=device)

    # Start from nearly homogeneous state (φ = 0.5 + noise)
    phi0 = random_uniform(
        params.nx, params.ny, phi_mean=0.5, noise_amplitude=0.01, seed=42, device=device
    )

    n_steps = 10000
    save_interval = 2000
    print(f"Running {n_steps} steps...")

    snapshots = solver.run(phi0, n_steps=n_steps, save_interval=save_interval)

    # Compute free energy at each snapshot
    print("\nFree energy evolution:")
    for step, phi in snapshots:
        phi_dev = phi.to(device)
        fe = solver.compute_total_free_energy(phi_dev)
        print(f"  step {step:6d}: F = {fe:.4f}, φ_min = {phi.min():.4f}, φ_max = {phi.max():.4f}")

    # Plot snapshots
    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]

    for ax, (step, phi) in zip(axes, snapshots):
        im = ax.imshow(
            phi.numpy().T,
            origin="lower",
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"t = {step * params.dt:.0f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes, label="φ (volume fraction of A)", shrink=0.8)
    fig.suptitle("Spinodal Decomposition of Polymer Blend", fontsize=14)
    plt.tight_layout()
    plt.savefig("spinodal_decomposition.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to spinodal_decomposition.png")


if __name__ == "__main__":
    main()
