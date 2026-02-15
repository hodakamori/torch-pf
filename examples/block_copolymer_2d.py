"""Block copolymer microphase separation (2D, Ohta-Kawasaki model).

Demonstrates lamellae (φ̄ = 0.5) and cylinder/dot morphology (φ̄ = 0.35)
driven by the competition between short-range (double-well) and
long-range (Ohta-Kawasaki) interactions.
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    DoubleWell,
    OhtaKawasakiSolver,
    SimulationParams,
    random_uniform,
)


def run_ok(phi_mean: float, params: SimulationParams, free_energy, alpha, device, n_steps, save_interval):
    solver = OhtaKawasakiSolver(params, free_energy, alpha=alpha, device=device)
    phi0 = random_uniform(*params.shape, phi_mean=phi_mean, noise_amplitude=0.05, seed=42, device=device)
    return solver, solver.run(phi0, n_steps=n_steps, save_interval=save_interval)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    free_energy = DoubleWell(W=1.0)
    N = 128
    params = SimulationParams(shape=(N, N), dx=1.0, dt=0.5, mobility=1.0, kappa=0.5)

    n_steps = 30000
    save_interval = 7500

    # Larger α narrows the band of unstable modes, enforcing periodic order.
    #   α_max = |f''(φ̄)|² / (4κ)   [instability threshold]
    #   φ̄=0.5  → |f''|=1.0,  α_max=0.50  → use α=0.30
    #   φ̄=0.35 → |f''|=0.73, α_max=0.27  → use α=0.15
    alpha_lam = 0.30
    alpha_cyl = 0.15

    # --- Lamellae (symmetric, φ̄ = 0.5) ---
    print(f"Running lamellae (φ̄ = 0.5, α = {alpha_lam}) ...")
    solver_lam, snaps_lam = run_ok(0.5, params, free_energy, alpha_lam, device, n_steps, save_interval)

    # --- Cylinders / dots (asymmetric, φ̄ = 0.35) ---
    print(f"Running cylinders (φ̄ = 0.35, α = {alpha_cyl}) ...")
    solver_cyl, snaps_cyl = run_ok(0.35, params, free_energy, alpha_cyl, device, n_steps, save_interval)

    # --- Plot ---
    n_cols = len(snaps_lam)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    for row, (solver, snaps, label) in enumerate([
        (solver_lam, snaps_lam, rf"Lamellae ($\bar\phi=0.5,\ \alpha={alpha_lam}$)"),
        (solver_cyl, snaps_cyl, rf"Cylinders ($\bar\phi=0.35,\ \alpha={alpha_cyl}$)"),
    ]):
        for col, (step, phi) in enumerate(snaps):
            ax = axes[row, col]
            im = ax.imshow(phi.numpy().T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_title(f"t = {step * params.dt:.0f}", fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
            ax.set_xlabel("x")

    fig.colorbar(im, ax=axes, label=r"$\phi$", shrink=0.6)
    fig.suptitle(
        rf"Ohta-Kawasaki Model ($\kappa={params.kappa}$, $W={free_energy.W}$)",
        fontsize=14,
    )
    plt.savefig("examples/results/block_copolymer_2d.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/block_copolymer_2d.png")

    # Print free energy evolution
    for label, solver, snaps in [
        ("Lamellae", solver_lam, snaps_lam),
        ("Cylinders", solver_cyl, snaps_cyl),
    ]:
        print(f"\n{label} free energy:")
        for step, phi in snaps:
            fe = solver.compute_total_free_energy(phi.to(device))
            print(f"  step {step:6d}: F = {fe:.2f}")


if __name__ == "__main__":
    main()
