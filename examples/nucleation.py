"""Nucleation and growth in a metastable polymer blend (2D).

Demonstrates the classical nucleation scenario using a double-well
free energy:

- Background φ = 0.05 sits in the metastable region (between equilibrium
  φ = 0 and spinodal φ ≈ 0.21 for the double-well).
- Supercritical nucleus (R = 15 > R_c ≈ 6): grows into a stable new phase.
- Subcritical nucleus (R = 5 < R_c ≈ 6): shrinks back to the metastable state.

For the double-well f(φ) = W φ²(1−φ)²:
- Equilibria: φ = 0, 1
- Spinodal:   φ ≈ 0.211, 0.789   (f''(φ) = 0)
- Metastable: 0 < φ < 0.211  and  0.789 < φ < 1
"""

import matplotlib.pyplot as plt
import torch

from torch_pf import (
    DoubleWell,
    FreeEnergyFunctional,
    GridParams,
    SpectralSolver,
    droplet,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Physics ---
    dw = DoubleWell(W=1.0)
    kappa = 4.0
    functional = FreeEnergyFunctional(local=dw, kappa=kappa)
    grid = GridParams(shape=(128, 128), dx=1.0, dt=0.1)
    solver = SpectralSolver(functional, grid, mobility=1.0, device=device)

    # --- Two nuclei: super- and sub-critical ---
    phi_bg = 0.05       # metastable background (between equil. 0 and spinodal 0.21)
    phi_nuc = 0.9        # nucleus composition
    R_super = 15         # > R_c ≈ 6  → grows
    R_sub = 5            # < R_c ≈ 6  → shrinks

    phi0_super = droplet(
        *grid.shape,
        phi_inside=phi_nuc, phi_outside=phi_bg,
        radius=R_super, device=device,
    )
    phi0_sub = droplet(
        *grid.shape,
        phi_inside=phi_nuc, phi_outside=phi_bg,
        radius=R_sub, device=device,
    )

    n_steps = 20000
    save_interval = 5000
    print(f"Running {n_steps} steps ...")

    print("  Supercritical nucleus ...")
    snaps_super = solver.run(phi0_super, n_steps=n_steps, save_interval=save_interval)
    print("  Subcritical nucleus ...")
    snaps_sub = solver.run(phi0_sub, n_steps=n_steps, save_interval=save_interval)

    # --- Plot ---
    n_cols = len(snaps_super)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    for row, (snaps, label) in enumerate([
        (snaps_super, rf"Supercritical ($R={R_super} > R_c$)"),
        (snaps_sub, rf"Subcritical ($R={R_sub} < R_c$)"),
    ]):
        for col, (step, phi) in enumerate(snaps):
            ax = axes[row, col]
            im = ax.imshow(
                phi.numpy().T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1,
            )
            ax.set_title(f"t = {step * grid.dt:.0f}", fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
            ax.set_xlabel("x")

    fig.colorbar(im, ax=axes, label=r"$\phi$", shrink=0.6)
    fig.suptitle(
        rf"Nucleation ($\phi_{{bg}}={phi_bg}$, $W={dw.W}$, "
        rf"$\kappa={kappa}$, $R_c \approx 6$)",
        fontsize=14,
    )
    plt.savefig("examples/results/nucleation.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/nucleation.png")

    # Free energy evolution
    for label, snaps in [
        ("Supercritical", snaps_super),
        ("Subcritical", snaps_sub),
    ]:
        print(f"\n{label} free energy:")
        for step, phi in snaps:
            fe = solver.compute_total_free_energy(phi.to(device))
            print(f"  step {step:6d}: F = {fe:.4f}")


if __name__ == "__main__":
    main()
