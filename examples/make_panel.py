"""Generate a panel of snapshots showing spinodal decomposition evolution."""

import matplotlib.pyplot as plt

from torch_pf import CahnHilliardSolver, FloryHuggins, SimulationParams, random_uniform


def main() -> None:
    free_energy = FloryHuggins(chi=0.1, n_a=100, n_b=100)
    params = SimulationParams(shape=(128, 128), dx=1.0, dt=0.5, mobility=1.0, kappa=0.5)
    solver = CahnHilliardSolver(params, free_energy)
    print(f"Stabilization C = {solver._C:.4f}")

    phi0 = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.05, seed=42)

    save_steps = [0, 500, 1500, 3000, 8000, 30000]
    phi = phi0.clone()
    snapshots = [(0, phi.clone())]

    current = 0
    for target in save_steps[1:]:
        for _ in range(target - current):
            phi = solver.step(phi)
        current = target
        snapshots.append((target, phi.clone()))
        print(f"step {target:6d}: min={phi.min():.4f} max={phi.max():.4f} std={phi.std():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes_flat = axes.flatten()

    for ax, (step, phi_snap) in zip(axes_flat, snapshots):
        im = ax.imshow(
            phi_snap.numpy().T,
            origin="lower",
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
        )
        t = step * params.dt
        ax.set_title(f"t = {t:.0f}", fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes_flat.tolist(), label=r"$\phi$ (volume fraction of A)", shrink=0.6, pad=0.02)
    fig.suptitle(
        f"Spinodal Decomposition of Polymer Blend\n"
        f"($N_A=N_B={int(free_energy.n_a)},\\ \\chi={free_energy.chi},\\ \\chi_c={free_energy.chi_critical:.2f}$)",
        fontsize=14,
    )
    out_path = "examples/results/spinodal_panel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
