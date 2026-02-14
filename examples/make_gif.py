"""Generate a GIF animation of spinodal decomposition."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch_pf import CahnHilliardSolver, FloryHuggins, SimulationParams, random_uniform


def main() -> None:
    free_energy = FloryHuggins(chi=0.1, n_a=100, n_b=100)
    params = SimulationParams(shape=(128, 128), dx=1.0, dt=0.5, mobility=1.0, kappa=0.5)
    solver = CahnHilliardSolver(params, free_energy)

    phi0 = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.05, seed=42)

    n_steps = 20000
    save_interval = 200
    print(f"Running {n_steps} steps...")
    snapshots = solver.run(phi0, n_steps=n_steps, save_interval=save_interval)
    print(f"Got {len(snapshots)} frames")

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        snapshots[0][1].numpy().T,
        origin="lower",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax, label=r"$\phi$", shrink=0.8)
    title = ax.set_title("t = 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle(
        f"Spinodal Decomposition ($N={int(free_energy.n_a)},\\ \\chi={free_energy.chi}$)",
        fontsize=13,
    )

    def update(frame_idx: int):
        step, phi = snapshots[frame_idx]
        im.set_data(phi.numpy().T)
        title.set_text(f"t = {step * params.dt:.0f}")
        return [im, title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=80, blit=True
    )
    out_path = "examples/results/spinodal_decomposition.gif"
    ani.save(out_path, writer="pillow", fps=12)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
