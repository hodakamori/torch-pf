"""Block copolymer microphase separation (3D, Ohta-Kawasaki model).

Renders phi=0.5 isosurfaces of the minority phase, showing sphere-like
microstructures that emerge from the long-range interaction.

Requires ``scikit-image`` (install with ``uv sync --extra vis3d``).
"""

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from torch_pf import (
    DoubleWell,
    OhtaKawasakiSolver,
    SimulationParams,
    random_uniform,
)


def plot_isosurface(ax, phi_np, level=0.5, color="#2060b0", alpha=0.6):
    """Render a phi=level isosurface on a 3D axes."""
    verts, faces, _, _ = marching_cubes(phi_np, level=level)
    mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor="k", linewidth=0.1)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    n = phi_np.shape[0]
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    free_energy = DoubleWell(W=1.0)
    N = 64
    params = SimulationParams(shape=(N, N, N), dx=1.0, dt=0.5, mobility=1.0, kappa=0.5)
    alpha = 0.05

    solver = OhtaKawasakiSolver(params, free_energy, alpha=alpha, device=device)

    phi0 = random_uniform(*params.shape, phi_mean=0.35, noise_amplitude=0.05, seed=42, device=device)

    n_steps = 5000
    save_interval = 1000
    print(f"Running {n_steps} steps (64^3 grid) ...")

    snapshots = solver.run(phi0, n_steps=n_steps, save_interval=save_interval)

    print("\nFree energy evolution:")
    for step, phi in snapshots:
        fe = solver.compute_total_free_energy(phi.to(device))
        print(f"  step {step:6d}: F = {fe:.2f}")

    # --- 3D isosurface rendering ---
    plot_snapshots = [(s, p) for s, p in snapshots if s > 0]
    n_plots = len(plot_snapshots)
    fig = plt.figure(figsize=(5 * n_plots, 5))

    for i, (step, phi) in enumerate(plot_snapshots):
        ax = fig.add_subplot(1, n_plots, i + 1, projection="3d")
        plot_isosurface(ax, phi.numpy(), level=0.5, color="#2060b0", alpha=0.6)
        ax.set_title(f"t = {step * params.dt:.0f}", fontsize=12)
        ax.view_init(elev=25, azim=-60)

    fig.suptitle(
        rf"3D Block Copolymer — $\phi=0.5$ isosurface "
        rf"($\bar\phi=0.35$, $\alpha={alpha}$)",
        fontsize=14,
    )
    plt.savefig("examples/results/block_copolymer_3d.png", dpi=150, bbox_inches="tight")
    print("\nSaved examples/results/block_copolymer_3d.png")


if __name__ == "__main__":
    main()
