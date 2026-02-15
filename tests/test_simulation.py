"""Tests for the phase-field simulation."""

import torch
import pytest

from torch_pf import (
    DoubleWell,
    FloryHuggins,
    FreeEnergyFunctional,
    GridParams,
    SpectralSolver,
    SurfaceEnergyWall,
    channel_walls,
    random_uniform,
    droplet,
)


class TestFloryHuggins:
    def test_critical_chi_symmetric(self):
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        assert abs(fh.chi_critical - 0.02) < 1e-10

    def test_critical_phi_symmetric(self):
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        assert abs(fh.phi_critical - 0.5) < 1e-10

    def test_free_energy_density_shape(self):
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        phi = torch.full((64, 64), 0.5)
        f = fh.free_energy_density(phi)
        assert f.shape == (64, 64)

    def test_chemical_potential_shape(self):
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        phi = torch.full((64, 64), 0.5)
        mu = fh.chemical_potential(phi)
        assert mu.shape == (64, 64)

    def test_chemical_potential_at_critical_point(self):
        fh = FloryHuggins(chi=0.02, n_a=100, n_b=100)
        phi = torch.tensor([0.5])
        mu = fh.chemical_potential(phi)
        assert abs(mu.item()) < 1e-6

    def test_second_derivative_unstable(self):
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        phi = torch.tensor([0.5])
        d2f = fh.second_derivative(phi)
        assert d2f.item() < 0

    def test_second_derivative_stable(self):
        fh = FloryHuggins(chi=0.01, n_a=100, n_b=100)
        phi = torch.tensor([0.5])
        d2f = fh.second_derivative(phi)
        assert d2f.item() > 0


class TestInitializers:
    # --- 2D ---
    def test_random_uniform_2d_shape(self):
        phi = random_uniform(64, 64)
        assert phi.shape == (64, 64)

    def test_random_uniform_2d_mean(self):
        phi = random_uniform(256, 256, phi_mean=0.4, noise_amplitude=0.01, seed=0)
        assert abs(phi.mean().item() - 0.4) < 0.01

    def test_random_uniform_2d_bounds(self):
        phi = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.1)
        assert phi.min() > 0
        assert phi.max() < 1

    def test_random_uniform_reproducible(self):
        a = random_uniform(32, 32, seed=123)
        b = random_uniform(32, 32, seed=123)
        assert torch.allclose(a, b)

    def test_droplet_2d_shape(self):
        phi = droplet(64, 64)
        assert phi.shape == (64, 64)

    def test_droplet_2d_center(self):
        phi = droplet(64, 64, phi_inside=0.8, phi_outside=0.2)
        assert phi[32, 32].item() > 0.7

    def test_droplet_2d_corner(self):
        phi = droplet(64, 64, phi_inside=0.8, phi_outside=0.2)
        assert phi[0, 0].item() < 0.3

    # --- 3D ---
    def test_random_uniform_3d_shape(self):
        phi = random_uniform(16, 16, 16)
        assert phi.shape == (16, 16, 16)

    def test_random_uniform_3d_mean(self):
        phi = random_uniform(32, 32, 32, phi_mean=0.5, noise_amplitude=0.01, seed=0)
        assert abs(phi.mean().item() - 0.5) < 0.01

    def test_droplet_3d_shape(self):
        phi = droplet(32, 32, 32)
        assert phi.shape == (32, 32, 32)

    def test_droplet_3d_center(self):
        phi = droplet(32, 32, 32, phi_inside=0.8, phi_outside=0.2)
        assert phi[16, 16, 16].item() > 0.7

    def test_droplet_3d_corner(self):
        phi = droplet(32, 32, 32, phi_inside=0.8, phi_outside=0.2)
        assert phi[0, 0, 0].item() < 0.3


class TestSpectralSolver2D:
    def setup_method(self):
        self.fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        self.functional = FreeEnergyFunctional(local=self.fh, kappa=0.5)
        self.grid = GridParams(shape=(32, 32), dx=1.0, dt=0.1)
        self.solver = SpectralSolver(self.functional, self.grid, mobility=1.0)

    def test_step_preserves_shape(self):
        phi = random_uniform(32, 32, seed=42)
        phi_new = self.solver.step(phi)
        assert phi_new.shape == (32, 32)

    def test_mass_conservation(self):
        phi = random_uniform(32, 32, phi_mean=0.5, seed=42)
        total_mass_before = phi.sum().item()
        phi_new = self.solver.step(phi)
        total_mass_after = phi_new.sum().item()
        assert abs(total_mass_before - total_mass_after) < 1e-3

    def test_mass_conservation_over_many_steps(self):
        phi = random_uniform(32, 32, phi_mean=0.5, seed=42)
        total_mass_0 = phi.sum().item()
        for _ in range(100):
            phi = self.solver.step(phi)
        assert abs(phi.sum().item() - total_mass_0) < 1e-4

    def test_run_returns_snapshots(self):
        phi0 = random_uniform(32, 32, seed=42)
        snapshots = self.solver.run(phi0, n_steps=100, save_interval=50)
        assert len(snapshots) == 3
        assert snapshots[0][0] == 0
        assert snapshots[1][0] == 50
        assert snapshots[2][0] == 100

    def test_free_energy_decreases(self):
        phi = random_uniform(32, 32, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_initial = self.solver.compute_total_free_energy(phi)
        for _ in range(500):
            phi = self.solver.step(phi)
        fe_final = self.solver.compute_total_free_energy(phi)
        assert fe_final < fe_initial

    def test_homogeneous_state_stable_below_chi_c(self):
        fh_stable = FloryHuggins(chi=0.01, n_a=100, n_b=100)
        func = FreeEnergyFunctional(local=fh_stable, kappa=0.5)
        solver = SpectralSolver(func, self.grid, mobility=1.0)
        phi = random_uniform(32, 32, phi_mean=0.5, noise_amplitude=0.001, seed=42)
        std_initial = phi.std().item()
        for _ in range(200):
            phi = solver.step(phi)
        std_final = phi.std().item()
        assert std_final <= std_initial

    def test_compute_total_free_energy(self):
        phi = torch.full((32, 32), 0.5)
        fe = self.solver.compute_total_free_energy(phi)
        expected_bulk = self.fh.free_energy_density(phi).sum().item() * self.grid.dx**2
        assert abs(fe - expected_bulk) < 1e-6


class TestSpectralSolver3D:
    def setup_method(self):
        self.fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        self.functional = FreeEnergyFunctional(local=self.fh, kappa=0.5)
        self.grid = GridParams(shape=(16, 16, 16), dx=1.0, dt=0.1)
        self.solver = SpectralSolver(self.functional, self.grid, mobility=1.0)

    def test_step_preserves_shape(self):
        phi = random_uniform(16, 16, 16, seed=42)
        phi_new = self.solver.step(phi)
        assert phi_new.shape == (16, 16, 16)

    def test_mass_conservation(self):
        phi = random_uniform(16, 16, 16, phi_mean=0.5, seed=42)
        total_mass_before = phi.sum().item()
        phi_new = self.solver.step(phi)
        total_mass_after = phi_new.sum().item()
        assert abs(total_mass_before - total_mass_after) < 1e-3

    def test_mass_conservation_over_many_steps(self):
        phi = random_uniform(16, 16, 16, phi_mean=0.5, seed=42)
        total_mass_0 = phi.sum().item()
        for _ in range(50):
            phi = self.solver.step(phi)
        assert abs(phi.sum().item() - total_mass_0) < 1e-3

    def test_free_energy_decreases(self):
        phi = random_uniform(16, 16, 16, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_initial = self.solver.compute_total_free_energy(phi)
        for _ in range(200):
            phi = self.solver.step(phi)
        fe_final = self.solver.compute_total_free_energy(phi)
        assert fe_final < fe_initial

    def test_compute_total_free_energy_3d(self):
        phi = torch.full((16, 16, 16), 0.5)
        fe = self.solver.compute_total_free_energy(phi)
        expected_bulk = self.fh.free_energy_density(phi).sum().item() * self.grid.dx**3
        assert abs(fe - expected_bulk) < 1e-6

    def test_ndim(self):
        assert self.grid.ndim == 3


class TestSpectralSolverOK2D:
    def setup_method(self):
        self.dw = DoubleWell(W=1.0)
        self.alpha = 0.05
        self.functional = FreeEnergyFunctional(local=self.dw, kappa=0.5, alpha=self.alpha)
        self.grid = GridParams(shape=(32, 32), dx=1.0, dt=0.1)
        self.solver = SpectralSolver(self.functional, self.grid, mobility=1.0)

    def test_step_preserves_shape(self):
        phi = random_uniform(32, 32, seed=42)
        phi_new = self.solver.step(phi)
        assert phi_new.shape == (32, 32)

    def test_mass_conservation(self):
        phi = random_uniform(32, 32, phi_mean=0.5, seed=42)
        total_before = phi.sum().item()
        phi_new = self.solver.step(phi)
        assert abs(phi_new.sum().item() - total_before) < 1e-3

    def test_mass_conservation_over_many_steps(self):
        phi = random_uniform(32, 32, phi_mean=0.5, seed=42)
        total_0 = phi.sum().item()
        for _ in range(100):
            phi = self.solver.step(phi)
        assert abs(phi.sum().item() - total_0) < 1e-3

    def test_free_energy_decreases(self):
        phi = random_uniform(32, 32, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_initial = self.solver.compute_total_free_energy(phi)
        for _ in range(500):
            phi = self.solver.step(phi)
        fe_final = self.solver.compute_total_free_energy(phi)
        assert fe_final < fe_initial

    def test_long_range_energy_nonnegative(self):
        phi = random_uniform(32, 32, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_ok = self.solver.compute_total_free_energy(phi)
        ch_func = FreeEnergyFunctional(local=self.dw, kappa=0.5)
        ch_solver = SpectralSolver(ch_func, self.grid, mobility=1.0)
        fe_ch = ch_solver.compute_total_free_energy(phi)
        assert fe_ok >= fe_ch - 1e-6

    def test_run_returns_snapshots(self):
        phi0 = random_uniform(32, 32, seed=42)
        snapshots = self.solver.run(phi0, n_steps=100, save_interval=50)
        assert len(snapshots) == 3
        assert snapshots[0][0] == 0
        assert snapshots[1][0] == 50
        assert snapshots[2][0] == 100


class TestSpectralSolverOK3D:
    def setup_method(self):
        self.dw = DoubleWell(W=1.0)
        self.functional = FreeEnergyFunctional(local=self.dw, kappa=0.5, alpha=0.05)
        self.grid = GridParams(shape=(16, 16, 16), dx=1.0, dt=0.1)
        self.solver = SpectralSolver(self.functional, self.grid, mobility=1.0)

    def test_step_preserves_shape(self):
        phi = random_uniform(16, 16, 16, seed=42)
        phi_new = self.solver.step(phi)
        assert phi_new.shape == (16, 16, 16)

    def test_mass_conservation(self):
        phi = random_uniform(16, 16, 16, phi_mean=0.5, seed=42)
        total_before = phi.sum().item()
        phi_new = self.solver.step(phi)
        assert abs(phi_new.sum().item() - total_before) < 1e-3

    def test_free_energy_decreases(self):
        phi = random_uniform(16, 16, 16, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_initial = self.solver.compute_total_free_energy(phi)
        for _ in range(200):
            phi = self.solver.step(phi)
        fe_final = self.solver.compute_total_free_energy(phi)
        assert fe_final < fe_initial


# ======================================================================
# Wall condition tests
# ======================================================================


class TestSurfaceEnergyWall:
    def setup_method(self):
        self.grid = GridParams(shape=(32, 32), dx=1.0, dt=0.1)
        self.functional = FreeEnergyFunctional(local=DoubleWell(W=1.0), kappa=0.5)
        self.mask = channel_walls(*self.grid.shape, wall_thickness=3, axis=1)

    def test_surface_delta_concentrated_at_interface(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        delta = wall._surface_delta
        # |∇Ω| should be near zero far from the wall surface
        # The interior of the fluid region (middle of domain) should have ~0
        mid = self.grid.shape[1] // 2
        assert delta[:, mid].max().item() < 0.01

    def test_surface_delta_nonzero_at_wall_surface(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        delta = wall._surface_delta
        # |∇Ω| should be large at the wall-fluid interface
        assert delta.max().item() > 0.1

    def test_chemical_potential_shape(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        phi = random_uniform(*self.grid.shape, seed=42)
        mu = wall.chemical_potential_contribution(phi)
        assert mu.shape == self.grid.shape

    def test_chemical_potential_independent_of_phi(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        phi_a = torch.full(self.grid.shape, 0.3)
        phi_b = torch.full(self.grid.shape, 0.8)
        mu_a = wall.chemical_potential_contribution(phi_a)
        mu_b = wall.chemical_potential_contribution(phi_b)
        # Linear surface energy: μ = −γ|∇Ω| does not depend on φ
        assert torch.allclose(mu_a, mu_b)

    def test_positive_gamma_attracts_phi1(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        phi = random_uniform(*self.grid.shape, seed=42)
        mu = wall.chemical_potential_contribution(phi)
        # γ > 0 → μ_surface < 0 near wall → lowers chemical potential → attracts φ=1
        assert mu.min().item() < 0

    def test_negative_gamma_attracts_phi0(self):
        wall = SurfaceEnergyWall(self.mask, gamma=-1.0)
        phi = random_uniform(*self.grid.shape, seed=42)
        mu = wall.chemical_potential_contribution(phi)
        # γ < 0 → μ_surface > 0 near wall → raises chemical potential → attracts φ=0
        assert mu.max().item() > 0

    def test_zero_gamma_is_neutral(self):
        wall = SurfaceEnergyWall(self.mask, gamma=0.0)
        phi = random_uniform(*self.grid.shape, seed=42)
        mu = wall.chemical_potential_contribution(phi)
        assert mu.abs().max().item() < 1e-10

    def test_stabilization_estimate_is_zero(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        assert wall.stabilization_estimate() == 0.0

    def test_to_device(self):
        wall = SurfaceEnergyWall(self.mask, gamma=1.0)
        wall_cpu = wall.to("cpu")
        assert wall_cpu.mask.device.type == "cpu"
        assert wall_cpu._surface_delta.device.type == "cpu"

    def test_mass_conservation_with_solver(self):
        wall = SurfaceEnergyWall(self.mask, gamma=0.5, dx=self.grid.dx)
        solver = SpectralSolver(
            self.functional, self.grid, mobility=1.0, wall=wall,
        )
        phi = random_uniform(*self.grid.shape, phi_mean=0.5, seed=42)
        total_0 = phi.sum().item()
        for _ in range(100):
            phi = solver.step(phi)
        assert abs(phi.sum().item() - total_0) < 1e-3

    def test_free_energy_decreases_with_solver(self):
        wall = SurfaceEnergyWall(self.mask, gamma=0.5, dx=self.grid.dx)
        solver = SpectralSolver(
            self.functional, self.grid, mobility=1.0, wall=wall,
        )
        phi = random_uniform(*self.grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_initial = solver.compute_total_free_energy(phi)
        for _ in range(500):
            phi = solver.step(phi)
        fe_final = solver.compute_total_free_energy(phi)
        assert fe_final < fe_initial

    def test_surface_energy_wetting_effect(self):
        """Surface energy should drive wetting near the wall."""
        wall = SurfaceEnergyWall(self.mask, gamma=1.0, dx=self.grid.dx)
        solver = SpectralSolver(
            self.functional, self.grid, mobility=1.0, wall=wall,
        )
        phi = random_uniform(*self.grid.shape, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        for _ in range(500):
            phi = solver.step(phi)
        # Near the wall (y=0 and y=N-1) the average φ should be higher
        # than in the bulk (γ > 0 attracts φ=1)
        near_wall = phi[:, :3].mean().item()
        bulk = phi[:, 13:19].mean().item()
        assert near_wall > bulk
