"""Tests for the phase-field simulation."""

import torch
import pytest

from torch_pf import (
    CahnHilliardSolver,
    FloryHuggins,
    SimulationParams,
    random_uniform,
    circular_droplet,
)


class TestFloryHuggins:
    def test_critical_chi_symmetric(self):
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        # For symmetric blend, χ_c = 2/N
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
        # At χ = χ_c and symmetric blend, μ should be ~0 at φ = 0.5
        # (1/N)(ln(0.5)+1) - (1/N)(ln(0.5)+1) + χ(1-2*0.5) = 0
        assert abs(mu.item()) < 1e-6

    def test_second_derivative_unstable(self):
        # Above χ_c, d²f/dφ² < 0 at critical composition → spinodal instability
        fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        phi = torch.tensor([0.5])
        d2f = fh.second_derivative(phi)
        assert d2f.item() < 0

    def test_second_derivative_stable(self):
        # Below χ_c, d²f/dφ² > 0 at critical composition → stable
        fh = FloryHuggins(chi=0.01, n_a=100, n_b=100)
        phi = torch.tensor([0.5])
        d2f = fh.second_derivative(phi)
        assert d2f.item() > 0


class TestInitializers:
    def test_random_uniform_shape(self):
        phi = random_uniform(64, 64)
        assert phi.shape == (64, 64)

    def test_random_uniform_mean(self):
        phi = random_uniform(256, 256, phi_mean=0.4, noise_amplitude=0.01, seed=0)
        assert abs(phi.mean().item() - 0.4) < 0.01

    def test_random_uniform_bounds(self):
        phi = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.1)
        assert phi.min() > 0
        assert phi.max() < 1

    def test_random_uniform_reproducible(self):
        a = random_uniform(32, 32, seed=123)
        b = random_uniform(32, 32, seed=123)
        assert torch.allclose(a, b)

    def test_circular_droplet_shape(self):
        phi = circular_droplet(64, 64)
        assert phi.shape == (64, 64)

    def test_circular_droplet_center(self):
        phi = circular_droplet(64, 64, phi_inside=0.8, phi_outside=0.2)
        # Center should be close to phi_inside
        assert phi[32, 32].item() > 0.7

    def test_circular_droplet_corner(self):
        phi = circular_droplet(64, 64, phi_inside=0.8, phi_outside=0.2)
        # Corner should be close to phi_outside
        assert phi[0, 0].item() < 0.3


class TestCahnHilliardSolver:
    def setup_method(self):
        self.params = SimulationParams(nx=32, ny=32, dx=1.0, dt=0.1, mobility=1.0, kappa=0.5)
        self.fh = FloryHuggins(chi=0.03, n_a=100, n_b=100)
        self.solver = CahnHilliardSolver(self.params, self.fh)

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
        # Should have snapshots at steps: 0, 50, 100
        assert len(snapshots) == 3
        assert snapshots[0][0] == 0
        assert snapshots[1][0] == 50
        assert snapshots[2][0] == 100

    def test_free_energy_decreases(self):
        """Total free energy must decrease over time (dissipative system)."""
        phi = random_uniform(32, 32, phi_mean=0.5, noise_amplitude=0.05, seed=42)
        fe_initial = self.solver.compute_total_free_energy(phi)
        for _ in range(500):
            phi = self.solver.step(phi)
        fe_final = self.solver.compute_total_free_energy(phi)
        assert fe_final < fe_initial

    def test_homogeneous_state_stable_below_chi_c(self):
        """Below χ_c, a homogeneous state should remain homogeneous."""
        fh_stable = FloryHuggins(chi=0.01, n_a=100, n_b=100)
        solver = CahnHilliardSolver(self.params, fh_stable)
        phi = random_uniform(32, 32, phi_mean=0.5, noise_amplitude=0.001, seed=42)
        std_initial = phi.std().item()
        for _ in range(200):
            phi = solver.step(phi)
        std_final = phi.std().item()
        # Fluctuations should decay for stable system
        assert std_final <= std_initial

    def test_compute_total_free_energy(self):
        phi = torch.full((32, 32), 0.5)
        fe = self.solver.compute_total_free_energy(phi)
        # Uniform field → no gradient energy, only bulk
        expected_bulk = self.fh.free_energy_density(phi).sum().item() * self.params.dx**2
        assert abs(fe - expected_bulk) < 1e-6
