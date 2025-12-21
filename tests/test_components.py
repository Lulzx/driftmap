"""Unit tests for VPFM-Plasma components."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from vpfm import Grid, ParticleSystem
from vpfm.core.transfers import P2G_vectorized, G2P
from vpfm.core.flow_map import _estimate_jacobian_error
from vpfm.numerics.poisson import solve_poisson_hm, solve_poisson_standard
from vpfm.numerics.velocity import compute_velocity, compute_velocity_gradient


class TestGrid:
    """Tests for Grid class."""

    def test_grid_initialization(self):
        """Test grid is properly initialized."""
        grid = Grid(64, 64, 2 * np.pi, 2 * np.pi)

        assert grid.nx == 64
        assert grid.ny == 64
        assert grid.q.shape == (64, 64)
        assert grid.phi.shape == (64, 64)

    def test_grid_spacing(self):
        """Test grid spacing is correct."""
        nx, ny = 32, 32
        Lx, Ly = 4.0, 4.0
        grid = Grid(nx, ny, Lx, Ly)

        assert np.isclose(grid.dx, Lx / nx)
        assert np.isclose(grid.dy, Ly / ny)

    def test_wrap_coordinates(self):
        """Test periodic wrapping of coordinates."""
        grid = Grid(32, 32, 2 * np.pi, 2 * np.pi)

        x = np.array([0.0, 2 * np.pi + 1.0, -1.0])
        y = np.array([0.0, 2 * np.pi + 0.5, -0.5])

        x_w, y_w = grid.wrap_coordinates(x, y)

        assert np.allclose(x_w, [0.0, 1.0, 2 * np.pi - 1.0])
        assert np.allclose(y_w, [0.0, 0.5, 2 * np.pi - 0.5])


class TestParticles:
    """Tests for ParticleSystem class."""

    def test_particle_initialization(self):
        """Test particle system initialization."""
        ps = ParticleSystem(100)

        assert ps.n_particles == 100
        assert ps.x.shape == (100,)
        assert ps.q.shape == (100,)
        assert ps.J.shape == (100, 2, 2)

    def test_jacobian_identity(self):
        """Test Jacobians are initialized to identity."""
        ps = ParticleSystem(50)

        for p in range(ps.n_particles):
            assert np.allclose(ps.J[p], np.eye(2))

    def test_from_grid(self):
        """Test particle seeding from grid."""
        grid = Grid(8, 8, 2 * np.pi, 2 * np.pi)
        ps = ParticleSystem.from_grid(grid, particles_per_cell=1)

        assert ps.n_particles == 64
        assert np.all(ps.x >= 0)
        assert np.all(ps.x < grid.Lx)
        assert np.all(ps.y >= 0)
        assert np.all(ps.y < grid.Ly)

    def test_jacobian_error_estimate(self):
        """Jacobian error should be non-zero for perturbed matrices."""
        J = np.zeros((2, 2, 2))
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0
        J[0, 0, 0] = 1.2  # Perturb one particle

        err = _estimate_jacobian_error(J)
        assert err > 0.1


class TestTransfers:
    """Tests for P2G and G2P operations."""

    def test_p2g_conservation(self):
        """Test P2G conserves total vorticity."""
        grid = Grid(32, 32, 2 * np.pi, 2 * np.pi)
        ps = ParticleSystem.from_grid(grid)

        # Set uniform vorticity on particles
        ps.q = np.ones(ps.n_particles)

        q_grid = P2G_vectorized(ps, grid)

        # Total should be conserved (approximately)
        total_particle = np.sum(ps.q)
        total_grid = np.sum(q_grid)

        # With uniform distribution, should be very close
        assert np.isclose(total_grid, total_particle, rtol=0.1)

    def test_g2p_interpolation(self):
        """Test G2P interpolates correctly."""
        grid = Grid(32, 32, 2 * np.pi, 2 * np.pi)
        ps = ParticleSystem.from_grid(grid)

        # Set a smooth, low-frequency field (better for bilinear interpolation)
        field = np.sin(grid.X) * np.cos(grid.Y)

        # Interpolate to particles at cell centers
        values = G2P(grid, ps, field)

        # The interpolated values should be reasonable
        # With particles at cell centers, bilinear interpolation averages neighbors
        # Check that values are in the right range and not NaN
        assert np.all(np.isfinite(values))
        assert np.max(np.abs(values)) <= 1.1  # sin*cos bounded by 1

        # Check correlation is high (interpolation preserves structure)
        expected = np.sin(ps.x) * np.cos(ps.y)
        correlation = np.corrcoef(values.flatten(), expected.flatten())[0, 1]
        assert correlation > 0.9  # Strong correlation


class TestPoisson:
    """Tests for Poisson solver."""

    def test_poisson_standard(self):
        """Test standard Poisson solver."""
        nx, ny = 64, 64
        Lx, Ly = 2 * np.pi, 2 * np.pi
        dx, dy = Lx / nx, Ly / ny

        # Create a test vorticity field
        x = np.linspace(dx/2, Lx - dx/2, nx)
        y = np.linspace(dy/2, Ly - dy/2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # For zeta = sin(x) * sin(y), phi should be -0.5 * sin(x) * sin(y)
        zeta = np.sin(X) * np.sin(Y)
        phi = solve_poisson_standard(zeta, Lx, Ly)

        expected = -0.5 * np.sin(X) * np.sin(Y)

        assert np.allclose(phi, expected, atol=0.01)

    def test_poisson_hm(self):
        """Test Hasegawa-Mima Poisson solver."""
        nx, ny = 64, 64
        Lx, Ly = 2 * np.pi, 2 * np.pi

        x = np.linspace(Lx/(2*nx), Lx - Lx/(2*nx), nx)
        y = np.linspace(Ly/(2*ny), Ly - Ly/(2*ny), ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Simple test: solve and verify (nabla^2 - 1) phi = -q
        q = np.sin(X) * np.sin(Y)
        phi = solve_poisson_hm(q, Lx, Ly)

        # Verify by computing (nabla^2 - 1) phi + q ≈ 0
        from numpy.fft import fft2, ifft2, fftfreq
        kx = fftfreq(nx, Lx / nx) * 2 * np.pi
        ky = fftfreq(ny, Ly / ny) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        phi_hat = fft2(phi)
        laplacian_phi = np.real(ifft2(-(KX**2 + KY**2) * phi_hat))

        residual = laplacian_phi - phi + q
        assert np.allclose(residual, 0, atol=1e-10)


class TestVelocity:
    """Tests for velocity computation."""

    def test_velocity_divergence_free(self):
        """Test that E×B velocity is divergence-free."""
        grid = Grid(32, 32, 2 * np.pi, 2 * np.pi)

        # Set a test potential
        grid.phi = np.sin(grid.X) * np.cos(grid.Y)

        compute_velocity(grid)
        compute_velocity_gradient(grid)

        # Divergence should be zero: dvx/dx + dvy/dy = 0
        divergence = grid.dvx_dx + grid.dvy_dy

        assert np.allclose(divergence, 0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
