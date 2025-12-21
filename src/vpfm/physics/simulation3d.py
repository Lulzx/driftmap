"""3D VPFM simulation for plasma turbulence.

Extends the 2D VPFM method to three dimensions:
- 3D particle positions and vorticity
- 3D flow map Jacobian (3x3 matrices)
- 3D B-spline interpolation
- Parallel dynamics along magnetic field

For tokamak edge turbulence, the third dimension represents
the parallel (to B) direction, with different physics:
- Perpendicular plane: E×B drift (incompressible)
- Parallel direction: Sound wave dynamics, Landau damping

Reference: Extension of Wang et al. (2025) to 3D.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class Grid3D:
    """3D Eulerian grid."""
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    Lz: float

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

        # Coordinate arrays
        self.x = np.linspace(0, self.Lx - self.dx, self.nx)
        self.y = np.linspace(0, self.Ly - self.dy, self.ny)
        self.z = np.linspace(0, self.Lz - self.dz, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # Fields
        self.q = np.zeros((self.nx, self.ny, self.nz))     # Vorticity/PV
        self.phi = np.zeros((self.nx, self.ny, self.nz))   # Potential
        self.vx = np.zeros((self.nx, self.ny, self.nz))    # E×B velocity x
        self.vy = np.zeros((self.nx, self.ny, self.nz))    # E×B velocity y
        self.vz = np.zeros((self.nx, self.ny, self.nz))    # Parallel velocity

        # Velocity gradients (for Jacobian evolution)
        self.dvx_dx = np.zeros((self.nx, self.ny, self.nz))
        self.dvx_dy = np.zeros((self.nx, self.ny, self.nz))
        self.dvx_dz = np.zeros((self.nx, self.ny, self.nz))
        self.dvy_dx = np.zeros((self.nx, self.ny, self.nz))
        self.dvy_dy = np.zeros((self.nx, self.ny, self.nz))
        self.dvy_dz = np.zeros((self.nx, self.ny, self.nz))
        self.dvz_dx = np.zeros((self.nx, self.ny, self.nz))
        self.dvz_dy = np.zeros((self.nx, self.ny, self.nz))
        self.dvz_dz = np.zeros((self.nx, self.ny, self.nz))


@dataclass
class Particles3D:
    """3D Lagrangian particle system."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    q: np.ndarray  # Vorticity carried by particles

    @property
    def n_particles(self) -> int:
        return len(self.x)

    @classmethod
    def from_grid(cls, grid: Grid3D, particles_per_cell: int = 1) -> 'Particles3D':
        """Initialize particles on a regular grid."""
        n_total = grid.nx * grid.ny * grid.nz * particles_per_cell
        x = np.zeros(n_total)
        y = np.zeros(n_total)
        z = np.zeros(n_total)

        idx = 0
        for i in range(grid.nx):
            for j in range(grid.ny):
                for k in range(grid.nz):
                    for _ in range(particles_per_cell):
                        x[idx] = (i + 0.5) * grid.dx
                        y[idx] = (j + 0.5) * grid.dy
                        z[idx] = (k + 0.5) * grid.dz
                        idx += 1

        return cls(x=x, y=y, z=z, q=np.zeros(n_total))


@dataclass
class FlowMap3D:
    """3D flow map state."""
    J: np.ndarray  # (n_particles, 3, 3) Jacobian
    steps_since_reinit: int = 0

    @classmethod
    def initialize(cls, n_particles: int) -> 'FlowMap3D':
        """Initialize with identity Jacobians."""
        J = np.zeros((n_particles, 3, 3))
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0
        J[:, 2, 2] = 1.0
        return cls(J=J)


# =============================================================================
# Numba-optimized 3D kernels
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _quadratic_bspline_scalar(x: float) -> float:
        """Quadratic B-spline kernel."""
        ax = abs(x)
        if ax < 0.5:
            return 0.75 - ax * ax
        elif ax < 1.5:
            t = 1.5 - ax
            return 0.5 * t * t
        return 0.0

    @njit(cache=True, fastmath=True)
    def _p2g_3d_numba(px, py, pz, pq, nx, ny, nz, dx, dy, dz):
        """3D P2G transfer with quadratic B-spline."""
        n_particles = len(px)
        q_grid = np.zeros((nx, ny, nz))
        weight_grid = np.zeros((nx, ny, nz))

        inv_dx, inv_dy, inv_dz = 1.0/dx, 1.0/dy, 1.0/dz

        for p in range(n_particles):
            x_norm = px[p] * inv_dx
            y_norm = py[p] * inv_dy
            z_norm = pz[p] * inv_dz

            base_i = int(np.floor(x_norm + 0.5)) - 1
            base_j = int(np.floor(y_norm + 0.5)) - 1
            base_k = int(np.floor(z_norm + 0.5)) - 1

            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
            fz = z_norm - (base_k + 1)

            q_p = pq[p]

            for di in range(3):
                xi = (di - 1) - fx
                wx = _quadratic_bspline_scalar(xi)
                gi = (base_i + di) % nx

                for dj in range(3):
                    yj = (dj - 1) - fy
                    wy = _quadratic_bspline_scalar(yj)
                    gj = (base_j + dj) % ny

                    for dk in range(3):
                        zk = (dk - 1) - fz
                        wz = _quadratic_bspline_scalar(zk)
                        gk = (base_k + dk) % nz

                        w = wx * wy * wz
                        q_grid[gi, gj, gk] += w * q_p
                        weight_grid[gi, gj, gk] += w

        return q_grid, weight_grid

    @njit(parallel=True, cache=True, fastmath=True)
    def _g2p_3d_numba(grid_field, px, py, pz, dx, dy, dz):
        """3D G2P interpolation with quadratic B-spline."""
        nx, ny, nz = grid_field.shape
        n_particles = len(px)
        values = np.zeros(n_particles)

        inv_dx, inv_dy, inv_dz = 1.0/dx, 1.0/dy, 1.0/dz

        for p in prange(n_particles):
            x_norm = px[p] * inv_dx
            y_norm = py[p] * inv_dy
            z_norm = pz[p] * inv_dz

            base_i = int(np.floor(x_norm + 0.5)) - 1
            base_j = int(np.floor(y_norm + 0.5)) - 1
            base_k = int(np.floor(z_norm + 0.5)) - 1

            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
            fz = z_norm - (base_k + 1)

            val = 0.0
            for di in range(3):
                xi = (di - 1) - fx
                wx = _quadratic_bspline_scalar(xi)
                gi = (base_i + di) % nx

                for dj in range(3):
                    yj = (dj - 1) - fy
                    wy = _quadratic_bspline_scalar(yj)
                    gj = (base_j + dj) % ny

                    for dk in range(3):
                        zk = (dk - 1) - fz
                        wz = _quadratic_bspline_scalar(zk)
                        gk = (base_k + dk) % nz

                        val += wx * wy * wz * grid_field[gi, gj, gk]

            values[p] = val

        return values

    @njit(parallel=True, cache=True, fastmath=True)
    def _jacobian_rhs_3d(J, grad_v):
        """Compute dJ/dt = -J · ∇v for 3D."""
        n = J.shape[0]
        dJ = np.zeros_like(J)

        for p in prange(n):
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        dJ[p, i, k] -= J[p, i, j] * grad_v[p, j, k]

        return dJ

    @njit(parallel=True, cache=True, fastmath=True)
    def _rk4_positions_3d(x0, y0, z0, vx1, vy1, vz1, vx2, vy2, vz2,
                          vx3, vy3, vz3, vx4, vy4, vz4, dt, Lx, Ly, Lz):
        """3D RK4 position update."""
        n = len(x0)
        x_new = np.empty(n)
        y_new = np.empty(n)
        z_new = np.empty(n)

        dt6 = dt / 6.0

        for p in prange(n):
            x_new[p] = x0[p] + dt6 * (vx1[p] + 2*vx2[p] + 2*vx3[p] + vx4[p])
            y_new[p] = y0[p] + dt6 * (vy1[p] + 2*vy2[p] + 2*vy3[p] + vy4[p])
            z_new[p] = z0[p] + dt6 * (vz1[p] + 2*vz2[p] + 2*vz3[p] + vz4[p])

            # Periodic wrap
            x_new[p] = x_new[p] - Lx * np.floor(x_new[p] / Lx)
            y_new[p] = y_new[p] - Ly * np.floor(y_new[p] / Ly)
            z_new[p] = z_new[p] - Lz * np.floor(z_new[p] / Lz)

        return x_new, y_new, z_new

else:
    # Fallback non-Numba implementations
    def _quadratic_bspline_scalar(x):
        ax = abs(x)
        if ax < 0.5:
            return 0.75 - ax * ax
        elif ax < 1.5:
            t = 1.5 - ax
            return 0.5 * t * t
        return 0.0

    def _p2g_3d_numba(px, py, pz, pq, nx, ny, nz, dx, dy, dz):
        # Simple loop fallback
        n_particles = len(px)
        q_grid = np.zeros((nx, ny, nz))
        weight_grid = np.zeros((nx, ny, nz))

        inv_dx, inv_dy, inv_dz = 1.0/dx, 1.0/dy, 1.0/dz

        for p in range(n_particles):
            x_norm = px[p] * inv_dx
            y_norm = py[p] * inv_dy
            z_norm = pz[p] * inv_dz

            base_i = int(np.floor(x_norm + 0.5)) - 1
            base_j = int(np.floor(y_norm + 0.5)) - 1
            base_k = int(np.floor(z_norm + 0.5)) - 1

            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
            fz = z_norm - (base_k + 1)

            q_p = pq[p]

            for di in range(3):
                wx = _quadratic_bspline_scalar((di - 1) - fx)
                gi = (base_i + di) % nx

                for dj in range(3):
                    wy = _quadratic_bspline_scalar((dj - 1) - fy)
                    gj = (base_j + dj) % ny

                    for dk in range(3):
                        wz = _quadratic_bspline_scalar((dk - 1) - fz)
                        gk = (base_k + dk) % nz

                        w = wx * wy * wz
                        q_grid[gi, gj, gk] += w * q_p
                        weight_grid[gi, gj, gk] += w

        return q_grid, weight_grid

    def _g2p_3d_numba(grid_field, px, py, pz, dx, dy, dz):
        nx, ny, nz = grid_field.shape
        n_particles = len(px)
        values = np.zeros(n_particles)

        inv_dx, inv_dy, inv_dz = 1.0/dx, 1.0/dy, 1.0/dz

        for p in range(n_particles):
            x_norm = px[p] * inv_dx
            y_norm = py[p] * inv_dy
            z_norm = pz[p] * inv_dz

            base_i = int(np.floor(x_norm + 0.5)) - 1
            base_j = int(np.floor(y_norm + 0.5)) - 1
            base_k = int(np.floor(z_norm + 0.5)) - 1

            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
            fz = z_norm - (base_k + 1)

            val = 0.0
            for di in range(3):
                wx = _quadratic_bspline_scalar((di - 1) - fx)
                gi = (base_i + di) % nx

                for dj in range(3):
                    wy = _quadratic_bspline_scalar((dj - 1) - fy)
                    gj = (base_j + dj) % ny

                    for dk in range(3):
                        wz = _quadratic_bspline_scalar((dk - 1) - fz)
                        gk = (base_k + dk) % nz

                        val += wx * wy * wz * grid_field[gi, gj, gk]

            values[p] = val

        return values

    def _jacobian_rhs_3d(J, grad_v):
        n = J.shape[0]
        dJ = np.zeros_like(J)
        for p in range(n):
            dJ[p] = -J[p] @ grad_v[p]
        return dJ

    def _rk4_positions_3d(x0, y0, z0, vx1, vy1, vz1, vx2, vy2, vz2,
                          vx3, vy3, vz3, vx4, vy4, vz4, dt, Lx, Ly, Lz):
        dt6 = dt / 6.0
        x_new = x0 + dt6 * (vx1 + 2*vx2 + 2*vx3 + vx4)
        y_new = y0 + dt6 * (vy1 + 2*vy2 + 2*vy3 + vy4)
        z_new = z0 + dt6 * (vz1 + 2*vz2 + 2*vz3 + vz4)
        x_new = x_new % Lx
        y_new = y_new % Ly
        z_new = z_new % Lz
        return x_new, y_new, z_new


class Simulation3D:
    """3D VPFM simulation for plasma turbulence.

    Extends 2D VPFM to three dimensions with:
    - Full 3D particle advection
    - 3x3 Jacobian evolution
    - Parallel dynamics (sound waves, Landau damping)

    The third dimension (z) represents the parallel-to-B direction.
    """

    def __init__(self,
                 nx: int = 64,
                 ny: int = 64,
                 nz: int = 32,
                 Lx: float = 2 * np.pi,
                 Ly: float = 2 * np.pi,
                 Lz: float = 2 * np.pi,
                 dt: float = 0.01,
                 particles_per_cell: int = 1,
                 reinit_threshold: float = 2.0,
                 max_reinit_steps: int = 100,
                 # Physics parameters
                 cs: float = 1.0,      # Sound speed (parallel)
                 nu_par: float = 0.0,  # Parallel viscosity
                 alpha: float = 0.0,   # HW adiabaticity
                 kappa: float = 0.0):  # Curvature drive
        """Initialize 3D simulation.

        Args:
            nx, ny, nz: Grid resolution
            Lx, Ly, Lz: Domain size
            dt: Time step
            particles_per_cell: Particle density
            reinit_threshold: Flow map error threshold
            max_reinit_steps: Max steps between reinitialization
            cs: Sound speed for parallel dynamics
            nu_par: Parallel viscosity
            alpha: HW adiabaticity parameter
            kappa: Curvature drive
        """
        self.grid = Grid3D(nx, ny, nz, Lx, Ly, Lz)
        self.particles = Particles3D.from_grid(self.grid, particles_per_cell)
        self.flow_map = FlowMap3D.initialize(self.particles.n_particles)

        self.dt = dt
        self.time = 0.0
        self.step_count = 0

        self.reinit_threshold = reinit_threshold
        self.max_reinit_steps = max_reinit_steps

        # Physics
        self.cs = cs
        self.nu_par = nu_par
        self.alpha = alpha
        self.kappa = kappa

        # Density on particles (for HW)
        self.n_particles = np.zeros(self.particles.n_particles)
        self.n_grid = np.zeros((nx, ny, nz))

        # Wave numbers for spectral operations
        kx = fftfreq(nx, Lx/nx) * 2 * np.pi
        ky = fftfreq(ny, Ly/ny) * 2 * np.pi
        kz = fftfreq(nz, Lz/nz) * 2 * np.pi
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K2_perp = self.KX**2 + self.KY**2  # Perpendicular only

        # Preallocated velocity gradient array
        self._grad_v = np.zeros((self.particles.n_particles, 3, 3))

        # History
        self.history = {
            'time': [],
            'energy': [],
            'enstrophy': [],
            'reinit_count': 0,
        }

    def _p2g(self):
        """Transfer vorticity from particles to grid."""
        q_grid, weight_grid = _p2g_3d_numba(
            self.particles.x, self.particles.y, self.particles.z,
            self.particles.q,
            self.grid.nx, self.grid.ny, self.grid.nz,
            self.grid.dx, self.grid.dy, self.grid.dz
        )
        mask = weight_grid > 1e-10
        q_grid[mask] /= weight_grid[mask]
        self.grid.q = q_grid

    def _g2p(self, field: np.ndarray) -> np.ndarray:
        """Interpolate grid field to particles."""
        return _g2p_3d_numba(
            field,
            self.particles.x, self.particles.y, self.particles.z,
            self.grid.dx, self.grid.dy, self.grid.dz
        )

    def _solve_poisson(self):
        """Solve 3D Poisson equation: ∇²φ = ζ."""
        zeta_hat = fftn(self.grid.q)

        K2_safe = self.K2.copy()
        K2_safe[0, 0, 0] = 1.0

        phi_hat = -zeta_hat / K2_safe
        phi_hat[0, 0, 0] = 0

        self.grid.phi = np.real(ifftn(phi_hat))

    def _compute_velocity(self):
        """Compute E×B velocity: v_perp = z × ∇φ, v_par from parallel dynamics."""
        phi_hat = fftn(self.grid.phi)

        # Perpendicular E×B drift
        dphi_dx = np.real(ifftn(1j * self.KX * phi_hat))
        dphi_dy = np.real(ifftn(1j * self.KY * phi_hat))

        self.grid.vx = -dphi_dy  # v_x = -∂φ/∂y
        self.grid.vy = dphi_dx   # v_y = ∂φ/∂x

        # Parallel velocity (sound wave): v_z = cs * ∂φ/∂z
        if abs(self.cs) > 1e-10:
            dphi_dz = np.real(ifftn(1j * self.KZ * phi_hat))
            self.grid.vz = self.cs * dphi_dz
        else:
            self.grid.vz.fill(0)

    def _compute_velocity_gradient(self):
        """Compute velocity gradient tensor on grid."""
        vx_hat = fftn(self.grid.vx)
        vy_hat = fftn(self.grid.vy)
        vz_hat = fftn(self.grid.vz)

        self.grid.dvx_dx = np.real(ifftn(1j * self.KX * vx_hat))
        self.grid.dvx_dy = np.real(ifftn(1j * self.KY * vx_hat))
        self.grid.dvx_dz = np.real(ifftn(1j * self.KZ * vx_hat))

        self.grid.dvy_dx = np.real(ifftn(1j * self.KX * vy_hat))
        self.grid.dvy_dy = np.real(ifftn(1j * self.KY * vy_hat))
        self.grid.dvy_dz = np.real(ifftn(1j * self.KZ * vy_hat))

        self.grid.dvz_dx = np.real(ifftn(1j * self.KX * vz_hat))
        self.grid.dvz_dy = np.real(ifftn(1j * self.KY * vz_hat))
        self.grid.dvz_dz = np.real(ifftn(1j * self.KZ * vz_hat))

    def _interpolate_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate velocity to particle positions."""
        vx = self._g2p(self.grid.vx)
        vy = self._g2p(self.grid.vy)
        vz = self._g2p(self.grid.vz)
        return vx, vy, vz

    def _interpolate_velocity_gradient(self) -> np.ndarray:
        """Interpolate velocity gradient to particles."""
        n = self.particles.n_particles

        self._grad_v[:, 0, 0] = self._g2p(self.grid.dvx_dx)
        self._grad_v[:, 0, 1] = self._g2p(self.grid.dvx_dy)
        self._grad_v[:, 0, 2] = self._g2p(self.grid.dvx_dz)

        self._grad_v[:, 1, 0] = self._g2p(self.grid.dvy_dx)
        self._grad_v[:, 1, 1] = self._g2p(self.grid.dvy_dy)
        self._grad_v[:, 1, 2] = self._g2p(self.grid.dvy_dz)

        self._grad_v[:, 2, 0] = self._g2p(self.grid.dvz_dx)
        self._grad_v[:, 2, 1] = self._g2p(self.grid.dvz_dy)
        self._grad_v[:, 2, 2] = self._g2p(self.grid.dvz_dz)

        return self._grad_v

    def _advect_particles_rk4(self):
        """Advect particles using RK4."""
        x0 = self.particles.x.copy()
        y0 = self.particles.y.copy()
        z0 = self.particles.z.copy()
        Lx, Ly, Lz = self.grid.Lx, self.grid.Ly, self.grid.Lz
        dt = self.dt

        # k1
        vx1, vy1, vz1 = self._interpolate_velocity()

        # k2
        self.particles.x = (x0 + 0.5*dt*vx1) % Lx
        self.particles.y = (y0 + 0.5*dt*vy1) % Ly
        self.particles.z = (z0 + 0.5*dt*vz1) % Lz
        vx2, vy2, vz2 = self._interpolate_velocity()

        # k3
        self.particles.x = (x0 + 0.5*dt*vx2) % Lx
        self.particles.y = (y0 + 0.5*dt*vy2) % Ly
        self.particles.z = (z0 + 0.5*dt*vz2) % Lz
        vx3, vy3, vz3 = self._interpolate_velocity()

        # k4
        self.particles.x = (x0 + dt*vx3) % Lx
        self.particles.y = (y0 + dt*vy3) % Ly
        self.particles.z = (z0 + dt*vz3) % Lz
        vx4, vy4, vz4 = self._interpolate_velocity()

        # Final update
        self.particles.x, self.particles.y, self.particles.z = _rk4_positions_3d(
            x0, y0, z0, vx1, vy1, vz1, vx2, vy2, vz2,
            vx3, vy3, vz3, vx4, vy4, vz4, dt, Lx, Ly, Lz
        )

    def _evolve_jacobian_rk4(self):
        """Evolve Jacobian using RK4: dJ/dt = -J·∇v."""
        J = self.flow_map.J
        dt = self.dt

        grad_v = self._interpolate_velocity_gradient()

        # Simple Euler for now (RK4 would require storing intermediate positions)
        dJ = _jacobian_rhs_3d(J, grad_v)
        self.flow_map.J = J + dt * dJ

    def _estimate_error(self) -> float:
        """Estimate flow map error."""
        J = self.flow_map.J
        I = np.eye(3)
        errors = np.sqrt(np.sum((J - I)**2, axis=(1, 2)))
        return np.max(errors)

    def _should_reinitialize(self) -> bool:
        """Check if reinitialization is needed."""
        if self.flow_map.steps_since_reinit >= self.max_reinit_steps:
            return True
        return self._estimate_error() > self.reinit_threshold

    def _reinitialize(self):
        """Reinitialize flow map."""
        # Reset Jacobian to identity
        self.flow_map.J[:, :, :] = 0
        self.flow_map.J[:, 0, 0] = 1.0
        self.flow_map.J[:, 1, 1] = 1.0
        self.flow_map.J[:, 2, 2] = 1.0
        self.flow_map.steps_since_reinit = 0

        # Update particle values from grid
        self.particles.q = self._g2p(self.grid.q)
        self.history['reinit_count'] += 1

    def set_initial_condition(self, q_func: Callable):
        """Set initial vorticity from function q(x, y, z)."""
        self.particles.q = q_func(self.particles.x, self.particles.y, self.particles.z)
        self._p2g()
        self._solve_poisson()
        self._compute_velocity()
        self._compute_velocity_gradient()

    def advance(self):
        """Advance one time step."""
        # 1. P2G
        self._p2g()

        # 2. Solve Poisson
        self._solve_poisson()

        # 3. Compute velocity
        self._compute_velocity()
        self._compute_velocity_gradient()

        # 4. Evolve Jacobian
        self._evolve_jacobian_rk4()

        # 5. Advect particles
        self._advect_particles_rk4()

        # 6. Update time
        self.time += self.dt
        self.step_count += 1
        self.flow_map.steps_since_reinit += 1

        # 7. Check reinitialization
        if self._should_reinitialize():
            self._reinitialize()

    def compute_diagnostics(self) -> dict:
        """Compute diagnostics."""
        dV = self.grid.dx * self.grid.dy * self.grid.dz

        # Kinetic energy: 0.5 * |∇φ|²
        phi_hat = fftn(self.grid.phi)
        dphi_dx = np.real(ifftn(1j * self.KX * phi_hat))
        dphi_dy = np.real(ifftn(1j * self.KY * phi_hat))
        dphi_dz = np.real(ifftn(1j * self.KZ * phi_hat))
        energy = 0.5 * np.sum(dphi_dx**2 + dphi_dy**2 + dphi_dz**2) * dV

        # Enstrophy: 0.5 * ζ²
        enstrophy = 0.5 * np.sum(self.grid.q**2) * dV

        return {
            'energy': energy,
            'enstrophy': enstrophy,
            'max_q': np.max(np.abs(self.grid.q)),
            'jacobian_error': self._estimate_error(),
        }

    def run(self, n_steps: int, diag_interval: int = 10,
            verbose: bool = True, progress: bool = True):
        """Run simulation."""
        iterator = tqdm(range(n_steps), desc="3D Simulation", disable=not progress)

        for i in iterator:
            self.advance()

            if (self.step_count % diag_interval == 0) or (i == n_steps - 1):
                diag = self.compute_diagnostics()

                self.history['time'].append(self.time)
                self.history['energy'].append(diag['energy'])
                self.history['enstrophy'].append(diag['enstrophy'])

                if verbose and self.step_count % (10 * diag_interval) == 0:
                    print(f"Step {self.step_count:5d}, t={self.time:.2f}, "
                          f"E={diag['energy']:.4f}, Z={diag['enstrophy']:.4f}, "
                          f"|J-I|={diag['jacobian_error']:.3f}")


# =============================================================================
# 3D initial conditions
# =============================================================================

def gaussian_blob_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     x0: float, y0: float, z0: float,
                     amplitude: float = 1.0,
                     rx: float = 1.0, ry: float = 1.0, rz: float = 1.0) -> np.ndarray:
    """3D Gaussian blob initial condition."""
    return amplitude * np.exp(
        -((x - x0)**2 / rx**2 + (y - y0)**2 / ry**2 + (z - z0)**2 / rz**2)
    )


def random_turbulence_3d(nx: int, ny: int, nz: int,
                         Lx: float, Ly: float, Lz: float,
                         k_peak: float = 5.0,
                         amplitude: float = 0.1,
                         seed: Optional[int] = None) -> np.ndarray:
    """Random 3D turbulent initial condition."""
    if seed is not None:
        np.random.seed(seed)

    kx = fftfreq(nx, Lx/nx) * 2 * np.pi
    ky = fftfreq(ny, Ly/ny) * 2 * np.pi
    kz = fftfreq(nz, Lz/nz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Energy spectrum
    E_k = K**4 * np.exp(-(K / k_peak)**2)

    # Random phases
    phases = np.random.uniform(0, 2*np.pi, (nx, ny, nz))

    zeta_hat = np.sqrt(E_k) * np.exp(1j * phases)
    zeta_hat[0, 0, 0] = 0

    return amplitude * np.real(ifftn(zeta_hat))
