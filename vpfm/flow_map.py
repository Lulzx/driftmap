"""Advanced flow map evolution for VPFM-Plasma.

Implements:
- RK4 Jacobian evolution (not just Euler)
- Hessian tracking for gradient accuracy
- Adaptive reinitialization based on error estimates
- Jacobian composition for long flow maps

Performance optimized with Numba JIT compilation.

Reference: Wang et al. (2025) "Fluid Simulation on Vortex Particle Flow Maps"
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from numba import njit, prange

from .grid import Grid
from .particles import ParticleSystem
from .kernels import InterpolationKernel, G2P_bspline, G2P_bspline_with_gradient


@dataclass
class FlowMapState:
    """State of the flow map for a particle system."""
    # Jacobian J = ∇F (deformation gradient)
    J: np.ndarray  # (n_particles, 2, 2)

    # Hessian H = ∇²F (for gradient accuracy)
    # H[p, i, j, k] = ∂²F_i / ∂x_j ∂x_k
    H: Optional[np.ndarray] = None  # (n_particles, 2, 2, 2)

    # Tracking
    steps_since_reinit: int = 0
    cumulative_error: float = 0.0


# =============================================================================
# Numba-optimized functions for flow map evolution
# =============================================================================

@njit(cache=True, fastmath=True)
def _jacobian_rhs_numba(J: np.ndarray, grad_v: np.ndarray) -> np.ndarray:
    """Compute dJ/dt = -J · ∇v (Numba-optimized).

    Args:
        J: Jacobian matrices (n, 2, 2)
        grad_v: Velocity gradient (n, 2, 2)

    Returns:
        dJ/dt (n, 2, 2)
    """
    n = J.shape[0]
    dJ = np.zeros_like(J)

    for p in range(n):
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    dJ[p, i, k] -= J[p, i, j] * grad_v[p, j, k]

    return dJ


@njit(parallel=True, cache=True, fastmath=True)
def _jacobian_rhs_parallel(J: np.ndarray, grad_v: np.ndarray) -> np.ndarray:
    """Parallel version of dJ/dt = -J · ∇v."""
    n = J.shape[0]
    dJ = np.zeros_like(J)

    for p in prange(n):
        # Unrolled 2x2 matrix multiply for speed
        dJ[p, 0, 0] = -(J[p, 0, 0] * grad_v[p, 0, 0] + J[p, 0, 1] * grad_v[p, 1, 0])
        dJ[p, 0, 1] = -(J[p, 0, 0] * grad_v[p, 0, 1] + J[p, 0, 1] * grad_v[p, 1, 1])
        dJ[p, 1, 0] = -(J[p, 1, 0] * grad_v[p, 0, 0] + J[p, 1, 1] * grad_v[p, 1, 0])
        dJ[p, 1, 1] = -(J[p, 1, 0] * grad_v[p, 0, 1] + J[p, 1, 1] * grad_v[p, 1, 1])

    return dJ


@njit(parallel=True, cache=True, fastmath=True)
def _rk4_positions(x0: np.ndarray, y0: np.ndarray,
                   vx1: np.ndarray, vy1: np.ndarray,
                   vx2: np.ndarray, vy2: np.ndarray,
                   vx3: np.ndarray, vy3: np.ndarray,
                   vx4: np.ndarray, vy4: np.ndarray,
                   dt: float, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 position update with periodic boundary conditions."""
    n = len(x0)
    x_new = np.empty(n)
    y_new = np.empty(n)

    dt6 = dt / 6.0

    for p in prange(n):
        x_new[p] = x0[p] + dt6 * (vx1[p] + 2.0*vx2[p] + 2.0*vx3[p] + vx4[p])
        y_new[p] = y0[p] + dt6 * (vy1[p] + 2.0*vy2[p] + 2.0*vy3[p] + vy4[p])

        # Periodic wrap
        x_new[p] = x_new[p] - Lx * np.floor(x_new[p] / Lx)
        y_new[p] = y_new[p] - Ly * np.floor(y_new[p] / Ly)

    return x_new, y_new


@njit(parallel=True, cache=True, fastmath=True)
def _midpoint_positions(x0: np.ndarray, y0: np.ndarray,
                        vx: np.ndarray, vy: np.ndarray,
                        dt: float, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute midpoint positions for RK stages."""
    n = len(x0)
    x_mid = np.empty(n)
    y_mid = np.empty(n)

    for p in prange(n):
        x_mid[p] = x0[p] + dt * vx[p]
        y_mid[p] = y0[p] + dt * vy[p]

        # Periodic wrap
        x_mid[p] = x_mid[p] - Lx * np.floor(x_mid[p] / Lx)
        y_mid[p] = y_mid[p] - Ly * np.floor(y_mid[p] / Ly)

    return x_mid, y_mid


@njit(parallel=True, cache=True, fastmath=True)
def _rk4_jacobian_update(J: np.ndarray, k1: np.ndarray, k2: np.ndarray,
                          k3: np.ndarray, k4: np.ndarray, dt: float) -> np.ndarray:
    """RK4 Jacobian update."""
    n = J.shape[0]
    J_new = np.empty_like(J)
    dt6 = dt / 6.0

    for p in prange(n):
        for i in range(2):
            for j in range(2):
                J_new[p, i, j] = J[p, i, j] + dt6 * (k1[p, i, j] + 2.0*k2[p, i, j] +
                                                      2.0*k3[p, i, j] + k4[p, i, j])

    return J_new


@njit(parallel=True, cache=True, fastmath=True)
def _estimate_jacobian_error(J: np.ndarray) -> float:
    """Compute max ||J - I|| across all particles."""
    n = J.shape[0]
    max_err = 0.0

    for p in prange(n):
        # Deviation from identity
        d00 = J[p, 0, 0] - 1.0
        d01 = J[p, 0, 1]
        d10 = J[p, 1, 0]
        d11 = J[p, 1, 1] - 1.0

        err = np.sqrt(d00*d00 + d01*d01 + d10*d10 + d11*d11)
        if err > max_err:
            max_err = err

    return max_err


@njit(parallel=True, cache=True, fastmath=True)
def _reset_jacobian(J: np.ndarray):
    """Reset Jacobian to identity."""
    n = J.shape[0]
    for p in prange(n):
        J[p, 0, 0] = 1.0
        J[p, 0, 1] = 0.0
        J[p, 1, 0] = 0.0
        J[p, 1, 1] = 1.0


class FlowMapIntegrator:
    """Advanced flow map integrator with RK4 and Hessian tracking."""

    def __init__(self, grid: Grid, kernel_order: str = 'quadratic',
                 track_hessian: bool = True):
        """Initialize integrator.

        Args:
            grid: Computational grid
            kernel_order: 'linear', 'quadratic', or 'cubic'
            track_hessian: Whether to track Hessian for gradient accuracy
        """
        self.grid = grid
        self.kernel = InterpolationKernel(kernel_order)
        self.track_hessian = track_hessian

        # Preallocate arrays for velocity gradient
        self._grad_v = None
        self._n_particles = 0

    def initialize_flow_map(self, n_particles: int) -> FlowMapState:
        """Initialize flow map state for particles."""
        J = np.zeros((n_particles, 2, 2))
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0

        H = None
        if self.track_hessian:
            H = np.zeros((n_particles, 2, 2, 2))

        # Preallocate gradient buffer
        self._n_particles = n_particles
        self._grad_v = np.zeros((n_particles, 2, 2))

        return FlowMapState(J=J, H=H)

    def _interpolate_velocity(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate velocity to particle positions."""
        vx = G2P_bspline(self.grid.vx, px, py,
                         self.grid.dx, self.grid.dy, self.kernel)
        vy = G2P_bspline(self.grid.vy, px, py,
                         self.grid.dx, self.grid.dy, self.kernel)
        return vx, vy

    def _interpolate_velocity_gradient(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Interpolate velocity gradient tensor to particles.

        Returns:
            grad_v (n_particles, 2, 2): Velocity gradient tensor
        """
        n = len(px)
        if self._grad_v is None or len(self._grad_v) != n:
            self._grad_v = np.zeros((n, 2, 2))

        self._grad_v[:, 0, 0] = G2P_bspline(self.grid.dvx_dx, px, py,
                                             self.grid.dx, self.grid.dy, self.kernel)
        self._grad_v[:, 0, 1] = G2P_bspline(self.grid.dvx_dy, px, py,
                                             self.grid.dx, self.grid.dy, self.kernel)
        self._grad_v[:, 1, 0] = G2P_bspline(self.grid.dvy_dx, px, py,
                                             self.grid.dx, self.grid.dy, self.kernel)
        self._grad_v[:, 1, 1] = G2P_bspline(self.grid.dvy_dy, px, py,
                                             self.grid.dx, self.grid.dy, self.kernel)
        return self._grad_v

    def advect_particles_rk4(self, particles: ParticleSystem, dt: float):
        """Advect particles using RK4."""
        x0 = np.ascontiguousarray(particles.x)
        y0 = np.ascontiguousarray(particles.y)
        Lx, Ly = self.grid.Lx, self.grid.Ly

        # k1
        vx1, vy1 = self._interpolate_velocity(x0, y0)

        # k2
        x_mid, y_mid = _midpoint_positions(x0, y0, vx1, vy1, 0.5*dt, Lx, Ly)
        vx2, vy2 = self._interpolate_velocity(x_mid, y_mid)

        # k3
        x_mid, y_mid = _midpoint_positions(x0, y0, vx2, vy2, 0.5*dt, Lx, Ly)
        vx3, vy3 = self._interpolate_velocity(x_mid, y_mid)

        # k4
        x_mid, y_mid = _midpoint_positions(x0, y0, vx3, vy3, dt, Lx, Ly)
        vx4, vy4 = self._interpolate_velocity(x_mid, y_mid)

        # Final update
        particles.x, particles.y = _rk4_positions(
            x0, y0, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4, dt, Lx, Ly)

    def evolve_jacobian_rk4(self, particles: ParticleSystem,
                            flow_map: FlowMapState, dt: float):
        """Evolve Jacobian using RK4.

        dJ/dt = -J · ∇v
        """
        J = flow_map.J
        Lx, Ly = self.grid.Lx, self.grid.Ly

        x0 = np.ascontiguousarray(particles.x)
        y0 = np.ascontiguousarray(particles.y)

        # Get velocity for position updates
        vx1, vy1 = self._interpolate_velocity(x0, y0)

        # k1 for J
        grad_v1 = self._interpolate_velocity_gradient(x0, y0)
        k1 = _jacobian_rhs_parallel(J, grad_v1)

        # k2 for J (at midpoint position and time)
        x_mid, y_mid = _midpoint_positions(x0, y0, vx1, vy1, 0.5*dt, Lx, Ly)
        vx2, vy2 = self._interpolate_velocity(x_mid, y_mid)

        grad_v2 = self._interpolate_velocity_gradient(x_mid, y_mid)
        J_mid = J + 0.5 * dt * k1
        k2 = _jacobian_rhs_parallel(J_mid, grad_v2)

        # k3
        x_mid2, y_mid2 = _midpoint_positions(x0, y0, vx2, vy2, 0.5*dt, Lx, Ly)

        grad_v3 = self._interpolate_velocity_gradient(x_mid2, y_mid2)
        J_mid = J + 0.5 * dt * k2
        k3 = _jacobian_rhs_parallel(J_mid, grad_v3)

        # k4
        vx3, vy3 = self._interpolate_velocity(x_mid2, y_mid2)
        x_end, y_end = _midpoint_positions(x0, y0, vx3, vy3, dt, Lx, Ly)

        grad_v4 = self._interpolate_velocity_gradient(x_end, y_end)
        J_mid = J + dt * k3
        k4 = _jacobian_rhs_parallel(J_mid, grad_v4)

        # Update Jacobian
        flow_map.J = _rk4_jacobian_update(J, k1, k2, k3, k4, dt)

    def evolve_hessian(self, particles: ParticleSystem,
                       flow_map: FlowMapState, dt: float):
        """Evolve Hessian using Euler (for simplicity).

        dH_ijk/dt = -H_ljk (∂v_i/∂x_l) - J_lj (∂²v_i/∂x_l∂x_k) - J_lk (∂²v_i/∂x_l∂x_j)
        """
        if flow_map.H is None:
            return

        n = particles.n_particles
        J = flow_map.J
        H = flow_map.H

        # Get velocity gradient
        grad_v = self._interpolate_velocity_gradient(particles.x, particles.y)

        # Get velocity Hessian on grid and interpolate
        hess_v = self._compute_velocity_hessian()

        d2v = np.zeros((n, 2, 2, 2))  # d2v[p, i, j, k] = ∂²v_i/∂x_j∂x_k

        d2v[:, 0, 0, 0] = G2P_bspline(hess_v['d2vx_dxdx'], particles.x, particles.y,
                                       self.grid.dx, self.grid.dy, self.kernel)
        d2v[:, 0, 0, 1] = G2P_bspline(hess_v['d2vx_dxdy'], particles.x, particles.y,
                                       self.grid.dx, self.grid.dy, self.kernel)
        d2v[:, 0, 1, 0] = d2v[:, 0, 0, 1]  # Symmetry
        d2v[:, 0, 1, 1] = G2P_bspline(hess_v['d2vx_dydy'], particles.x, particles.y,
                                       self.grid.dx, self.grid.dy, self.kernel)

        d2v[:, 1, 0, 0] = G2P_bspline(hess_v['d2vy_dxdx'], particles.x, particles.y,
                                       self.grid.dx, self.grid.dy, self.kernel)
        d2v[:, 1, 0, 1] = G2P_bspline(hess_v['d2vy_dxdy'], particles.x, particles.y,
                                       self.grid.dx, self.grid.dy, self.kernel)
        d2v[:, 1, 1, 0] = d2v[:, 1, 0, 1]
        d2v[:, 1, 1, 1] = G2P_bspline(hess_v['d2vy_dydy'], particles.x, particles.y,
                                       self.grid.dx, self.grid.dy, self.kernel)

        # Compute dH/dt (simplified Euler update)
        dH = self._compute_hessian_rhs(H, J, grad_v, d2v)
        flow_map.H = H + dt * dH

    def _compute_velocity_hessian(self) -> dict:
        """Compute second derivatives of velocity on grid."""
        from numpy.fft import fft2, ifft2, fftfreq

        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy

        kx = fftfreq(nx, dx) * 2 * np.pi
        ky = fftfreq(ny, dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        vx_hat = fft2(self.grid.vx)
        vy_hat = fft2(self.grid.vy)

        # Second derivatives
        d2vx_dxdx = np.real(ifft2(-KX**2 * vx_hat))
        d2vx_dxdy = np.real(ifft2(-KX * KY * vx_hat))
        d2vx_dydy = np.real(ifft2(-KY**2 * vx_hat))

        d2vy_dxdx = np.real(ifft2(-KX**2 * vy_hat))
        d2vy_dxdy = np.real(ifft2(-KX * KY * vy_hat))
        d2vy_dydy = np.real(ifft2(-KY**2 * vy_hat))

        return {
            'd2vx_dxdx': d2vx_dxdx, 'd2vx_dxdy': d2vx_dxdy, 'd2vx_dydy': d2vx_dydy,
            'd2vy_dxdx': d2vy_dxdx, 'd2vy_dxdy': d2vy_dxdy, 'd2vy_dydy': d2vy_dydy,
        }

    @staticmethod
    def _compute_hessian_rhs(H, J, grad_v, d2v):
        """Compute dH/dt."""
        n = H.shape[0]
        dH = np.zeros_like(H)

        for p in range(n):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # Term 1: -H_ljk * ∂v_i/∂x_l
                        for l in range(2):
                            dH[p, i, j, k] -= H[p, l, j, k] * grad_v[p, i, l]

                        # Term 2: -J_lj * ∂²v_i/∂x_l∂x_k
                        for l in range(2):
                            dH[p, i, j, k] -= J[p, l, j] * d2v[p, i, l, k]

                        # Term 3: -J_lk * ∂²v_i/∂x_l∂x_j
                        for l in range(2):
                            dH[p, i, j, k] -= J[p, l, k] * d2v[p, i, l, j]

        return dH

    def step(self, particles: ParticleSystem, flow_map: FlowMapState, dt: float):
        """Complete flow map integration step."""
        # Evolve Jacobian first (needs current positions)
        self.evolve_jacobian_rk4(particles, flow_map, dt)

        # Evolve Hessian if tracking
        if self.track_hessian and flow_map.H is not None:
            self.evolve_hessian(particles, flow_map, dt)

        # Advect particles
        self.advect_particles_rk4(particles, dt)

        flow_map.steps_since_reinit += 1

    def estimate_error(self, flow_map: FlowMapState) -> float:
        """Estimate flow map error for reinitialization decision."""
        J_error = _estimate_jacobian_error(flow_map.J)

        # Hessian norm (if available)
        H_error = 0.0
        if flow_map.H is not None:
            H_error = np.max(np.sqrt(np.sum(flow_map.H**2, axis=(1, 2, 3))))

        return J_error + 0.1 * H_error

    def should_reinitialize(self, flow_map: FlowMapState,
                            threshold: float = 0.5,
                            max_steps: int = 50) -> bool:
        """Check if flow map should be reinitialized."""
        if flow_map.steps_since_reinit >= max_steps:
            return True

        error = self.estimate_error(flow_map)
        return error > threshold

    def reinitialize(self, particles: ParticleSystem, flow_map: FlowMapState,
                     grid_field: np.ndarray) -> np.ndarray:
        """Reinitialize flow map and update particle values."""
        # Reset Jacobian to identity
        _reset_jacobian(flow_map.J)

        # Reset Hessian to zero
        if flow_map.H is not None:
            flow_map.H.fill(0.0)

        flow_map.steps_since_reinit = 0
        flow_map.cumulative_error = 0.0

        # Interpolate current grid field to particles
        new_values = G2P_bspline(grid_field, particles.x, particles.y,
                                  self.grid.dx, self.grid.dy, self.kernel)

        return new_values
