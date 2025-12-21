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
    # Jacobian J = ∇ψ (backward flow map)
    J: np.ndarray  # (n_particles, 2, 2)

    # Hessian H = ∇²F (for gradient accuracy)
    # H[p, i, j, k] = ∂²F_i / ∂x_j ∂x_k
    H: Optional[np.ndarray] = None  # (n_particles, 2, 2, 2)

    # Tracking
    steps_since_reinit: int = 0
    cumulative_error: float = 0.0


@dataclass
class DualScaleFlowMapState:
    """Dual-scale flow map state for improved accuracy.

    Maintains two timescales:
    - Long flow map (n_L): For vorticity values, reinitialized less frequently
    - Short flow map (n_S): For vorticity gradients, reinitialized more frequently

    This allows longer stable flow maps while maintaining gradient accuracy.
    At short reinitialization, Jacobians are composed: J_long = J_short · J_long

    Reference: Wang et al. (2025) Section 4.3.3
    """
    # Long flow map for vorticity (reinitialized less frequently)
    J_long: np.ndarray  # (n_particles, 2, 2)
    steps_since_long_reinit: int = 0

    # Short flow map for gradients (reinitialized more frequently)
    J_short: np.ndarray = None  # (n_particles, 2, 2)
    H_short: Optional[np.ndarray] = None  # (n_particles, 2, 2, 2)
    steps_since_short_reinit: int = 0

    # Cumulative Jacobian from long to current: J_total = J_short · J_long
    # Used for pulling back vorticity from initial to current configuration

    def __post_init__(self):
        """Initialize short arrays if not provided."""
        if self.J_short is None:
            n = self.J_long.shape[0]
            self.J_short = np.zeros((n, 2, 2))
            self.J_short[:, 0, 0] = 1.0
            self.J_short[:, 1, 1] = 1.0


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


@njit(cache=True, fastmath=True)
def _estimate_jacobian_error(J: np.ndarray) -> float:
    """Compute max ||J - I|| across all particles."""
    n = J.shape[0]
    max_err = 0.0

    for p in range(n):
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


@njit(parallel=True, cache=True, fastmath=True)
def _compose_jacobians(J_short: np.ndarray, J_long: np.ndarray) -> np.ndarray:
    """Compose Jacobians: J_composed = J_short · J_long.

    This implements the flow map composition F^[a,c] = F^[b,c] ∘ F^[a,b]
    which gives J^[a,c] = J^[b,c] · J^[a,b]

    Args:
        J_short: Short-term Jacobian J^[b,c] (n, 2, 2)
        J_long: Long-term Jacobian J^[a,b] (n, 2, 2)

    Returns:
        Composed Jacobian J^[a,c] = J_short · J_long (n, 2, 2)
    """
    n = J_short.shape[0]
    J_composed = np.empty_like(J_short)

    for p in prange(n):
        # 2x2 matrix multiplication (unrolled for speed)
        J_composed[p, 0, 0] = J_short[p, 0, 0] * J_long[p, 0, 0] + J_short[p, 0, 1] * J_long[p, 1, 0]
        J_composed[p, 0, 1] = J_short[p, 0, 0] * J_long[p, 0, 1] + J_short[p, 0, 1] * J_long[p, 1, 1]
        J_composed[p, 1, 0] = J_short[p, 1, 0] * J_long[p, 0, 0] + J_short[p, 1, 1] * J_long[p, 1, 0]
        J_composed[p, 1, 1] = J_short[p, 1, 0] * J_long[p, 0, 1] + J_short[p, 1, 1] * J_long[p, 1, 1]

    return J_composed


@njit(parallel=True, cache=True, fastmath=True)
def _copy_jacobian(J_src: np.ndarray, J_dst: np.ndarray):
    """Copy Jacobian array in-place."""
    n = J_src.shape[0]
    for p in prange(n):
        J_dst[p, 0, 0] = J_src[p, 0, 0]
        J_dst[p, 0, 1] = J_src[p, 0, 1]
        J_dst[p, 1, 0] = J_src[p, 1, 0]
        J_dst[p, 1, 1] = J_src[p, 1, 1]


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

        dH_ijk/dt = -H_ijl (∂v_l/∂x_k) - H_ilk (∂v_l/∂x_j)
                   - J_il (∂²v_l/∂x_j∂x_k)
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

    def evolve_particle_gradients_rk4(self, particles: ParticleSystem, dt: float):
        """Evolve particle vorticity gradients for a materially conserved scalar.

        Uses: D(grad q)/Dt = -(grad v)^T grad q
        """
        if not hasattr(particles, "grad_q_x"):
            return

        g0 = np.stack((particles.grad_q_x, particles.grad_q_y), axis=1)
        x0 = np.ascontiguousarray(particles.x)
        y0 = np.ascontiguousarray(particles.y)
        Lx, Ly = self.grid.Lx, self.grid.Ly

        # k1
        vx1, vy1 = self._interpolate_velocity(x0, y0)
        grad_v1 = self._interpolate_velocity_gradient(x0, y0)
        k1 = -np.einsum('pji,pj->pi', grad_v1, g0)

        # k2
        x_mid, y_mid = _midpoint_positions(x0, y0, vx1, vy1, 0.5 * dt, Lx, Ly)
        g_mid = g0 + 0.5 * dt * k1
        vx2, vy2 = self._interpolate_velocity(x_mid, y_mid)
        grad_v2 = self._interpolate_velocity_gradient(x_mid, y_mid)
        k2 = -np.einsum('pji,pj->pi', grad_v2, g_mid)

        # k3
        x_mid2, y_mid2 = _midpoint_positions(x0, y0, vx2, vy2, 0.5 * dt, Lx, Ly)
        g_mid2 = g0 + 0.5 * dt * k2
        vx3, vy3 = self._interpolate_velocity(x_mid2, y_mid2)
        grad_v3 = self._interpolate_velocity_gradient(x_mid2, y_mid2)
        k3 = -np.einsum('pji,pj->pi', grad_v3, g_mid2)

        # k4
        x_end, y_end = _midpoint_positions(x0, y0, vx3, vy3, dt, Lx, Ly)
        g_end = g0 + dt * k3
        grad_v4 = self._interpolate_velocity_gradient(x_end, y_end)
        k4 = -np.einsum('pji,pj->pi', grad_v4, g_end)

        g_new = g0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        particles.grad_q_x = g_new[:, 0]
        particles.grad_q_y = g_new[:, 1]

    @staticmethod
    def _compute_hessian_rhs(H, J, grad_v, d2v):
        """Compute dH/dt for the backward flow map Hessian."""
        n = H.shape[0]
        dH = np.zeros_like(H)

        for p in range(n):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        term = 0.0
                        for l in range(2):
                            term += H[p, i, j, l] * grad_v[p, l, k]
                            term += H[p, i, l, k] * grad_v[p, l, j]
                            term += J[p, i, l] * d2v[p, l, j, k]
                        dH[p, i, j, k] = -term

        return dH

    def step(self, particles: ParticleSystem, flow_map: FlowMapState, dt: float,
             update_gradients: bool = False):
        """Complete flow map integration step."""
        # Evolve Jacobian first (needs current positions)
        self.evolve_jacobian_rk4(particles, flow_map, dt)

        # Evolve Hessian if tracking
        if self.track_hessian and flow_map.H is not None:
            self.evolve_hessian(particles, flow_map, dt)

        # Evolve particle gradients if requested
        if update_gradients:
            self.evolve_particle_gradients_rk4(particles, dt)

        # Advect particles
        self.advect_particles_rk4(particles, dt)

        flow_map.steps_since_reinit += 1

    def estimate_error(self, flow_map: FlowMapState) -> float:
        """Estimate flow map error for reinitialization decision."""
        J_error = _estimate_jacobian_error(flow_map.J)

        # Hessian norm (if available)
        H_error = 0.0
        if flow_map.H is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                H_error = np.max(np.sqrt(np.sum(flow_map.H**2, axis=(1, 2, 3))))
            if not np.isfinite(H_error):
                H_error = np.nanmax(np.abs(flow_map.H))

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


class DualScaleFlowMapIntegrator:
    """Dual-scale flow map integrator for improved accuracy.

    Maintains two timescales:
    - Long flow map (n_L): For vorticity values, reinitialized less frequently (every n_L steps)
    - Short flow map (n_S): For vorticity gradients, reinitialized more frequently (every n_S steps)

    When the short map is reinitialized, the Jacobians are composed:
        J_long_new = J_short · J_long_old

    This allows longer stable flow maps (better vorticity preservation) while
    maintaining gradient accuracy through more frequent short-map reinitialization.

    Typical settings: n_L = 100-200, n_S = 20-50 (ratio 3-12x)

    Reference: Wang et al. (2025) Section 4.3.3
    """

    def __init__(self, grid: Grid, kernel_order: str = 'quadratic',
                 track_hessian: bool = True,
                 n_L: int = 100,  # Long map reinit interval
                 n_S: int = 20,   # Short map reinit interval
                 error_threshold_long: float = 3.0,
                 error_threshold_short: float = 1.0):
        """Initialize dual-scale integrator.

        Args:
            grid: Computational grid
            kernel_order: 'linear', 'quadratic', or 'cubic'
            track_hessian: Whether to track Hessian for gradient accuracy
            n_L: Maximum steps between long map reinitializations
            n_S: Maximum steps between short map reinitializations
            error_threshold_long: Error threshold for long map reinit
            error_threshold_short: Error threshold for short map reinit
        """
        self.grid = grid
        self.kernel = InterpolationKernel(kernel_order)
        self.track_hessian = track_hessian

        self.n_L = n_L
        self.n_S = n_S
        self.error_threshold_long = error_threshold_long
        self.error_threshold_short = error_threshold_short

        # Internal single-scale integrator for actual evolution
        self._integrator = FlowMapIntegrator(grid, kernel_order, track_hessian)

        # Preallocate arrays
        self._grad_v = None
        self._n_particles = 0

    def initialize_flow_map(self, n_particles: int) -> DualScaleFlowMapState:
        """Initialize dual-scale flow map state for particles."""
        # Long Jacobian
        J_long = np.zeros((n_particles, 2, 2))
        J_long[:, 0, 0] = 1.0
        J_long[:, 1, 1] = 1.0

        # Short Jacobian
        J_short = np.zeros((n_particles, 2, 2))
        J_short[:, 0, 0] = 1.0
        J_short[:, 1, 1] = 1.0

        # Hessian (only for short map)
        H_short = None
        if self.track_hessian:
            H_short = np.zeros((n_particles, 2, 2, 2))

        self._n_particles = n_particles

        return DualScaleFlowMapState(
            J_long=J_long,
            J_short=J_short,
            H_short=H_short,
            steps_since_long_reinit=0,
            steps_since_short_reinit=0,
        )

    def step(self, particles: ParticleSystem, flow_map: DualScaleFlowMapState, dt: float):
        """Advance dual-scale flow map by one timestep.

        Evolves the short Jacobian (and Hessian) using RK4, then advects particles.
        """
        # Create temporary single-scale state for evolution
        temp_state = FlowMapState(
            J=flow_map.J_short,
            H=flow_map.H_short,
            steps_since_reinit=flow_map.steps_since_short_reinit
        )

        # Evolve using single-scale integrator
        self._integrator.evolve_jacobian_rk4(particles, temp_state, dt)

        # Copy the updated Jacobian back (evolve_jacobian_rk4 creates new array)
        np.copyto(flow_map.J_short, temp_state.J)

        if self.track_hessian and flow_map.H_short is not None:
            temp_state.H = flow_map.H_short
            self._integrator.evolve_hessian(particles, temp_state, dt)
            # Hessian is updated in-place, but copy back just in case
            np.copyto(flow_map.H_short, temp_state.H)

        # Advect particles
        self._integrator.advect_particles_rk4(particles, dt)

        # Update step counters
        flow_map.steps_since_short_reinit += 1
        flow_map.steps_since_long_reinit += 1

    def estimate_short_error(self, flow_map: DualScaleFlowMapState) -> float:
        """Estimate short flow map error."""
        J_error = _estimate_jacobian_error(flow_map.J_short)

        H_error = 0.0
        if flow_map.H_short is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                H_error = np.max(np.sqrt(np.sum(flow_map.H_short**2, axis=(1, 2, 3))))
            if not np.isfinite(H_error):
                H_error = np.nanmax(np.abs(flow_map.H_short))

        return J_error + 0.1 * H_error

    def estimate_long_error(self, flow_map: DualScaleFlowMapState) -> float:
        """Estimate cumulative (long) flow map error.

        Uses the composed Jacobian J_total = J_short · J_long
        """
        J_total = _compose_jacobians(flow_map.J_short, flow_map.J_long)
        return _estimate_jacobian_error(J_total)

    def should_reinit_short(self, flow_map: DualScaleFlowMapState) -> bool:
        """Check if short flow map should be reinitialized."""
        if flow_map.steps_since_short_reinit >= self.n_S:
            return True
        return self.estimate_short_error(flow_map) > self.error_threshold_short

    def should_reinit_long(self, flow_map: DualScaleFlowMapState) -> bool:
        """Check if long flow map should be reinitialized."""
        if flow_map.steps_since_long_reinit >= self.n_L:
            return True
        return self.estimate_long_error(flow_map) > self.error_threshold_long

    def reinit_short(self, particles: ParticleSystem, flow_map: DualScaleFlowMapState,
                     grid_field: np.ndarray):
        """Reinitialize short flow map.

        Composes the short Jacobian into the long Jacobian before resetting:
            J_long_new = J_short · J_long_old

        Then resets J_short to identity and updates particle gradients from grid.
        """
        # Compose Jacobians: J_long = J_short · J_long
        J_composed = _compose_jacobians(flow_map.J_short, flow_map.J_long)
        _copy_jacobian(J_composed, flow_map.J_long)

        # Reset short Jacobian to identity
        _reset_jacobian(flow_map.J_short)

        # Reset Hessian to zero
        if flow_map.H_short is not None:
            flow_map.H_short.fill(0.0)

        flow_map.steps_since_short_reinit = 0

        # Update particle gradients from grid (if gradient tracking enabled)
        # The vorticity values are NOT updated here - that happens at long reinit

    def reinit_long(self, particles: ParticleSystem, flow_map: DualScaleFlowMapState,
                    grid_field: np.ndarray) -> np.ndarray:
        """Reinitialize long flow map.

        This is a full reinitialization:
        1. Reset both J_long and J_short to identity
        2. Reset Hessian to zero
        3. Update particle vorticity from grid

        Returns:
            New particle vorticity values interpolated from grid
        """
        # Reset long Jacobian to identity
        _reset_jacobian(flow_map.J_long)

        # Reset short Jacobian to identity
        _reset_jacobian(flow_map.J_short)

        # Reset Hessian to zero
        if flow_map.H_short is not None:
            flow_map.H_short.fill(0.0)

        flow_map.steps_since_long_reinit = 0
        flow_map.steps_since_short_reinit = 0

        # Interpolate current grid field to particles
        new_values = G2P_bspline(grid_field, particles.x, particles.y,
                                  self.grid.dx, self.grid.dy, self.kernel)

        return new_values

    def get_total_jacobian(self, flow_map: DualScaleFlowMapState) -> np.ndarray:
        """Get the total (composed) Jacobian J_total = J_short · J_long."""
        return _compose_jacobians(flow_map.J_short, flow_map.J_long)
