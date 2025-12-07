"""Advanced flow map evolution for VPFM-Plasma.

Implements:
- RK4 Jacobian evolution (not just Euler)
- Hessian tracking for gradient accuracy
- Adaptive reinitialization based on error estimates
- Jacobian composition for long flow maps

Reference: Wang et al. (2025) "Fluid Simulation on Vortex Particle Flow Maps"
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .grid import Grid
from .particles import ParticleSystem
from .transfers import G2P
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

    def initialize_flow_map(self, n_particles: int) -> FlowMapState:
        """Initialize flow map state for particles.

        Args:
            n_particles: Number of particles

        Returns:
            Initial FlowMapState with J=I, H=0
        """
        J = np.zeros((n_particles, 2, 2))
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0

        H = None
        if self.track_hessian:
            H = np.zeros((n_particles, 2, 2, 2))

        return FlowMapState(J=J, H=H)

    def _interpolate_velocity(self, particles: ParticleSystem) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate velocity to particle positions."""
        vx = G2P_bspline(self.grid.vx, particles.x, particles.y,
                         self.grid.dx, self.grid.dy, self.kernel)
        vy = G2P_bspline(self.grid.vy, particles.x, particles.y,
                         self.grid.dx, self.grid.dy, self.kernel)
        return vx, vy

    def _interpolate_velocity_gradient(self, particles: ParticleSystem) -> Tuple:
        """Interpolate velocity gradient tensor to particles.

        Returns:
            (dvx_dx, dvx_dy, dvy_dx, dvy_dy)
        """
        dvx_dx = G2P_bspline(self.grid.dvx_dx, particles.x, particles.y,
                             self.grid.dx, self.grid.dy, self.kernel)
        dvx_dy = G2P_bspline(self.grid.dvx_dy, particles.x, particles.y,
                             self.grid.dx, self.grid.dy, self.kernel)
        dvy_dx = G2P_bspline(self.grid.dvy_dx, particles.x, particles.y,
                             self.grid.dx, self.grid.dy, self.kernel)
        dvy_dy = G2P_bspline(self.grid.dvy_dy, particles.x, particles.y,
                             self.grid.dx, self.grid.dy, self.kernel)
        return dvx_dx, dvx_dy, dvy_dx, dvy_dy

    def _compute_velocity_hessian(self) -> dict:
        """Compute second derivatives of velocity on grid.

        Returns dict with d2vx_dxdx, d2vx_dxdy, d2vx_dydy, etc.
        """
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

    def _jacobian_rhs(self, J: np.ndarray, grad_v: np.ndarray) -> np.ndarray:
        """Compute dJ/dt = -J · ∇v.

        Args:
            J: Jacobian matrices (n, 2, 2)
            grad_v: Velocity gradient (n, 2, 2)

        Returns:
            dJ/dt (n, 2, 2)
        """
        return -np.einsum('pij,pjk->pik', J, grad_v)

    def advect_particles_rk4(self, particles: ParticleSystem, dt: float):
        """Advect particles using RK4.

        Args:
            particles: Particle system
            dt: Time step
        """
        x0, y0 = particles.x.copy(), particles.y.copy()
        Lx, Ly = self.grid.Lx, self.grid.Ly

        # k1
        vx1, vy1 = self._interpolate_velocity(particles)

        # k2
        particles.x = (x0 + 0.5 * dt * vx1) % Lx
        particles.y = (y0 + 0.5 * dt * vy1) % Ly
        vx2, vy2 = self._interpolate_velocity(particles)

        # k3
        particles.x = (x0 + 0.5 * dt * vx2) % Lx
        particles.y = (y0 + 0.5 * dt * vy2) % Ly
        vx3, vy3 = self._interpolate_velocity(particles)

        # k4
        particles.x = (x0 + dt * vx3) % Lx
        particles.y = (y0 + dt * vy3) % Ly
        vx4, vy4 = self._interpolate_velocity(particles)

        # Final update
        particles.x = (x0 + (dt / 6) * (vx1 + 2*vx2 + 2*vx3 + vx4)) % Lx
        particles.y = (y0 + (dt / 6) * (vy1 + 2*vy2 + 2*vy3 + vy4)) % Ly

    def evolve_jacobian_rk4(self, particles: ParticleSystem,
                            flow_map: FlowMapState, dt: float):
        """Evolve Jacobian using RK4.

        dJ/dt = -J · ∇v

        Args:
            particles: Particle system (for positions)
            flow_map: Flow map state to update
            dt: Time step
        """
        n = particles.n_particles
        J = flow_map.J
        Lx, Ly = self.grid.Lx, self.grid.Ly

        # Save original positions
        x0, y0 = particles.x.copy(), particles.y.copy()

        def get_grad_v(px, py):
            """Get velocity gradient at positions."""
            # Temporarily update particle positions
            particles.x = px % Lx
            particles.y = py % Ly
            dvx_dx, dvx_dy, dvy_dx, dvy_dy = self._interpolate_velocity_gradient(particles)

            grad_v = np.zeros((n, 2, 2))
            grad_v[:, 0, 0] = dvx_dx
            grad_v[:, 0, 1] = dvx_dy
            grad_v[:, 1, 0] = dvy_dx
            grad_v[:, 1, 1] = dvy_dy
            return grad_v

        # Get velocity for position updates
        particles.x, particles.y = x0, y0
        vx1, vy1 = self._interpolate_velocity(particles)

        # k1 for J
        grad_v1 = get_grad_v(x0, y0)
        k1 = self._jacobian_rhs(J, grad_v1)

        # k2 for J (at midpoint position and time)
        x_mid = (x0 + 0.5 * dt * vx1) % Lx
        y_mid = (y0 + 0.5 * dt * vy1) % Ly
        particles.x, particles.y = x_mid, y_mid
        vx2, vy2 = self._interpolate_velocity(particles)

        grad_v2 = get_grad_v(x_mid, y_mid)
        k2 = self._jacobian_rhs(J + 0.5 * dt * k1, grad_v2)

        # k3
        x_mid2 = (x0 + 0.5 * dt * vx2) % Lx
        y_mid2 = (y0 + 0.5 * dt * vy2) % Ly

        grad_v3 = get_grad_v(x_mid2, y_mid2)
        k3 = self._jacobian_rhs(J + 0.5 * dt * k2, grad_v3)

        # k4
        particles.x, particles.y = x_mid2, y_mid2
        vx3, vy3 = self._interpolate_velocity(particles)
        x_end = (x0 + dt * vx3) % Lx
        y_end = (y0 + dt * vy3) % Ly

        grad_v4 = get_grad_v(x_end, y_end)
        k4 = self._jacobian_rhs(J + dt * k3, grad_v4)

        # Update Jacobian
        flow_map.J = J + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Restore original positions (will be updated by advect_particles)
        particles.x, particles.y = x0, y0

    def evolve_hessian(self, particles: ParticleSystem,
                       flow_map: FlowMapState, dt: float):
        """Evolve Hessian using Euler (for simplicity).

        dH_ijk/dt = -H_ljk (∂v_i/∂x_l) - J_lj (∂²v_i/∂x_l∂x_k) - J_lk (∂²v_i/∂x_l∂x_j)

        Args:
            particles: Particle system
            flow_map: Flow map state
            dt: Time step
        """
        if flow_map.H is None:
            return

        n = particles.n_particles
        J = flow_map.J
        H = flow_map.H

        # Get velocity gradient
        dvx_dx, dvx_dy, dvy_dx, dvy_dy = self._interpolate_velocity_gradient(particles)

        grad_v = np.zeros((n, 2, 2))
        grad_v[:, 0, 0] = dvx_dx
        grad_v[:, 0, 1] = dvx_dy
        grad_v[:, 1, 0] = dvy_dx
        grad_v[:, 1, 1] = dvy_dy

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

        # Compute dH/dt
        # dH_ijk/dt = -H_ljk (∂v_i/∂x_l) - J_lj (∂²v_i/∂x_l∂x_k) - J_lk (∂²v_i/∂x_l∂x_j)
        dH = np.zeros_like(H)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Term 1: -H_ljk * ∂v_i/∂x_l
                    for l in range(2):
                        dH[:, i, j, k] -= H[:, l, j, k] * grad_v[:, i, l]

                    # Term 2: -J_lj * ∂²v_i/∂x_l∂x_k
                    for l in range(2):
                        dH[:, i, j, k] -= J[:, l, j] * d2v[:, i, l, k]

                    # Term 3: -J_lk * ∂²v_i/∂x_l∂x_j
                    for l in range(2):
                        dH[:, i, j, k] -= J[:, l, k] * d2v[:, i, l, j]

        flow_map.H = H + dt * dH

    def step(self, particles: ParticleSystem, flow_map: FlowMapState, dt: float):
        """Complete flow map integration step.

        Args:
            particles: Particle system
            flow_map: Flow map state
            dt: Time step
        """
        # Evolve Jacobian first (needs current positions)
        self.evolve_jacobian_rk4(particles, flow_map, dt)

        # Evolve Hessian if tracking
        if self.track_hessian and flow_map.H is not None:
            self.evolve_hessian(particles, flow_map, dt)

        # Advect particles
        self.advect_particles_rk4(particles, dt)

        flow_map.steps_since_reinit += 1

    def estimate_error(self, flow_map: FlowMapState) -> float:
        """Estimate flow map error for reinitialization decision.

        Uses ||J - I|| as primary metric, with Hessian norm as secondary.

        Returns:
            Error estimate (larger = more reinit needed)
        """
        # Jacobian deviation from identity
        J_dev = flow_map.J.copy()
        J_dev[:, 0, 0] -= 1.0
        J_dev[:, 1, 1] -= 1.0
        J_error = np.max(np.sqrt(np.sum(J_dev**2, axis=(1, 2))))

        # Hessian norm (if available)
        H_error = 0.0
        if flow_map.H is not None:
            H_error = np.max(np.sqrt(np.sum(flow_map.H**2, axis=(1, 2, 3))))

        return J_error + 0.1 * H_error

    def should_reinitialize(self, flow_map: FlowMapState,
                            threshold: float = 0.5,
                            max_steps: int = 50) -> bool:
        """Check if flow map should be reinitialized.

        Args:
            flow_map: Current flow map state
            threshold: Error threshold for reinit
            max_steps: Maximum steps before forced reinit

        Returns:
            True if reinitialization recommended
        """
        if flow_map.steps_since_reinit >= max_steps:
            return True

        error = self.estimate_error(flow_map)
        return error > threshold

    def reinitialize(self, particles: ParticleSystem, flow_map: FlowMapState,
                     grid_field: np.ndarray) -> np.ndarray:
        """Reinitialize flow map and update particle values.

        Args:
            particles: Particle system
            flow_map: Flow map state to reset
            grid_field: Current field on grid

        Returns:
            Updated particle field values
        """
        # Reset Jacobian to identity
        flow_map.J.fill(0.0)
        flow_map.J[:, 0, 0] = 1.0
        flow_map.J[:, 1, 1] = 1.0

        # Reset Hessian to zero
        if flow_map.H is not None:
            flow_map.H.fill(0.0)

        flow_map.steps_since_reinit = 0
        flow_map.cumulative_error = 0.0

        # Interpolate current grid field to particles
        new_values = G2P_bspline(grid_field, particles.x, particles.y,
                                  self.grid.dx, self.grid.dy, self.kernel)

        return new_values
