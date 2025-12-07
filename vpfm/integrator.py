"""Time integration for VPFM-Plasma.

Implements RK4 integration for particle positions and Jacobian evolution.
"""

import numpy as np
from .grid import Grid
from .particles import ParticleSystem
from .transfers import G2P, G2P_velocity, G2P_velocity_gradient


class RK4Integrator:
    """Fourth-order Runge-Kutta integrator for particle advection.

    Handles:
    - Particle position advection
    - Jacobian matrix evolution
    - Periodic boundary conditions
    """

    def __init__(self, grid: Grid):
        """Initialize integrator.

        Args:
            grid: Grid for domain information and field interpolation
        """
        self.grid = grid

    def advect_particles(self, particles: ParticleSystem, dt: float):
        """Advance particle positions using RK4.

        Args:
            particles: Particle system to advect
            dt: Time step
        """
        # Store original positions
        x0 = particles.x.copy()
        y0 = particles.y.copy()

        # Stage 1: k1 = v(x0, t)
        vx1, vy1 = G2P_velocity(self.grid, particles)
        k1x, k1y = vx1, vy1

        # Stage 2: k2 = v(x0 + 0.5*dt*k1, t + 0.5*dt)
        particles.x = (x0 + 0.5 * dt * k1x) % self.grid.Lx
        particles.y = (y0 + 0.5 * dt * k1y) % self.grid.Ly
        vx2, vy2 = G2P_velocity(self.grid, particles)
        k2x, k2y = vx2, vy2

        # Stage 3: k3 = v(x0 + 0.5*dt*k2, t + 0.5*dt)
        particles.x = (x0 + 0.5 * dt * k2x) % self.grid.Lx
        particles.y = (y0 + 0.5 * dt * k2y) % self.grid.Ly
        vx3, vy3 = G2P_velocity(self.grid, particles)
        k3x, k3y = vx3, vy3

        # Stage 4: k4 = v(x0 + dt*k3, t + dt)
        particles.x = (x0 + dt * k3x) % self.grid.Lx
        particles.y = (y0 + dt * k3y) % self.grid.Ly
        vx4, vy4 = G2P_velocity(self.grid, particles)
        k4x, k4y = vx4, vy4

        # Final position update
        particles.x = (x0 + (dt / 6) * (k1x + 2*k2x + 2*k3x + k4x)) % self.grid.Lx
        particles.y = (y0 + (dt / 6) * (k1y + 2*k2y + 2*k3y + k4y)) % self.grid.Ly

    def evolve_jacobian(self, particles: ParticleSystem, dt: float):
        """Evolve Jacobian matrices using RK4.

        The Jacobian evolves according to: dJ/dt = -J · nabla(v)

        Args:
            particles: Particle system with Jacobians
            dt: Time step
        """
        n = particles.n_particles

        # Store original Jacobians
        J0 = particles.J.copy()

        # Get velocity gradient at particle positions
        dvx_dx, dvx_dy, dvy_dx, dvy_dy = G2P_velocity_gradient(self.grid, particles)

        # Build gradient tensor for all particles: grad_v[p] = [[dvx/dx, dvx/dy], [dvy/dx, dvy/dy]]
        grad_v = np.zeros((n, 2, 2))
        grad_v[:, 0, 0] = dvx_dx
        grad_v[:, 0, 1] = dvx_dy
        grad_v[:, 1, 0] = dvy_dx
        grad_v[:, 1, 1] = dvy_dy

        # dJ/dt = -J @ grad_v
        # Simple Euler for POC (RK4 would require velocity gradient at intermediate times)
        dJ_dt = -np.einsum('pij,pjk->pik', particles.J, grad_v)

        particles.J = J0 + dt * dJ_dt

    def evolve_jacobian_euler(self, particles: ParticleSystem, dt: float):
        """Evolve Jacobian matrices using simple Euler method.

        For POC simplicity - less accurate but simpler.

        Args:
            particles: Particle system with Jacobians
            dt: Time step
        """
        n = particles.n_particles

        # Get velocity gradient at particle positions
        dvx_dx, dvx_dy, dvy_dx, dvy_dy = G2P_velocity_gradient(self.grid, particles)

        # Build gradient tensor
        grad_v = np.zeros((n, 2, 2))
        grad_v[:, 0, 0] = dvx_dx
        grad_v[:, 0, 1] = dvx_dy
        grad_v[:, 1, 0] = dvy_dx
        grad_v[:, 1, 1] = dvy_dy

        # dJ/dt = -J @ grad_v
        dJ_dt = -np.einsum('pij,pjk->pik', particles.J, grad_v)

        particles.J += dt * dJ_dt

    def step(self, particles: ParticleSystem, dt: float):
        """Perform one complete integration step.

        Advects particles and evolves Jacobians.

        Args:
            particles: Particle system to update
            dt: Time step
        """
        self.advect_particles(particles, dt)
        self.evolve_jacobian_euler(particles, dt)


def compute_cfl_dt(grid: Grid, cfl: float = 0.5) -> float:
    """Compute time step based on CFL condition.

    Args:
        grid: Grid with velocity fields
        cfl: CFL number (typically 0.5)

    Returns:
        Maximum stable time step
    """
    from .velocity import max_velocity

    v_max = max_velocity(grid)
    if v_max < 1e-10:
        return 1.0  # Default for zero velocity

    dx_min = min(grid.dx, grid.dy)
    dt_cfl = cfl * dx_min / v_max

    return dt_cfl
