"""Particle system for VPFM-Plasma."""

import numpy as np
from .grid import Grid


class ParticleSystem:
    """Lagrangian particle system storing vorticity and flow map quantities.

    Attributes:
        n_particles: Number of particles
        x, y: Particle positions (n_particles,)
        q: Potential vorticity carried by particles (n_particles,)
        J: Jacobian matrices for each particle (n_particles, 2, 2)
    """

    def __init__(self, n_particles: int):
        """Initialize particle system.

        Args:
            n_particles: Number of particles to allocate
        """
        self.n_particles = n_particles

        # Particle positions
        self.x = np.zeros(n_particles)
        self.y = np.zeros(n_particles)

        # Potential vorticity
        self.q = np.zeros(n_particles)

        # Vorticity gradient (for gradient-enhanced P2G)
        self.grad_q_x = np.zeros(n_particles)
        self.grad_q_y = np.zeros(n_particles)

        # Flow map Jacobian (2x2 matrix per particle)
        # J[p] = [[J_xx, J_xy], [J_yx, J_yy]]
        self.J = np.zeros((n_particles, 2, 2))

        # Initialize Jacobians to identity
        self.J[:, 0, 0] = 1.0
        self.J[:, 1, 1] = 1.0

    @classmethod
    def from_grid(cls, grid: Grid, particles_per_cell: int = 1) -> "ParticleSystem":
        """Create particle system with uniform seeding from grid.

        Seeds particles at cell centers (or sub-cell positions for multiple
        particles per cell).

        Args:
            grid: Grid to seed particles on
            particles_per_cell: Number of particles per grid cell

        Returns:
            Initialized ParticleSystem with positions set
        """
        n_particles = grid.nx * grid.ny * particles_per_cell
        ps = cls(n_particles)

        if particles_per_cell == 1:
            # One particle at each cell center
            X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
            ps.x = X.flatten()
            ps.y = Y.flatten()
        else:
            # Multiple particles per cell - sub-cell positions
            idx = 0
            n_side = int(np.sqrt(particles_per_cell))
            if n_side * n_side != particles_per_cell:
                raise ValueError("particles_per_cell must be a perfect square")

            for i in range(grid.nx):
                for j in range(grid.ny):
                    for pi in range(n_side):
                        for pj in range(n_side):
                            ps.x[idx] = (i + (pi + 0.5) / n_side) * grid.dx
                            ps.y[idx] = (j + (pj + 0.5) / n_side) * grid.dy
                            idx += 1

        return ps

    def reset_jacobian(self):
        """Reset all Jacobians to identity matrix."""
        self.J.fill(0.0)
        self.J[:, 0, 0] = 1.0
        self.J[:, 1, 1] = 1.0

    def wrap_positions(self, Lx: float, Ly: float):
        """Apply periodic boundary conditions to particle positions.

        Args:
            Lx: Domain length in x direction
            Ly: Domain length in y direction
        """
        self.x = self.x % Lx
        self.y = self.y % Ly

    def set_initial_condition(self, q_func, grid: Grid):
        """Set initial potential vorticity from a function.

        Args:
            q_func: Function q(x, y) returning potential vorticity
            grid: Grid for domain information
        """
        self.q = q_func(self.x, self.y)

    def jacobian_norm(self) -> np.ndarray:
        """Compute ||J - I|| for each particle (Frobenius norm).

        Returns:
            Array of Jacobian deviations from identity (n_particles,)
        """
        # J - I
        J_dev = self.J.copy()
        J_dev[:, 0, 0] -= 1.0
        J_dev[:, 1, 1] -= 1.0

        # Frobenius norm
        return np.sqrt(np.sum(J_dev**2, axis=(1, 2)))

    def max_jacobian_deviation(self) -> float:
        """Get maximum Jacobian deviation from identity.

        Returns:
            Maximum ||J - I|| across all particles
        """
        return np.max(self.jacobian_norm())
