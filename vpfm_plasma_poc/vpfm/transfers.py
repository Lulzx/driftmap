"""Particle-Grid transfer operations for VPFM-Plasma.

Implements P2G (Particle to Grid) and G2P (Grid to Particle) transfers
using bilinear interpolation.
"""

import numpy as np
from .grid import Grid
from .particles import ParticleSystem


def P2G(particles: ParticleSystem, grid: Grid) -> np.ndarray:
    """Transfer potential vorticity from particles to grid.

    Uses bilinear interpolation weights to distribute particle vorticity
    to the four surrounding grid nodes.

    Args:
        particles: Particle system with vorticity values
        grid: Target grid

    Returns:
        Grid vorticity field (nx, ny)
    """
    q_grid = np.zeros((grid.nx, grid.ny))
    weight_grid = np.zeros((grid.nx, grid.ny))

    # Get interpolation weights for all particles
    i, j, fx, fy = grid.get_bilinear_weights(particles.x, particles.y)

    # Neighbor indices with periodic wrapping
    ip1 = (i + 1) % grid.nx
    jp1 = (j + 1) % grid.ny

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    # Accumulate contributions from each particle
    for p in range(particles.n_particles):
        q_p = particles.q[p]

        # Distribute to 4 corners
        q_grid[i[p], j[p]] += w00[p] * q_p
        q_grid[ip1[p], j[p]] += w10[p] * q_p
        q_grid[i[p], jp1[p]] += w01[p] * q_p
        q_grid[ip1[p], jp1[p]] += w11[p] * q_p

        # Track weights for normalization
        weight_grid[i[p], j[p]] += w00[p]
        weight_grid[ip1[p], j[p]] += w10[p]
        weight_grid[i[p], jp1[p]] += w01[p]
        weight_grid[ip1[p], jp1[p]] += w11[p]

    # Normalize by total weight (avoid division by zero)
    mask = weight_grid > 1e-10
    q_grid[mask] /= weight_grid[mask]

    return q_grid


def P2G_vectorized(particles: ParticleSystem, grid: Grid) -> np.ndarray:
    """Vectorized P2G transfer (faster for large particle counts).

    Uses numpy advanced indexing with np.add.at for atomic accumulation.

    Args:
        particles: Particle system with vorticity values
        grid: Target grid

    Returns:
        Grid vorticity field (nx, ny)
    """
    q_grid = np.zeros((grid.nx, grid.ny))
    weight_grid = np.zeros((grid.nx, grid.ny))

    # Get interpolation weights
    i, j, fx, fy = grid.get_bilinear_weights(particles.x, particles.y)

    # Neighbor indices
    ip1 = (i + 1) % grid.nx
    jp1 = (j + 1) % grid.ny

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    # Weighted vorticity contributions
    q = particles.q

    # Accumulate using np.add.at (handles duplicate indices correctly)
    np.add.at(q_grid, (i, j), w00 * q)
    np.add.at(q_grid, (ip1, j), w10 * q)
    np.add.at(q_grid, (i, jp1), w01 * q)
    np.add.at(q_grid, (ip1, jp1), w11 * q)

    np.add.at(weight_grid, (i, j), w00)
    np.add.at(weight_grid, (ip1, j), w10)
    np.add.at(weight_grid, (i, jp1), w01)
    np.add.at(weight_grid, (ip1, jp1), w11)

    # Normalize
    mask = weight_grid > 1e-10
    q_grid[mask] /= weight_grid[mask]

    return q_grid


def G2P(grid: Grid, particles: ParticleSystem, field: np.ndarray) -> np.ndarray:
    """Transfer a grid field to particle positions.

    Uses bilinear interpolation from the four surrounding grid nodes.

    Args:
        grid: Source grid
        particles: Particle system (for positions)
        field: Grid field to interpolate (nx, ny)

    Returns:
        Field values at particle positions (n_particles,)
    """
    # Get interpolation weights
    i, j, fx, fy = grid.get_bilinear_weights(particles.x, particles.y)

    # Neighbor indices
    ip1 = (i + 1) % grid.nx
    jp1 = (j + 1) % grid.ny

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    # Interpolate
    values = (w00 * field[i, j] +
              w10 * field[ip1, j] +
              w01 * field[i, jp1] +
              w11 * field[ip1, jp1])

    return values


def G2P_velocity(grid: Grid, particles: ParticleSystem) -> tuple:
    """Transfer velocity field to particle positions.

    Args:
        grid: Source grid with velocity fields
        particles: Particle system

    Returns:
        (vx, vy) velocity components at particle positions
    """
    vx = G2P(grid, particles, grid.vx)
    vy = G2P(grid, particles, grid.vy)
    return vx, vy


def G2P_velocity_gradient(grid: Grid, particles: ParticleSystem) -> tuple:
    """Transfer velocity gradient to particle positions.

    Args:
        grid: Source grid with velocity gradient fields
        particles: Particle system

    Returns:
        (dvx_dx, dvx_dy, dvy_dx, dvy_dy) gradient components at particles
    """
    dvx_dx = G2P(grid, particles, grid.dvx_dx)
    dvx_dy = G2P(grid, particles, grid.dvx_dy)
    dvy_dx = G2P(grid, particles, grid.dvy_dx)
    dvy_dy = G2P(grid, particles, grid.dvy_dy)

    return dvx_dx, dvx_dy, dvy_dx, dvy_dy
