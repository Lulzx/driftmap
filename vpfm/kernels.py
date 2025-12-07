"""Higher-order interpolation kernels for VPFM-Plasma.

Implements B-spline kernels for improved P2G/G2P accuracy:
- Linear (tent): C⁰ continuous, 4 neighbors - current default
- Quadratic B-spline: C¹ continuous, 9 neighbors
- Cubic B-spline: C² continuous, 16 neighbors

Higher-order kernels reduce interpolation error and improve
structure preservation, at the cost of more computation per transfer.
"""

import numpy as np
from typing import Tuple


def linear_kernel_1d(x: np.ndarray) -> np.ndarray:
    """Linear (tent) kernel in 1D.

    W(x) = 1 - |x|  for |x| < 1
           0        otherwise

    Support: [-1, 1]
    """
    ax = np.abs(x)
    return np.where(ax < 1, 1 - ax, 0.0)


def quadratic_bspline_1d(x: np.ndarray) -> np.ndarray:
    """Quadratic B-spline kernel in 1D.

    W(x) = 3/4 - x²           for |x| < 0.5
           (3/2 - |x|)²/2     for 0.5 ≤ |x| < 1.5
           0                  otherwise

    Support: [-1.5, 1.5], C¹ continuous
    """
    ax = np.abs(x)
    result = np.zeros_like(x)

    # |x| < 0.5
    mask1 = ax < 0.5
    result[mask1] = 0.75 - ax[mask1]**2

    # 0.5 ≤ |x| < 1.5
    mask2 = (ax >= 0.5) & (ax < 1.5)
    result[mask2] = 0.5 * (1.5 - ax[mask2])**2

    return result


def cubic_bspline_1d(x: np.ndarray) -> np.ndarray:
    """Cubic B-spline kernel in 1D.

    W(x) = 2/3 - x² + |x|³/2           for |x| < 1
           (2 - |x|)³/6                 for 1 ≤ |x| < 2
           0                            otherwise

    Support: [-2, 2], C² continuous
    """
    ax = np.abs(x)
    result = np.zeros_like(x)

    # |x| < 1
    mask1 = ax < 1
    result[mask1] = 2/3 - ax[mask1]**2 + ax[mask1]**3 / 2

    # 1 ≤ |x| < 2
    mask2 = (ax >= 1) & (ax < 2)
    result[mask2] = (2 - ax[mask2])**3 / 6

    return result


def quadratic_bspline_derivative_1d(x: np.ndarray) -> np.ndarray:
    """Derivative of quadratic B-spline kernel.

    dW/dx for computing gradients during G2P.
    """
    ax = np.abs(x)
    sx = np.sign(x)
    result = np.zeros_like(x)

    # |x| < 0.5: d/dx(3/4 - x²) = -2x
    mask1 = ax < 0.5
    result[mask1] = -2 * x[mask1]

    # 0.5 ≤ |x| < 1.5: d/dx((3/2 - |x|)²/2) = -(3/2 - |x|) * sign(x)
    mask2 = (ax >= 0.5) & (ax < 1.5)
    result[mask2] = -(1.5 - ax[mask2]) * sx[mask2]

    return result


def cubic_bspline_derivative_1d(x: np.ndarray) -> np.ndarray:
    """Derivative of cubic B-spline kernel."""
    ax = np.abs(x)
    sx = np.sign(x)
    result = np.zeros_like(x)

    # |x| < 1: d/dx(2/3 - x² + |x|³/2) = -2x + 3x²/2 * sign(x)
    mask1 = ax < 1
    result[mask1] = -2 * x[mask1] + 1.5 * ax[mask1]**2 * sx[mask1]

    # 1 ≤ |x| < 2: d/dx((2 - |x|)³/6) = -(2 - |x|)²/2 * sign(x)
    mask2 = (ax >= 1) & (ax < 2)
    result[mask2] = -0.5 * (2 - ax[mask2])**2 * sx[mask2]

    return result


class InterpolationKernel:
    """2D interpolation kernel for P2G/G2P operations."""

    def __init__(self, order: str = 'quadratic'):
        """Initialize kernel.

        Args:
            order: 'linear', 'quadratic', or 'cubic'
        """
        self.order = order

        if order == 'linear':
            self.kernel_1d = linear_kernel_1d
            self.derivative_1d = lambda x: np.where(np.abs(x) < 1, -np.sign(x), 0.0)
            self.support = 1  # Half-width of support
            self.n_neighbors = 2  # Neighbors per dimension
        elif order == 'quadratic':
            self.kernel_1d = quadratic_bspline_1d
            self.derivative_1d = quadratic_bspline_derivative_1d
            self.support = 1.5
            self.n_neighbors = 3
        elif order == 'cubic':
            self.kernel_1d = cubic_bspline_1d
            self.derivative_1d = cubic_bspline_derivative_1d
            self.support = 2
            self.n_neighbors = 4
        else:
            raise ValueError(f"Unknown kernel order: {order}")

    def weights_2d(self, x: np.ndarray, y: np.ndarray,
                   dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D weights for a set of particles.

        Args:
            x, y: Particle positions (n_particles,)
            dx, dy: Grid spacing

        Returns:
            (base_i, base_j, weights) where:
            - base_i, base_j: Base grid indices (n_particles,)
            - weights: Weight array (n_particles, n_neighbors, n_neighbors)
        """
        n_particles = len(x)
        n = self.n_neighbors

        # Normalized positions
        x_norm = x / dx
        y_norm = y / dy

        # Base indices (lower-left of stencil)
        if self.order == 'linear':
            base_i = np.floor(x_norm).astype(int)
            base_j = np.floor(y_norm).astype(int)
            # Offset from base
            fx = x_norm - base_i
            fy = y_norm - base_j
        elif self.order == 'quadratic':
            # For quadratic, center on nearest grid point
            base_i = np.floor(x_norm + 0.5).astype(int) - 1
            base_j = np.floor(y_norm + 0.5).astype(int) - 1
            fx = x_norm - (base_i + 1)  # Offset from center
            fy = y_norm - (base_j + 1)
        else:  # cubic
            base_i = np.floor(x_norm).astype(int) - 1
            base_j = np.floor(y_norm).astype(int) - 1
            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)

        # Compute 1D weights
        weights = np.zeros((n_particles, n, n))

        for di in range(n):
            for dj in range(n):
                if self.order == 'linear':
                    xi = di - fx
                    yj = dj - fy
                elif self.order == 'quadratic':
                    xi = (di - 1) - fx  # di=0,1,2 -> offset -1,0,1
                    yj = (dj - 1) - fy
                else:  # cubic
                    xi = (di - 1) - fx  # di=0,1,2,3 -> offset -1,0,1,2
                    yj = (dj - 1) - fy

                wx = self.kernel_1d(xi)
                wy = self.kernel_1d(yj)
                weights[:, di, dj] = wx * wy

        return base_i, base_j, weights

    def weights_and_gradients_2d(self, x: np.ndarray, y: np.ndarray,
                                  dx: float, dy: float) -> Tuple:
        """Compute weights and gradient weights.

        Returns:
            (base_i, base_j, weights, grad_x, grad_y)
        """
        n_particles = len(x)
        n = self.n_neighbors

        x_norm = x / dx
        y_norm = y / dy

        if self.order == 'linear':
            base_i = np.floor(x_norm).astype(int)
            base_j = np.floor(y_norm).astype(int)
            fx = x_norm - base_i
            fy = y_norm - base_j
        elif self.order == 'quadratic':
            base_i = np.floor(x_norm + 0.5).astype(int) - 1
            base_j = np.floor(y_norm + 0.5).astype(int) - 1
            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
        else:
            base_i = np.floor(x_norm).astype(int) - 1
            base_j = np.floor(y_norm).astype(int) - 1
            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)

        weights = np.zeros((n_particles, n, n))
        grad_x = np.zeros((n_particles, n, n))
        grad_y = np.zeros((n_particles, n, n))

        for di in range(n):
            for dj in range(n):
                if self.order == 'linear':
                    xi = di - fx
                    yj = dj - fy
                elif self.order == 'quadratic':
                    xi = (di - 1) - fx
                    yj = (dj - 1) - fy
                else:
                    xi = (di - 1) - fx
                    yj = (dj - 1) - fy

                wx = self.kernel_1d(xi)
                wy = self.kernel_1d(yj)
                dwx = self.derivative_1d(xi) / dx
                dwy = self.derivative_1d(yj) / dy

                weights[:, di, dj] = wx * wy
                grad_x[:, di, dj] = dwx * wy
                grad_y[:, di, dj] = wx * dwy

        return base_i, base_j, weights, grad_x, grad_y


def P2G_bspline(particles_x: np.ndarray, particles_y: np.ndarray,
                particles_q: np.ndarray, nx: int, ny: int,
                dx: float, dy: float, kernel: InterpolationKernel) -> np.ndarray:
    """Transfer particle vorticity to grid using B-spline kernel.

    Args:
        particles_x, particles_y: Particle positions
        particles_q: Particle vorticity
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
        kernel: Interpolation kernel

    Returns:
        Grid vorticity field (nx, ny)
    """
    q_grid = np.zeros((nx, ny))
    weight_grid = np.zeros((nx, ny))

    base_i, base_j, weights = kernel.weights_2d(particles_x, particles_y, dx, dy)

    n = kernel.n_neighbors
    n_particles = len(particles_x)

    for p in range(n_particles):
        q_p = particles_q[p]
        bi, bj = base_i[p], base_j[p]

        for di in range(n):
            for dj in range(n):
                gi = (bi + di) % nx
                gj = (bj + dj) % ny
                w = weights[p, di, dj]

                q_grid[gi, gj] += w * q_p
                weight_grid[gi, gj] += w

    # Normalize
    mask = weight_grid > 1e-10
    q_grid[mask] /= weight_grid[mask]

    return q_grid


def G2P_bspline(grid_field: np.ndarray, particles_x: np.ndarray,
                particles_y: np.ndarray, dx: float, dy: float,
                kernel: InterpolationKernel) -> np.ndarray:
    """Interpolate grid field to particles using B-spline kernel.

    Args:
        grid_field: Field on grid (nx, ny)
        particles_x, particles_y: Particle positions
        dx, dy: Grid spacing
        kernel: Interpolation kernel

    Returns:
        Field values at particles (n_particles,)
    """
    nx, ny = grid_field.shape
    n_particles = len(particles_x)

    base_i, base_j, weights = kernel.weights_2d(particles_x, particles_y, dx, dy)

    n = kernel.n_neighbors
    values = np.zeros(n_particles)

    for p in range(n_particles):
        bi, bj = base_i[p], base_j[p]

        for di in range(n):
            for dj in range(n):
                gi = (bi + di) % nx
                gj = (bj + dj) % ny
                values[p] += weights[p, di, dj] * grid_field[gi, gj]

    return values


def G2P_bspline_with_gradient(grid_field: np.ndarray, particles_x: np.ndarray,
                               particles_y: np.ndarray, dx: float, dy: float,
                               kernel: InterpolationKernel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate grid field and its gradient to particles.

    Returns:
        (values, grad_x, grad_y) at particle positions
    """
    nx, ny = grid_field.shape
    n_particles = len(particles_x)

    base_i, base_j, weights, gx, gy = kernel.weights_and_gradients_2d(
        particles_x, particles_y, dx, dy)

    n = kernel.n_neighbors
    values = np.zeros(n_particles)
    grad_x = np.zeros(n_particles)
    grad_y = np.zeros(n_particles)

    for p in range(n_particles):
        bi, bj = base_i[p], base_j[p]

        for di in range(n):
            for dj in range(n):
                gi = (bi + di) % nx
                gj = (bj + dj) % ny
                f = grid_field[gi, gj]

                values[p] += weights[p, di, dj] * f
                grad_x[p] += gx[p, di, dj] * f
                grad_y[p] += gy[p, di, dj] * f

    return values, grad_x, grad_y
