"""Higher-order interpolation kernels for VPFM-Plasma.

Implements B-spline kernels for improved P2G/G2P accuracy:
- Linear (tent): C⁰ continuous, 4 neighbors - current default
- Quadratic B-spline: C¹ continuous, 9 neighbors
- Cubic B-spline: C² continuous, 16 neighbors

Higher-order kernels reduce interpolation error and improve
structure preservation, at the cost of more computation per transfer.

Performance optimized with Numba JIT compilation.
"""

import numpy as np
from typing import Tuple
from numba import njit, prange


# =============================================================================
# Numba-compiled 1D kernel functions (scalar versions for JIT)
# =============================================================================

@njit(cache=True, fastmath=True)
def _linear_kernel_scalar(x: float) -> float:
    """Linear (tent) kernel for single value."""
    ax = abs(x)
    if ax < 1.0:
        return 1.0 - ax
    return 0.0


@njit(cache=True, fastmath=True)
def _quadratic_bspline_scalar(x: float) -> float:
    """Quadratic B-spline kernel for single value."""
    ax = abs(x)
    if ax < 0.5:
        return 0.75 - ax * ax
    elif ax < 1.5:
        t = 1.5 - ax
        return 0.5 * t * t
    return 0.0


@njit(cache=True, fastmath=True)
def _cubic_bspline_scalar(x: float) -> float:
    """Cubic B-spline kernel for single value."""
    ax = abs(x)
    if ax < 1.0:
        return 2.0/3.0 - ax*ax + 0.5*ax*ax*ax
    elif ax < 2.0:
        t = 2.0 - ax
        return t*t*t / 6.0
    return 0.0


@njit(cache=True, fastmath=True)
def _quadratic_bspline_deriv_scalar(x: float) -> float:
    """Derivative of quadratic B-spline."""
    ax = abs(x)
    if ax < 0.5:
        return -2.0 * x
    elif ax < 1.5:
        sx = 1.0 if x > 0 else -1.0
        return -(1.5 - ax) * sx
    return 0.0


@njit(cache=True, fastmath=True)
def _cubic_bspline_deriv_scalar(x: float) -> float:
    """Derivative of cubic B-spline."""
    ax = abs(x)
    sx = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    if ax < 1.0:
        return -2.0 * x + 1.5 * ax * ax * sx
    elif ax < 2.0:
        t = 2.0 - ax
        return -0.5 * t * t * sx
    return 0.0


# =============================================================================
# Vectorized kernel functions (for compatibility)
# =============================================================================

def linear_kernel_1d(x: np.ndarray) -> np.ndarray:
    """Linear (tent) kernel in 1D."""
    ax = np.abs(x)
    return np.where(ax < 1, 1 - ax, 0.0)


def quadratic_bspline_1d(x: np.ndarray) -> np.ndarray:
    """Quadratic B-spline kernel in 1D."""
    ax = np.abs(x)
    result = np.zeros_like(x)
    mask1 = ax < 0.5
    result[mask1] = 0.75 - ax[mask1]**2
    mask2 = (ax >= 0.5) & (ax < 1.5)
    result[mask2] = 0.5 * (1.5 - ax[mask2])**2
    return result


def cubic_bspline_1d(x: np.ndarray) -> np.ndarray:
    """Cubic B-spline kernel in 1D."""
    ax = np.abs(x)
    result = np.zeros_like(x)
    mask1 = ax < 1
    result[mask1] = 2/3 - ax[mask1]**2 + ax[mask1]**3 / 2
    mask2 = (ax >= 1) & (ax < 2)
    result[mask2] = (2 - ax[mask2])**3 / 6
    return result


def quadratic_bspline_derivative_1d(x: np.ndarray) -> np.ndarray:
    """Derivative of quadratic B-spline kernel."""
    ax = np.abs(x)
    sx = np.sign(x)
    result = np.zeros_like(x)
    mask1 = ax < 0.5
    result[mask1] = -2 * x[mask1]
    mask2 = (ax >= 0.5) & (ax < 1.5)
    result[mask2] = -(1.5 - ax[mask2]) * sx[mask2]
    return result


def cubic_bspline_derivative_1d(x: np.ndarray) -> np.ndarray:
    """Derivative of cubic B-spline kernel."""
    ax = np.abs(x)
    sx = np.sign(x)
    result = np.zeros_like(x)
    mask1 = ax < 1
    result[mask1] = -2 * x[mask1] + 1.5 * ax[mask1]**2 * sx[mask1]
    mask2 = (ax >= 1) & (ax < 2)
    result[mask2] = -0.5 * (2 - ax[mask2])**2 * sx[mask2]
    return result


# =============================================================================
# Numba-optimized P2G transfers
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _p2g_quadratic_numba(px: np.ndarray, py: np.ndarray, pq: np.ndarray,
                          nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized P2G with quadratic B-spline kernel.

    Returns (q_grid, weight_grid) to be normalized afterwards.
    """
    n_particles = len(px)
    q_grid = np.zeros((nx, ny))
    weight_grid = np.zeros((nx, ny))

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        # Base index (center on nearest grid point)
        base_i = int(np.floor(x_norm + 0.5)) - 1
        base_j = int(np.floor(y_norm + 0.5)) - 1

        # Offset from center
        fx = x_norm - (base_i + 1)
        fy = y_norm - (base_j + 1)

        q_p = pq[p]

        # 3x3 stencil for quadratic
        for di in range(3):
            xi = (di - 1) - fx
            wx = _quadratic_bspline_scalar(xi)
            gi = (base_i + di) % nx

            for dj in range(3):
                yj = (dj - 1) - fy
                wy = _quadratic_bspline_scalar(yj)
                gj = (base_j + dj) % ny

                w = wx * wy
                # Atomic add (handled by numba for parallel)
                q_grid[gi, gj] += w * q_p
                weight_grid[gi, gj] += w

    return q_grid, weight_grid


@njit(parallel=True, cache=True, fastmath=True)
def _p2g_gradient_enhanced_quadratic_numba(
    px: np.ndarray, py: np.ndarray, pq: np.ndarray,
    grad_q_x: np.ndarray, grad_q_y: np.ndarray,
    nx: int, ny: int, dx: float, dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient-enhanced P2G with quadratic B-spline kernel.

    Implements: ω_i^g = (Σ_p s_ip · [ω_p + ∇ω_p · (x_i - x_p)]) / (Σ_p s_ip)

    This improves accuracy by including local gradient information.

    Returns (q_grid, weight_grid) to be normalized afterwards.
    """
    n_particles = len(px)
    q_grid = np.zeros((nx, ny))
    weight_grid = np.zeros((nx, ny))

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        # Base index (center on nearest grid point)
        base_i = int(np.floor(x_norm + 0.5)) - 1
        base_j = int(np.floor(y_norm + 0.5)) - 1

        # Offset from center
        fx = x_norm - (base_i + 1)
        fy = y_norm - (base_j + 1)

        q_p = pq[p]
        gx_p = grad_q_x[p]
        gy_p = grad_q_y[p]

        # 3x3 stencil for quadratic
        for di in range(3):
            xi = (di - 1) - fx
            wx = _quadratic_bspline_scalar(xi)
            gi = (base_i + di) % nx

            # Distance from particle to grid point in physical units
            delta_x = xi * dx

            for dj in range(3):
                yj = (dj - 1) - fy
                wy = _quadratic_bspline_scalar(yj)
                gj = (base_j + dj) % ny

                delta_y = yj * dy

                w = wx * wy

                # Gradient-enhanced contribution: q_p + ∇q_p · Δx
                q_enhanced = q_p + gx_p * delta_x + gy_p * delta_y

                q_grid[gi, gj] += w * q_enhanced
                weight_grid[gi, gj] += w

    return q_grid, weight_grid


@njit(parallel=True, cache=True, fastmath=True)
def _p2g_linear_numba(px: np.ndarray, py: np.ndarray, pq: np.ndarray,
                       nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized P2G with linear kernel."""
    n_particles = len(px)
    q_grid = np.zeros((nx, ny))
    weight_grid = np.zeros((nx, ny))

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        base_i = int(np.floor(x_norm))
        base_j = int(np.floor(y_norm))

        fx = x_norm - base_i
        fy = y_norm - base_j

        q_p = pq[p]

        for di in range(2):
            xi = di - fx
            wx = _linear_kernel_scalar(xi)
            gi = (base_i + di) % nx

            for dj in range(2):
                yj = dj - fy
                wy = _linear_kernel_scalar(yj)
                gj = (base_j + dj) % ny

                w = wx * wy
                q_grid[gi, gj] += w * q_p
                weight_grid[gi, gj] += w

    return q_grid, weight_grid


@njit(parallel=True, cache=True, fastmath=True)
def _p2g_cubic_numba(px: np.ndarray, py: np.ndarray, pq: np.ndarray,
                      nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized P2G with cubic B-spline kernel."""
    n_particles = len(px)
    q_grid = np.zeros((nx, ny))
    weight_grid = np.zeros((nx, ny))

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        base_i = int(np.floor(x_norm)) - 1
        base_j = int(np.floor(y_norm)) - 1

        fx = x_norm - (base_i + 1)
        fy = y_norm - (base_j + 1)

        q_p = pq[p]

        for di in range(4):
            xi = (di - 1) - fx
            wx = _cubic_bspline_scalar(xi)
            gi = (base_i + di) % nx

            for dj in range(4):
                yj = (dj - 1) - fy
                wy = _cubic_bspline_scalar(yj)
                gj = (base_j + dj) % ny

                w = wx * wy
                q_grid[gi, gj] += w * q_p
                weight_grid[gi, gj] += w

    return q_grid, weight_grid


# =============================================================================
# Numba-optimized G2P transfers
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _g2p_quadratic_numba(grid_field: np.ndarray, px: np.ndarray, py: np.ndarray,
                          dx: float, dy: float) -> np.ndarray:
    """Numba-optimized G2P with quadratic B-spline kernel."""
    nx, ny = grid_field.shape
    n_particles = len(px)
    values = np.zeros(n_particles)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        base_i = int(np.floor(x_norm + 0.5)) - 1
        base_j = int(np.floor(y_norm + 0.5)) - 1

        fx = x_norm - (base_i + 1)
        fy = y_norm - (base_j + 1)

        val = 0.0
        for di in range(3):
            xi = (di - 1) - fx
            wx = _quadratic_bspline_scalar(xi)
            gi = (base_i + di) % nx

            for dj in range(3):
                yj = (dj - 1) - fy
                wy = _quadratic_bspline_scalar(yj)
                gj = (base_j + dj) % ny

                val += wx * wy * grid_field[gi, gj]

        values[p] = val

    return values


@njit(parallel=True, cache=True, fastmath=True)
def _g2p_linear_numba(grid_field: np.ndarray, px: np.ndarray, py: np.ndarray,
                       dx: float, dy: float) -> np.ndarray:
    """Numba-optimized G2P with linear kernel."""
    nx, ny = grid_field.shape
    n_particles = len(px)
    values = np.zeros(n_particles)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        base_i = int(np.floor(x_norm))
        base_j = int(np.floor(y_norm))

        fx = x_norm - base_i
        fy = y_norm - base_j

        val = 0.0
        for di in range(2):
            xi = di - fx
            wx = _linear_kernel_scalar(xi)
            gi = (base_i + di) % nx

            for dj in range(2):
                yj = dj - fy
                wy = _linear_kernel_scalar(yj)
                gj = (base_j + dj) % ny

                val += wx * wy * grid_field[gi, gj]

        values[p] = val

    return values


@njit(parallel=True, cache=True, fastmath=True)
def _g2p_cubic_numba(grid_field: np.ndarray, px: np.ndarray, py: np.ndarray,
                      dx: float, dy: float) -> np.ndarray:
    """Numba-optimized G2P with cubic B-spline kernel."""
    nx, ny = grid_field.shape
    n_particles = len(px)
    values = np.zeros(n_particles)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        base_i = int(np.floor(x_norm)) - 1
        base_j = int(np.floor(y_norm)) - 1

        fx = x_norm - (base_i + 1)
        fy = y_norm - (base_j + 1)

        val = 0.0
        for di in range(4):
            xi = (di - 1) - fx
            wx = _cubic_bspline_scalar(xi)
            gi = (base_i + di) % nx

            for dj in range(4):
                yj = (dj - 1) - fy
                wy = _cubic_bspline_scalar(yj)
                gj = (base_j + dj) % ny

                val += wx * wy * grid_field[gi, gj]

        values[p] = val

    return values


@njit(parallel=True, cache=True, fastmath=True)
def _g2p_with_gradient_quadratic_numba(grid_field: np.ndarray, px: np.ndarray, py: np.ndarray,
                                        dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized G2P with gradient for quadratic kernel."""
    nx, ny = grid_field.shape
    n_particles = len(px)
    values = np.zeros(n_particles)
    grad_x = np.zeros(n_particles)
    grad_y = np.zeros(n_particles)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in prange(n_particles):
        x_norm = px[p] * inv_dx
        y_norm = py[p] * inv_dy

        base_i = int(np.floor(x_norm + 0.5)) - 1
        base_j = int(np.floor(y_norm + 0.5)) - 1

        fx = x_norm - (base_i + 1)
        fy = y_norm - (base_j + 1)

        val = 0.0
        gx = 0.0
        gy = 0.0

        for di in range(3):
            xi = (di - 1) - fx
            wx = _quadratic_bspline_scalar(xi)
            dwx = _quadratic_bspline_deriv_scalar(xi) * inv_dx
            gi = (base_i + di) % nx

            for dj in range(3):
                yj = (dj - 1) - fy
                wy = _quadratic_bspline_scalar(yj)
                dwy = _quadratic_bspline_deriv_scalar(yj) * inv_dy
                gj = (base_j + dj) % ny

                f = grid_field[gi, gj]
                val += wx * wy * f
                gx += dwx * wy * f
                gy += wx * dwy * f

        values[p] = val
        grad_x[p] = gx
        grad_y[p] = gy

    return values, grad_x, grad_y


# =============================================================================
# InterpolationKernel class (compatibility wrapper)
# =============================================================================

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
            self.support = 1
            self.n_neighbors = 2
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
        """Compute 2D weights for a set of particles."""
        n_particles = len(x)
        n = self.n_neighbors

        x_norm = x / dx
        y_norm = y / dy

        if self.order == 'linear':
            base_i = np.floor(x_norm).astype(np.int32)
            base_j = np.floor(y_norm).astype(np.int32)
            fx = x_norm - base_i
            fy = y_norm - base_j
        elif self.order == 'quadratic':
            base_i = np.floor(x_norm + 0.5).astype(np.int32) - 1
            base_j = np.floor(y_norm + 0.5).astype(np.int32) - 1
            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
        else:
            base_i = np.floor(x_norm).astype(np.int32) - 1
            base_j = np.floor(y_norm).astype(np.int32) - 1
            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)

        weights = np.zeros((n_particles, n, n))

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
                weights[:, di, dj] = wx * wy

        return base_i, base_j, weights

    def weights_and_gradients_2d(self, x: np.ndarray, y: np.ndarray,
                                  dx: float, dy: float) -> Tuple:
        """Compute weights and gradient weights."""
        n_particles = len(x)
        n = self.n_neighbors

        x_norm = x / dx
        y_norm = y / dy

        if self.order == 'linear':
            base_i = np.floor(x_norm).astype(np.int32)
            base_j = np.floor(y_norm).astype(np.int32)
            fx = x_norm - base_i
            fy = y_norm - base_j
        elif self.order == 'quadratic':
            base_i = np.floor(x_norm + 0.5).astype(np.int32) - 1
            base_j = np.floor(y_norm + 0.5).astype(np.int32) - 1
            fx = x_norm - (base_i + 1)
            fy = y_norm - (base_j + 1)
        else:
            base_i = np.floor(x_norm).astype(np.int32) - 1
            base_j = np.floor(y_norm).astype(np.int32) - 1
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


# =============================================================================
# Public API functions (use Numba-optimized versions)
# =============================================================================

def P2G_bspline(particles_x: np.ndarray, particles_y: np.ndarray,
                particles_q: np.ndarray, nx: int, ny: int,
                dx: float, dy: float, kernel: InterpolationKernel) -> np.ndarray:
    """Transfer particle vorticity to grid using B-spline kernel.

    Uses Numba-optimized implementation for performance.
    """
    # Ensure contiguous arrays
    px = np.ascontiguousarray(particles_x)
    py = np.ascontiguousarray(particles_y)
    pq = np.ascontiguousarray(particles_q)

    if kernel.order == 'linear':
        q_grid, weight_grid = _p2g_linear_numba(px, py, pq, nx, ny, dx, dy)
    elif kernel.order == 'quadratic':
        q_grid, weight_grid = _p2g_quadratic_numba(px, py, pq, nx, ny, dx, dy)
    else:  # cubic
        q_grid, weight_grid = _p2g_cubic_numba(px, py, pq, nx, ny, dx, dy)

    # Normalize
    mask = weight_grid > 1e-10
    q_grid[mask] /= weight_grid[mask]

    return q_grid


def P2G_bspline_gradient_enhanced(
    particles_x: np.ndarray, particles_y: np.ndarray,
    particles_q: np.ndarray,
    grad_q_x: np.ndarray, grad_q_y: np.ndarray,
    nx: int, ny: int, dx: float, dy: float,
    kernel: InterpolationKernel
) -> np.ndarray:
    """Gradient-enhanced P2G transfer.

    Implements: ω_i^g = (Σ_p s_ip · [ω_p + ∇ω_p · (x_i - x_p)]) / (Σ_p s_ip)

    This improves accuracy by including local gradient information from particles.

    Args:
        particles_x, particles_y: Particle positions
        particles_q: Particle vorticity values
        grad_q_x, grad_q_y: Vorticity gradients on particles
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
        kernel: Interpolation kernel

    Returns:
        Grid field with gradient-enhanced interpolation
    """
    # Ensure contiguous arrays
    px = np.ascontiguousarray(particles_x)
    py = np.ascontiguousarray(particles_y)
    pq = np.ascontiguousarray(particles_q)
    gx = np.ascontiguousarray(grad_q_x)
    gy = np.ascontiguousarray(grad_q_y)

    if kernel.order == 'quadratic':
        q_grid, weight_grid = _p2g_gradient_enhanced_quadratic_numba(
            px, py, pq, gx, gy, nx, ny, dx, dy
        )
    else:
        # Fallback to standard P2G for non-quadratic kernels
        # (gradient-enhanced only implemented for quadratic so far)
        return P2G_bspline(particles_x, particles_y, particles_q,
                          nx, ny, dx, dy, kernel)

    # Normalize
    mask = weight_grid > 1e-10
    q_grid[mask] /= weight_grid[mask]

    return q_grid


def G2P_bspline(grid_field: np.ndarray, particles_x: np.ndarray,
                particles_y: np.ndarray, dx: float, dy: float,
                kernel: InterpolationKernel) -> np.ndarray:
    """Interpolate grid field to particles using B-spline kernel.

    Uses Numba-optimized implementation for performance.
    """
    # Ensure contiguous arrays
    field = np.ascontiguousarray(grid_field)
    px = np.ascontiguousarray(particles_x)
    py = np.ascontiguousarray(particles_y)

    if kernel.order == 'linear':
        return _g2p_linear_numba(field, px, py, dx, dy)
    elif kernel.order == 'quadratic':
        return _g2p_quadratic_numba(field, px, py, dx, dy)
    else:  # cubic
        return _g2p_cubic_numba(field, px, py, dx, dy)


def G2P_bspline_with_gradient(grid_field: np.ndarray, particles_x: np.ndarray,
                               particles_y: np.ndarray, dx: float, dy: float,
                               kernel: InterpolationKernel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate grid field and its gradient to particles.

    Uses Numba-optimized implementation for quadratic kernel.
    """
    # Ensure contiguous arrays
    field = np.ascontiguousarray(grid_field)
    px = np.ascontiguousarray(particles_x)
    py = np.ascontiguousarray(particles_y)

    if kernel.order == 'quadratic':
        return _g2p_with_gradient_quadratic_numba(field, px, py, dx, dy)

    # Fallback to non-optimized for other kernels
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
