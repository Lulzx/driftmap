"""GPU-accelerated P2G/G2P kernels using CuPy.

Provides CUDA implementations of:
- P2G (particle to grid) transfers with B-spline kernels
- G2P (grid to particle) interpolation
- Flow map Jacobian evolution

These kernels provide significant speedup for large particle counts.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import CuPy
try:
    import cupy as cp
    from cupyx import jit
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return CUPY_AVAILABLE


if CUPY_AVAILABLE:
    # =========================================================================
    # CUDA kernel for P2G with quadratic B-spline
    # =========================================================================
    _p2g_quadratic_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void p2g_quadratic(
        const float* px, const float* py, const float* pq,
        float* q_grid, float* weight_grid,
        int n_particles, int nx, int ny,
        float inv_dx, float inv_dy
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n_particles) return;

        float x_norm = px[p] * inv_dx;
        float y_norm = py[p] * inv_dy;

        int base_i = (int)floorf(x_norm + 0.5f) - 1;
        int base_j = (int)floorf(y_norm + 0.5f) - 1;

        float fx = x_norm - (base_i + 1);
        float fy = y_norm - (base_j + 1);

        float q_p = pq[p];

        // 3x3 stencil for quadratic B-spline
        for (int di = 0; di < 3; di++) {
            float xi = (di - 1) - fx;
            float ax = fabsf(xi);
            float wx;
            if (ax < 0.5f) {
                wx = 0.75f - ax * ax;
            } else if (ax < 1.5f) {
                float t = 1.5f - ax;
                wx = 0.5f * t * t;
            } else {
                wx = 0.0f;
            }

            int gi = (base_i + di + nx) % nx;

            for (int dj = 0; dj < 3; dj++) {
                float yj = (dj - 1) - fy;
                float ay = fabsf(yj);
                float wy;
                if (ay < 0.5f) {
                    wy = 0.75f - ay * ay;
                } else if (ay < 1.5f) {
                    float t = 1.5f - ay;
                    wy = 0.5f * t * t;
                } else {
                    wy = 0.0f;
                }

                int gj = (base_j + dj + ny) % ny;

                float w = wx * wy;
                int idx = gi * ny + gj;

                atomicAdd(&q_grid[idx], w * q_p);
                atomicAdd(&weight_grid[idx], w);
            }
        }
    }
    ''', 'p2g_quadratic')

    # =========================================================================
    # CUDA kernel for G2P with quadratic B-spline
    # =========================================================================
    _g2p_quadratic_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void g2p_quadratic(
        const float* grid_field,
        const float* px, const float* py,
        float* values,
        int n_particles, int nx, int ny,
        float inv_dx, float inv_dy
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n_particles) return;

        float x_norm = px[p] * inv_dx;
        float y_norm = py[p] * inv_dy;

        int base_i = (int)floorf(x_norm + 0.5f) - 1;
        int base_j = (int)floorf(y_norm + 0.5f) - 1;

        float fx = x_norm - (base_i + 1);
        float fy = y_norm - (base_j + 1);

        float val = 0.0f;

        for (int di = 0; di < 3; di++) {
            float xi = (di - 1) - fx;
            float ax = fabsf(xi);
            float wx;
            if (ax < 0.5f) {
                wx = 0.75f - ax * ax;
            } else if (ax < 1.5f) {
                float t = 1.5f - ax;
                wx = 0.5f * t * t;
            } else {
                wx = 0.0f;
            }

            int gi = (base_i + di + nx) % nx;

            for (int dj = 0; dj < 3; dj++) {
                float yj = (dj - 1) - fy;
                float ay = fabsf(yj);
                float wy;
                if (ay < 0.5f) {
                    wy = 0.75f - ay * ay;
                } else if (ay < 1.5f) {
                    float t = 1.5f - ay;
                    wy = 0.5f * t * t;
                } else {
                    wy = 0.0f;
                }

                int gj = (base_j + dj + ny) % ny;
                int idx = gi * ny + gj;

                val += wx * wy * grid_field[idx];
            }
        }

        values[p] = val;
    }
    ''', 'g2p_quadratic')

    # =========================================================================
    # CUDA kernel for Jacobian RHS: dJ/dt = -J * grad_v
    # =========================================================================
    _jacobian_rhs_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void jacobian_rhs(
        const float* J, const float* grad_v, float* dJ,
        int n_particles
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n_particles) return;

        int base = p * 4;
        int gbase = p * 4;

        // J is stored as [J00, J01, J10, J11]
        float J00 = J[base];
        float J01 = J[base + 1];
        float J10 = J[base + 2];
        float J11 = J[base + 3];

        float gv00 = grad_v[gbase];
        float gv01 = grad_v[gbase + 1];
        float gv10 = grad_v[gbase + 2];
        float gv11 = grad_v[gbase + 3];

        // dJ = -J * grad_v
        dJ[base]     = -(J00 * gv00 + J01 * gv10);
        dJ[base + 1] = -(J00 * gv01 + J01 * gv11);
        dJ[base + 2] = -(J10 * gv00 + J11 * gv10);
        dJ[base + 3] = -(J10 * gv01 + J11 * gv11);
    }
    ''', 'jacobian_rhs')

    # =========================================================================
    # CUDA kernel for RK4 position update
    # =========================================================================
    _rk4_position_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void rk4_position(
        const float* x0, const float* y0,
        const float* vx1, const float* vy1,
        const float* vx2, const float* vy2,
        const float* vx3, const float* vy3,
        const float* vx4, const float* vy4,
        float* x_new, float* y_new,
        int n, float dt, float Lx, float Ly
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n) return;

        float dt6 = dt / 6.0f;

        float x = x0[p] + dt6 * (vx1[p] + 2.0f*vx2[p] + 2.0f*vx3[p] + vx4[p]);
        float y = y0[p] + dt6 * (vy1[p] + 2.0f*vy2[p] + 2.0f*vy3[p] + vy4[p]);

        // Periodic wrap
        x = x - Lx * floorf(x / Lx);
        y = y - Ly * floorf(y / Ly);

        x_new[p] = x;
        y_new[p] = y;
    }
    ''', 'rk4_position')


# =========================================================================
# Python wrapper functions
# =========================================================================

def P2G_gpu(px: 'cp.ndarray', py: 'cp.ndarray', pq: 'cp.ndarray',
            nx: int, ny: int, dx: float, dy: float,
            kernel_order: str = 'quadratic') -> 'cp.ndarray':
    """GPU-accelerated P2G transfer.

    Args:
        px, py: Particle positions (CuPy arrays)
        pq: Particle values (CuPy arrays)
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
        kernel_order: 'quadratic' (only supported currently)

    Returns:
        Grid values (CuPy array)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU P2G")

    if kernel_order != 'quadratic':
        raise NotImplementedError(f"GPU kernel only supports 'quadratic', got '{kernel_order}'")

    n_particles = len(px)

    # Ensure float32 for CUDA kernels
    px_f = px.astype(cp.float32)
    py_f = py.astype(cp.float32)
    pq_f = pq.astype(cp.float32)

    q_grid = cp.zeros((nx, ny), dtype=cp.float32)
    weight_grid = cp.zeros((nx, ny), dtype=cp.float32)

    # Launch kernel
    block_size = 256
    grid_size = (n_particles + block_size - 1) // block_size

    _p2g_quadratic_kernel(
        (grid_size,), (block_size,),
        (px_f, py_f, pq_f, q_grid, weight_grid,
         np.int32(n_particles), np.int32(nx), np.int32(ny),
         np.float32(1.0 / dx), np.float32(1.0 / dy))
    )

    # Normalize
    mask = weight_grid > 1e-10
    q_grid[mask] /= weight_grid[mask]

    return q_grid.astype(cp.float64)


def G2P_gpu(grid_field: 'cp.ndarray', px: 'cp.ndarray', py: 'cp.ndarray',
            dx: float, dy: float,
            kernel_order: str = 'quadratic') -> 'cp.ndarray':
    """GPU-accelerated G2P interpolation.

    Args:
        grid_field: Grid values (CuPy array)
        px, py: Particle positions (CuPy arrays)
        dx, dy: Grid spacing
        kernel_order: 'quadratic' (only supported currently)

    Returns:
        Particle values (CuPy array)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU G2P")

    if kernel_order != 'quadratic':
        raise NotImplementedError(f"GPU kernel only supports 'quadratic', got '{kernel_order}'")

    nx, ny = grid_field.shape
    n_particles = len(px)

    # Ensure float32
    field_f = grid_field.astype(cp.float32)
    px_f = px.astype(cp.float32)
    py_f = py.astype(cp.float32)

    values = cp.zeros(n_particles, dtype=cp.float32)

    # Launch kernel
    block_size = 256
    grid_size = (n_particles + block_size - 1) // block_size

    _g2p_quadratic_kernel(
        (grid_size,), (block_size,),
        (field_f, px_f, py_f, values,
         np.int32(n_particles), np.int32(nx), np.int32(ny),
         np.float32(1.0 / dx), np.float32(1.0 / dy))
    )

    return values.astype(cp.float64)


def jacobian_rhs_gpu(J: 'cp.ndarray', grad_v: 'cp.ndarray') -> 'cp.ndarray':
    """GPU-accelerated Jacobian RHS computation.

    Args:
        J: Jacobian matrices (n, 2, 2) CuPy array
        grad_v: Velocity gradient (n, 2, 2) CuPy array

    Returns:
        dJ/dt (n, 2, 2) CuPy array
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n = J.shape[0]

    # Flatten to (n, 4) for kernel
    J_flat = J.reshape(n, 4).astype(cp.float32)
    grad_v_flat = grad_v.reshape(n, 4).astype(cp.float32)
    dJ_flat = cp.zeros((n, 4), dtype=cp.float32)

    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    _jacobian_rhs_kernel(
        (grid_size,), (block_size,),
        (J_flat, grad_v_flat, dJ_flat, np.int32(n))
    )

    return dJ_flat.reshape(n, 2, 2).astype(cp.float64)


def rk4_positions_gpu(x0: 'cp.ndarray', y0: 'cp.ndarray',
                      vx1: 'cp.ndarray', vy1: 'cp.ndarray',
                      vx2: 'cp.ndarray', vy2: 'cp.ndarray',
                      vx3: 'cp.ndarray', vy3: 'cp.ndarray',
                      vx4: 'cp.ndarray', vy4: 'cp.ndarray',
                      dt: float, Lx: float, Ly: float) -> Tuple['cp.ndarray', 'cp.ndarray']:
    """GPU-accelerated RK4 position update.

    Returns:
        (x_new, y_new) CuPy arrays
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    n = len(x0)

    # Convert to float32
    args = [a.astype(cp.float32) for a in [x0, y0, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4]]
    x_new = cp.zeros(n, dtype=cp.float32)
    y_new = cp.zeros(n, dtype=cp.float32)

    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    _rk4_position_kernel(
        (grid_size,), (block_size,),
        (*args, x_new, y_new, np.int32(n),
         np.float32(dt), np.float32(Lx), np.float32(Ly))
    )

    return x_new.astype(cp.float64), y_new.astype(cp.float64)
