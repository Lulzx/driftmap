"""Apple Silicon GPU acceleration using MLX.

Provides Metal-accelerated implementations of:
- P2G (particle to grid) transfers with B-spline kernels
- G2P (grid to particle) interpolation
- Flow map Jacobian evolution
- FFT-based Poisson solver

MLX provides unified memory (no CPU-GPU transfers needed) and
automatic differentiation, making it ideal for Apple Silicon.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


def check_mlx_available() -> bool:
    """Check if MLX is available."""
    return MLX_AVAILABLE


if MLX_AVAILABLE:
    def _fftfreq_mlx(n: int, d: float) -> mx.array:
        """MLX-compatible fftfreq using NumPy for frequency bins."""
        return mx.array(np.fft.fftfreq(n, d))

    # =========================================================================
    # B-spline kernel functions for MLX
    # =========================================================================

    def _quadratic_bspline_mlx(x: mx.array) -> mx.array:
        """Quadratic B-spline kernel (vectorized for MLX)."""
        ax = mx.abs(x)
        # Region 1: |x| < 0.5
        r1 = 0.75 - ax * ax
        # Region 2: 0.5 <= |x| < 1.5
        t = 1.5 - ax
        r2 = 0.5 * t * t
        # Combine
        result = mx.where(ax < 0.5, r1, mx.where(ax < 1.5, r2, mx.zeros_like(x)))
        return result

    def _quadratic_bspline_deriv_mlx(x: mx.array) -> mx.array:
        """Derivative of quadratic B-spline."""
        ax = mx.abs(x)
        sx = mx.sign(x)
        # Region 1: |x| < 0.5
        r1 = -2.0 * x
        # Region 2: 0.5 <= |x| < 1.5
        r2 = -(1.5 - ax) * sx
        result = mx.where(ax < 0.5, r1, mx.where(ax < 1.5, r2, mx.zeros_like(x)))
        return result

    # =========================================================================
    # P2G Transfer using MLX
    # =========================================================================

    def P2G_mlx(px: mx.array, py: mx.array, pq: mx.array,
                nx: int, ny: int, dx: float, dy: float) -> mx.array:
        """MLX-accelerated P2G transfer with quadratic B-spline.

        Uses vectorized scatter-add via mx.at.
        """
        n_particles = px.shape[0]
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        # Normalize positions
        x_norm = px * inv_dx
        y_norm = py * inv_dy

        # Base indices for quadratic stencil
        base_i = mx.floor(x_norm + 0.5).astype(mx.int32) - 1
        base_j = mx.floor(y_norm + 0.5).astype(mx.int32) - 1

        # Fractional positions
        fx = x_norm - (base_i.astype(mx.float32) + 1)
        fy = y_norm - (base_j.astype(mx.float32) + 1)

        # Compute all weights for 3x3 stencil
        di_offsets = mx.array([-1, 0, 1], dtype=mx.float32)
        dj_offsets = mx.array([-1, 0, 1], dtype=mx.float32)

        xi = di_offsets[None, :] - fx[:, None]  # (n, 3)
        yj = dj_offsets[None, :] - fy[:, None]  # (n, 3)
        wx = _quadratic_bspline_mlx(xi)         # (n, 3)
        wy = _quadratic_bspline_mlx(yj)         # (n, 3)

        weights = wx[:, :, None] * wy[:, None, :]  # (n, 3, 3)

        gi = (base_i[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % nx
        gj = (base_j[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % ny

        gi_flat = mx.broadcast_to(gi[:, :, None], (n_particles, 3, 3)).reshape(-1)
        gj_flat = mx.broadcast_to(gj[:, None, :], (n_particles, 3, 3)).reshape(-1)
        weights_flat = weights.reshape(-1)
        pq_flat = mx.broadcast_to(pq[:, None, None], (n_particles, 3, 3)).reshape(-1)

        q_grid = mx.zeros((nx, ny))
        weight_grid = mx.zeros((nx, ny))

        q_grid = q_grid.at[gi_flat, gj_flat].add(weights_flat * pq_flat)
        weight_grid = weight_grid.at[gi_flat, gj_flat].add(weights_flat)

        # Normalize
        safe_weight = mx.where(weight_grid > 1e-10, weight_grid, mx.ones_like(weight_grid))
        q_grid = mx.where(weight_grid > 1e-10, q_grid / safe_weight, q_grid)

        return q_grid

    def P2G_mlx_vectorized(px: mx.array, py: mx.array, pq: mx.array,
                           nx: int, ny: int, dx: float, dy: float) -> mx.array:
        """Alias for P2G_mlx (kept for backwards compatibility)."""
        return P2G_mlx(px, py, pq, nx, ny, dx, dy)

    def P2G_mlx_gradient_enhanced(
        px: mx.array, py: mx.array, pq: mx.array,
        grad_q_x: mx.array, grad_q_y: mx.array,
        nx: int, ny: int, dx: float, dy: float
    ) -> mx.array:
        """Gradient-enhanced P2G with quadratic B-spline."""
        n_particles = px.shape[0]
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        x_norm = px * inv_dx
        y_norm = py * inv_dy

        base_i = mx.floor(x_norm + 0.5).astype(mx.int32) - 1
        base_j = mx.floor(y_norm + 0.5).astype(mx.int32) - 1

        fx = x_norm - (base_i.astype(mx.float32) + 1)
        fy = y_norm - (base_j.astype(mx.float32) + 1)

        di_offsets = mx.array([-1, 0, 1], dtype=mx.float32)
        dj_offsets = mx.array([-1, 0, 1], dtype=mx.float32)

        xi = di_offsets[None, :] - fx[:, None]  # (n, 3)
        yj = dj_offsets[None, :] - fy[:, None]  # (n, 3)
        wx = _quadratic_bspline_mlx(xi)
        wy = _quadratic_bspline_mlx(yj)

        weights = wx[:, :, None] * wy[:, None, :]  # (n, 3, 3)

        delta_x = xi[:, :, None] * dx
        delta_y = yj[:, None, :] * dy
        q_enhanced = (pq[:, None, None] +
                      grad_q_x[:, None, None] * delta_x +
                      grad_q_y[:, None, None] * delta_y)

        gi = (base_i[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % nx
        gj = (base_j[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % ny

        gi_flat = mx.broadcast_to(gi[:, :, None], (n_particles, 3, 3)).reshape(-1)
        gj_flat = mx.broadcast_to(gj[:, None, :], (n_particles, 3, 3)).reshape(-1)
        weights_flat = weights.reshape(-1)
        q_flat = q_enhanced.reshape(-1)

        q_grid = mx.zeros((nx, ny))
        weight_grid = mx.zeros((nx, ny))

        q_grid = q_grid.at[gi_flat, gj_flat].add(weights_flat * q_flat)
        weight_grid = weight_grid.at[gi_flat, gj_flat].add(weights_flat)

        safe_weight = mx.where(weight_grid > 1e-10, weight_grid, mx.ones_like(weight_grid))
        q_grid = mx.where(weight_grid > 1e-10, q_grid / safe_weight, q_grid)

        return q_grid

    # =========================================================================
    # G2P Interpolation using MLX
    # =========================================================================

    def G2P_mlx(grid_field: mx.array, px: mx.array, py: mx.array,
                dx: float, dy: float) -> mx.array:
        """MLX-accelerated G2P interpolation with quadratic B-spline."""
        nx, ny = grid_field.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        x_norm = px * inv_dx
        y_norm = py * inv_dy

        base_i = mx.floor(x_norm + 0.5).astype(mx.int32) - 1
        base_j = mx.floor(y_norm + 0.5).astype(mx.int32) - 1

        fx = x_norm - (base_i.astype(mx.float32) + 1)
        fy = y_norm - (base_j.astype(mx.float32) + 1)

        di_offsets = mx.array([-1, 0, 1], dtype=mx.float32)
        dj_offsets = mx.array([-1, 0, 1], dtype=mx.float32)

        xi = di_offsets[None, :] - fx[:, None]
        yj = dj_offsets[None, :] - fy[:, None]
        wx = _quadratic_bspline_mlx(xi)
        wy = _quadratic_bspline_mlx(yj)

        weights = wx[:, :, None] * wy[:, None, :]

        gi = (base_i[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % nx
        gj = (base_j[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % ny

        flat_idx = gi[:, :, None] * ny + gj[:, None, :]
        grid_flat = grid_field.reshape(-1)
        values = mx.take(grid_flat, flat_idx)

        return mx.sum(weights * values, axis=(1, 2))

    def G2P_mlx_with_gradient(grid_field: mx.array, px: mx.array, py: mx.array,
                              dx: float, dy: float) -> Tuple[mx.array, mx.array, mx.array]:
        """MLX G2P interpolation returning value and gradients."""
        nx, ny = grid_field.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        x_norm = px * inv_dx
        y_norm = py * inv_dy

        base_i = mx.floor(x_norm + 0.5).astype(mx.int32) - 1
        base_j = mx.floor(y_norm + 0.5).astype(mx.int32) - 1

        fx = x_norm - (base_i.astype(mx.float32) + 1)
        fy = y_norm - (base_j.astype(mx.float32) + 1)

        di_offsets = mx.array([-1, 0, 1], dtype=mx.float32)
        dj_offsets = mx.array([-1, 0, 1], dtype=mx.float32)

        xi = di_offsets[None, :] - fx[:, None]
        yj = dj_offsets[None, :] - fy[:, None]
        wx = _quadratic_bspline_mlx(xi)
        wy = _quadratic_bspline_mlx(yj)
        dwx = _quadratic_bspline_deriv_mlx(xi) * inv_dx
        dwy = _quadratic_bspline_deriv_mlx(yj) * inv_dy

        weights = wx[:, :, None] * wy[:, None, :]
        grad_x_w = dwx[:, :, None] * wy[:, None, :]
        grad_y_w = wx[:, :, None] * dwy[:, None, :]

        gi = (base_i[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % nx
        gj = (base_j[:, None] + mx.array([0, 1, 2], dtype=mx.int32)[None, :]) % ny

        flat_idx = gi[:, :, None] * ny + gj[:, None, :]
        grid_flat = grid_field.reshape(-1)
        values = mx.take(grid_flat, flat_idx)

        val = mx.sum(weights * values, axis=(1, 2))
        grad_x = mx.sum(grad_x_w * values, axis=(1, 2))
        grad_y = mx.sum(grad_y_w * values, axis=(1, 2))

        return val, grad_x, grad_y

    # =========================================================================
    # Jacobian RHS using MLX
    # =========================================================================

    def jacobian_rhs_mlx(J: mx.array, grad_v: mx.array) -> mx.array:
        """Compute dJ/dt = -J · ∇v using MLX.

        Args:
            J: Jacobian matrices (n, 2, 2)
            grad_v: Velocity gradient (n, 2, 2)

        Returns:
            dJ/dt (n, 2, 2)
        """
        # Matrix multiply: dJ = -J @ grad_v
        # For batch matrix multiply in MLX
        dJ = -mx.matmul(J, grad_v)
        return dJ

    def jacobian_rhs_3d_mlx(J: mx.array, grad_v: mx.array) -> mx.array:
        """Compute dJ/dt = -J · ∇v for 3D using MLX.

        Args:
            J: Jacobian matrices (n, 3, 3)
            grad_v: Velocity gradient (n, 3, 3)

        Returns:
            dJ/dt (n, 3, 3)
        """
        dJ = -mx.matmul(J, grad_v)
        return dJ

    # =========================================================================
    # RK4 Position Update using MLX
    # =========================================================================

    def rk4_positions_mlx(x0: mx.array, y0: mx.array,
                          vx1: mx.array, vy1: mx.array,
                          vx2: mx.array, vy2: mx.array,
                          vx3: mx.array, vy3: mx.array,
                          vx4: mx.array, vy4: mx.array,
                          dt: float, Lx: float, Ly: float) -> Tuple[mx.array, mx.array]:
        """MLX-accelerated RK4 position update.

        Returns:
            (x_new, y_new) MLX arrays
        """
        dt6 = dt / 6.0

        x_new = x0 + dt6 * (vx1 + 2.0*vx2 + 2.0*vx3 + vx4)
        y_new = y0 + dt6 * (vy1 + 2.0*vy2 + 2.0*vy3 + vy4)

        # Periodic wrap
        x_new = x_new - Lx * mx.floor(x_new / Lx)
        y_new = y_new - Ly * mx.floor(y_new / Ly)

        return x_new, y_new

    # =========================================================================
    # FFT-based Poisson Solver using MLX
    # =========================================================================

    def solve_poisson_mlx(rhs: mx.array, Lx: float, Ly: float,
                          modified: bool = False) -> mx.array:
        """Solve Poisson equation using MLX FFT.

        Solves ∇²φ = rhs (standard) or (∇² - 1)φ = -rhs (modified/HM)

        Args:
            rhs: Right-hand side (vorticity)
            Lx, Ly: Domain size
            modified: If True, solve HM equation (∇² - 1)φ = -rhs

        Returns:
            Solution φ
        """
        nx, ny = rhs.shape

        # Wave numbers
        kx = _fftfreq_mlx(nx, d=Lx / nx) * 2 * mx.pi
        ky = _fftfreq_mlx(ny, d=Ly / ny) * 2 * mx.pi
        KX, KY = mx.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2

        # FFT of RHS
        rhs_hat = mx.fft.fft2(rhs)

        if modified:
            # (∇² - 1)φ = -rhs  =>  φ = rhs / (k² + 1)
            denom = K2 + 1.0
        else:
            # ∇²φ = rhs  =>  φ = -rhs / k²
            denom = K2
            # Avoid division by zero at k=0
            denom = mx.where(mx.abs(denom) < 1e-10, mx.ones_like(denom), denom)

        phi_hat = -rhs_hat / denom
        # Zero out mean without complex scatter (not supported on MLX GPU).
        i = mx.arange(nx)[:, None]
        j = mx.arange(ny)[None, :]
        mask = mx.where((i == 0) & (j == 0), 0.0, 1.0)
        phi_hat = phi_hat * mask

        phi = mx.real(mx.fft.ifft2(phi_hat))

        return phi

    # =========================================================================
    # Velocity computation using MLX FFTs
    # =========================================================================

    def compute_velocity_mlx(phi: mx.array, Lx: float, Ly: float) -> Tuple[mx.array, mx.array]:
        """Compute E×B velocity from potential using MLX FFTs."""
        nx, ny = phi.shape

        kx = _fftfreq_mlx(nx, d=Lx / nx) * 2 * mx.pi
        ky = _fftfreq_mlx(ny, d=Ly / ny) * 2 * mx.pi
        KX, KY = mx.meshgrid(kx, ky, indexing='ij')

        phi_hat = mx.fft.fft2(phi)
        dphi_dx = mx.real(mx.fft.ifft2(1j * KX * phi_hat))
        dphi_dy = mx.real(mx.fft.ifft2(1j * KY * phi_hat))

        vx = -dphi_dy
        vy = dphi_dx
        return vx, vy

    def compute_velocity_gradient_mlx(vx: mx.array, vy: mx.array,
                                      Lx: float, Ly: float) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute velocity gradients using MLX FFTs."""
        nx, ny = vx.shape

        kx = _fftfreq_mlx(nx, d=Lx / nx) * 2 * mx.pi
        ky = _fftfreq_mlx(ny, d=Ly / ny) * 2 * mx.pi
        KX, KY = mx.meshgrid(kx, ky, indexing='ij')

        vx_hat = mx.fft.fft2(vx)
        vy_hat = mx.fft.fft2(vy)

        dvx_dx = mx.real(mx.fft.ifft2(1j * KX * vx_hat))
        dvx_dy = mx.real(mx.fft.ifft2(1j * KY * vx_hat))
        dvy_dx = mx.real(mx.fft.ifft2(1j * KX * vy_hat))
        dvy_dy = mx.real(mx.fft.ifft2(1j * KY * vy_hat))

        return dvx_dx, dvx_dy, dvy_dx, dvy_dy

    # =========================================================================
    # Utility functions
    # =========================================================================

    def to_mlx(arr: np.ndarray) -> mx.array:
        """Convert numpy array to MLX array."""
        return mx.array(arr)

    def to_numpy(arr: mx.array) -> np.ndarray:
        """Convert MLX array to numpy array."""
        return np.array(arr)

    def synchronize():
        """Synchronize MLX operations (evaluate lazy computations)."""
        mx.eval()


# Fallback for when MLX is not available
else:
    def P2G_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def P2G_mlx_gradient_enhanced(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def G2P_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def G2P_mlx_with_gradient(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def jacobian_rhs_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def rk4_positions_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def solve_poisson_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def compute_velocity_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def compute_velocity_gradient_mlx(*args, **kwargs):
        raise RuntimeError("MLX not available")

    def to_mlx(arr):
        raise RuntimeError("MLX not available")

    def to_numpy(arr):
        raise RuntimeError("MLX not available")

    def synchronize():
        pass
