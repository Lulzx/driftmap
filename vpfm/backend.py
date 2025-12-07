"""Backend abstraction for CPU/GPU computation.

Provides a unified interface for array operations that can run on:
- CPU: NumPy + Numba
- GPU (NVIDIA): CuPy with custom CUDA kernels
- GPU (Apple Silicon): MLX with Metal acceleration

Usage:
    from vpfm.backend import get_backend, set_backend

    # Auto-detect best backend (MLX on Apple Silicon, CuPy on NVIDIA, else CPU)
    xp = get_backend()

    # Force specific backend
    set_backend('cpu')    # NumPy + Numba
    set_backend('cuda')   # CuPy (NVIDIA GPU)
    set_backend('mlx')    # MLX (Apple Silicon)
    set_backend('auto')   # Auto-detect
"""

import numpy as np
import platform
from typing import Literal, Optional

# Global backend state
_BACKEND: Literal['cpu', 'cuda', 'mlx', 'auto'] = 'auto'
_CUPY_AVAILABLE: Optional[bool] = None
_MLX_AVAILABLE: Optional[bool] = None
_XP = None  # Cached array module
_BACKEND_NAME: Optional[str] = None


def _check_cupy() -> bool:
    """Check if CuPy is available."""
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp
            # Try a simple operation to verify GPU is accessible
            cp.array([1.0])
            _CUPY_AVAILABLE = True
        except (ImportError, Exception):
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def _check_mlx() -> bool:
    """Check if MLX is available (Apple Silicon)."""
    global _MLX_AVAILABLE
    if _MLX_AVAILABLE is None:
        try:
            import mlx.core as mx
            # Try a simple operation
            mx.array([1.0])
            _MLX_AVAILABLE = True
        except (ImportError, Exception):
            _MLX_AVAILABLE = False
    return _MLX_AVAILABLE


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def set_backend(backend: Literal['cpu', 'cuda', 'mlx', 'gpu', 'auto']):
    """Set the computation backend.

    Args:
        backend: 'cpu', 'cuda', 'mlx', 'gpu' (alias for auto GPU), or 'auto'
    """
    global _BACKEND, _XP, _BACKEND_NAME
    if backend == 'gpu':
        # 'gpu' is an alias - pick the best available GPU backend
        backend = 'auto'
    _BACKEND = backend
    _XP = None  # Clear cache
    _BACKEND_NAME = None


def get_backend():
    """Get the array module based on current backend setting.

    Returns:
        numpy, cupy, or mlx.core module
    """
    global _XP, _BACKEND_NAME

    if _XP is not None:
        return _XP

    if _BACKEND == 'cpu':
        _XP = np
        _BACKEND_NAME = 'cpu'
    elif _BACKEND == 'cuda':
        if not _check_cupy():
            raise RuntimeError("CUDA backend requested but CuPy is not available")
        import cupy as cp
        _XP = cp
        _BACKEND_NAME = 'cuda'
    elif _BACKEND == 'mlx':
        if not _check_mlx():
            raise RuntimeError("MLX backend requested but MLX is not available")
        import mlx.core as mx
        _XP = mx
        _BACKEND_NAME = 'mlx'
    else:  # auto
        # Priority: MLX (Apple Silicon) > CuPy (NVIDIA) > NumPy (CPU)
        if _is_apple_silicon() and _check_mlx():
            import mlx.core as mx
            _XP = mx
            _BACKEND_NAME = 'mlx'
        elif _check_cupy():
            import cupy as cp
            _XP = cp
            _BACKEND_NAME = 'cuda'
        else:
            _XP = np
            _BACKEND_NAME = 'cpu'

    return _XP


def get_backend_name() -> str:
    """Get the name of the current backend."""
    global _BACKEND_NAME
    if _BACKEND_NAME is None:
        get_backend()  # Initialize
    return _BACKEND_NAME


def is_gpu_backend() -> bool:
    """Check if current backend is GPU-accelerated."""
    name = get_backend_name()
    return name in ('cuda', 'mlx')


def to_cpu(arr):
    """Move array to CPU (numpy)."""
    name = get_backend_name()
    if name == 'cuda':
        return arr.get()
    elif name == 'mlx':
        import mlx.core as mx
        return np.array(arr)
    return arr


def to_gpu(arr):
    """Move array to GPU (cupy or mlx depending on platform)."""
    name = get_backend_name()
    if name == 'cuda':
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)
    elif name == 'mlx':
        import mlx.core as mx
        if isinstance(arr, mx.array):
            return arr
        return mx.array(arr)
    else:
        raise RuntimeError("No GPU backend available")


def synchronize():
    """Synchronize GPU operations."""
    name = get_backend_name()
    if name == 'cuda':
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif name == 'mlx':
        import mlx.core as mx
        mx.eval()


# FFT functions that work on all backends
def fft2(arr):
    """2D FFT that works on CPU and GPU."""
    name = get_backend_name()
    if name == 'cuda':
        import cupy as cp
        return cp.fft.fft2(arr)
    elif name == 'mlx':
        import mlx.core as mx
        return mx.fft.fft2(arr)
    return np.fft.fft2(arr)


def ifft2(arr):
    """2D inverse FFT that works on CPU and GPU."""
    name = get_backend_name()
    if name == 'cuda':
        import cupy as cp
        return cp.fft.ifft2(arr)
    elif name == 'mlx':
        import mlx.core as mx
        return mx.fft.ifft2(arr)
    return np.fft.ifft2(arr)


def fftfreq(n, d):
    """FFT frequencies that work on all backends."""
    name = get_backend_name()
    if name == 'cuda':
        import cupy as cp
        return cp.fft.fftfreq(n, d)
    elif name == 'mlx':
        import mlx.core as mx
        # MLX uses similar API
        return mx.fft.fftfreq(n, d=d)
    return np.fft.fftfreq(n, d)


def real(arr):
    """Real part that works on all backends."""
    xp = get_backend()
    return xp.real(arr)


def zeros(shape, dtype=None):
    """Create zeros array on current backend."""
    xp = get_backend()
    return xp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None):
    """Create ones array on current backend."""
    xp = get_backend()
    return xp.ones(shape, dtype=dtype)


def array(data, dtype=None):
    """Create array on current backend."""
    xp = get_backend()
    return xp.array(data, dtype=dtype)
