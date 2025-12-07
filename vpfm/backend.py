"""Backend abstraction for CPU/GPU computation.

Provides a unified interface for array operations that can run on:
- CPU: NumPy + Numba
- GPU: CuPy with custom CUDA kernels

Usage:
    from vpfm.backend import get_backend, set_backend

    # Auto-detect GPU
    xp = get_backend()

    # Force CPU
    set_backend('cpu')
    xp = get_backend()

    # Force GPU
    set_backend('gpu')
    xp = get_backend()
"""

import numpy as np
from typing import Literal, Optional

# Global backend state
_BACKEND: Literal['cpu', 'gpu', 'auto'] = 'auto'
_CUPY_AVAILABLE: Optional[bool] = None
_XP = None  # Cached array module


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


def set_backend(backend: Literal['cpu', 'gpu', 'auto']):
    """Set the computation backend.

    Args:
        backend: 'cpu', 'gpu', or 'auto' (default)
    """
    global _BACKEND, _XP
    _BACKEND = backend
    _XP = None  # Clear cache


def get_backend():
    """Get the array module (numpy or cupy) based on current backend setting.

    Returns:
        numpy or cupy module
    """
    global _XP

    if _XP is not None:
        return _XP

    if _BACKEND == 'cpu':
        _XP = np
    elif _BACKEND == 'gpu':
        if not _check_cupy():
            raise RuntimeError("GPU backend requested but CuPy is not available")
        import cupy as cp
        _XP = cp
    else:  # auto
        if _check_cupy():
            import cupy as cp
            _XP = cp
        else:
            _XP = np

    return _XP


def get_backend_name() -> str:
    """Get the name of the current backend."""
    xp = get_backend()
    return 'gpu' if xp.__name__ == 'cupy' else 'cpu'


def to_cpu(arr):
    """Move array to CPU (numpy)."""
    xp = get_backend()
    if xp.__name__ == 'cupy':
        return arr.get()
    return arr


def to_gpu(arr):
    """Move array to GPU (cupy)."""
    if not _check_cupy():
        raise RuntimeError("CuPy is not available")
    import cupy as cp
    if isinstance(arr, cp.ndarray):
        return arr
    return cp.asarray(arr)


def synchronize():
    """Synchronize GPU if using CuPy."""
    xp = get_backend()
    if xp.__name__ == 'cupy':
        xp.cuda.Stream.null.synchronize()


# FFT functions that work on both backends
def fft2(arr):
    """2D FFT that works on both CPU and GPU."""
    xp = get_backend()
    if xp.__name__ == 'cupy':
        return xp.fft.fft2(arr)
    return np.fft.fft2(arr)


def ifft2(arr):
    """2D inverse FFT that works on both CPU and GPU."""
    xp = get_backend()
    if xp.__name__ == 'cupy':
        return xp.fft.ifft2(arr)
    return np.fft.ifft2(arr)


def fftfreq(n, d):
    """FFT frequencies that work on both CPU and GPU."""
    xp = get_backend()
    if xp.__name__ == 'cupy':
        return xp.fft.fftfreq(n, d)
    return np.fft.fftfreq(n, d)


def real(arr):
    """Real part that works on both backends."""
    xp = get_backend()
    return xp.real(arr)
