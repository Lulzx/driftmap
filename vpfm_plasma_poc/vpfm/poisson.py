"""FFT-based Poisson solver for VPFM-Plasma.

Solves the Hasegawa-Mima modified Poisson equation:
    (nabla^2 - 1) phi = -q
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def solve_poisson_hm(q_grid: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Solve Hasegawa-Mima Poisson equation using FFT.

    Solves (nabla^2 - 1) phi = -q with periodic boundary conditions.

    In Fourier space: phi_hat(k) = q_hat(k) / (kx^2 + ky^2 + 1)

    Args:
        q_grid: Potential vorticity field (nx, ny)
        Lx: Domain length in x
        Ly: Domain length in y

    Returns:
        Electrostatic potential phi (nx, ny)
    """
    nx, ny = q_grid.shape

    # Wave numbers
    kx = fftfreq(nx, Lx / nx) * 2 * np.pi
    ky = fftfreq(ny, Ly / ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Forward transform
    q_hat = fft2(q_grid)

    # Solve in Fourier space: (nabla^2 - 1) phi = -q
    # => (-k^2 - 1) phi_hat = -q_hat
    # => phi_hat = q_hat / (k^2 + 1)
    K2 = KX**2 + KY**2 + 1  # +1 for Hasegawa-Mima polarization term

    phi_hat = q_hat / K2

    # Zero mean (gauge condition)
    phi_hat[0, 0] = 0

    # Inverse transform
    phi = np.real(ifft2(phi_hat))

    return phi


def solve_poisson_standard(zeta_grid: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Solve standard Poisson equation using FFT.

    Solves nabla^2 phi = zeta with periodic boundary conditions.

    In Fourier space: phi_hat(k) = -zeta_hat(k) / k^2

    Args:
        zeta_grid: Vorticity field (nx, ny)
        Lx: Domain length in x
        Ly: Domain length in y

    Returns:
        Stream function phi (nx, ny)
    """
    nx, ny = zeta_grid.shape

    # Wave numbers
    kx = fftfreq(nx, Lx / nx) * 2 * np.pi
    ky = fftfreq(ny, Ly / ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Forward transform
    zeta_hat = fft2(zeta_grid)

    # Solve: nabla^2 phi = zeta
    # => -k^2 phi_hat = zeta_hat
    # => phi_hat = -zeta_hat / k^2
    K2 = KX**2 + KY**2
    K2[0, 0] = 1  # Avoid division by zero

    phi_hat = -zeta_hat / K2
    phi_hat[0, 0] = 0  # Zero mean

    # Inverse transform
    phi = np.real(ifft2(phi_hat))

    return phi
