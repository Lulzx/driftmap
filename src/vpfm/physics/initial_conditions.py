"""Initial condition helpers for VPFM simulations."""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from typing import Optional


def lamb_oseen(x: np.ndarray, y: np.ndarray,
               x0: float, y0: float,
               Gamma: float = 2 * np.pi,
               r0: float = 1.0) -> np.ndarray:
    """Lamb-Oseen (Gaussian) vortex initial condition.

    Args:
        x, y: Coordinate arrays
        x0, y0: Vortex center
        Gamma: Circulation
        r0: Core radius

    Returns:
        Vorticity field zeta = nabla^2(phi)
    """
    r2 = (x - x0)**2 + (y - y0)**2
    zeta = (Gamma / (np.pi * r0**2)) * np.exp(-r2 / r0**2)
    return zeta


def vortex_pair(x: np.ndarray, y: np.ndarray,
                Lx: float, Ly: float,
                separation: float = 3.0,
                Gamma: float = 2 * np.pi,
                r0: float = 1.0) -> np.ndarray:
    """Two co-rotating vortices initial condition.

    Args:
        x, y: Coordinate arrays
        Lx, Ly: Domain size
        separation: Distance between vortex centers
        Gamma: Circulation (same sign for co-rotating)
        r0: Core radius

    Returns:
        Combined vorticity field
    """
    x_center = Lx / 2
    y_center = Ly / 2

    x1 = x_center - separation / 2
    x2 = x_center + separation / 2

    q1 = lamb_oseen(x, y, x1, y_center, Gamma, r0)
    q2 = lamb_oseen(x, y, x2, y_center, Gamma, r0)

    return q1 + q2


def kelvin_helmholtz(x: np.ndarray, y: np.ndarray,
                     Lx: float, Ly: float,
                     shear_width: float = 0.1,
                     perturbation: float = 0.05,
                     k_mode: int = 2) -> np.ndarray:
    """Kelvin-Helmholtz shear layer initial condition.

    Creates a shear layer at y = Ly/2 with a sinusoidal perturbation
    to seed the KH instability.

    Args:
        x, y: Coordinate arrays
        Lx, Ly: Domain size
        shear_width: Width of the shear layer (fraction of Ly)
        perturbation: Amplitude of initial perturbation
        k_mode: Number of wavelengths in x direction

    Returns:
        Vorticity field for KH instability
    """
    y_mid = Ly / 2

    # Shear layer thickness
    delta = shear_width * Ly

    # Base shear layer vorticity: d(tanh)/dy = sech^2
    # v_x = tanh((y - y_mid)/delta) gives vorticity ζ = -∂v_x/∂y
    arg = (y - y_mid) / delta
    zeta_base = -1.0 / (delta * np.cosh(arg)**2)

    # Sinusoidal perturbation in y-position of shear layer
    k_x = 2 * np.pi * k_mode / Lx
    y_pert = y_mid + perturbation * Ly * np.sin(k_x * x)

    # Perturbed vorticity
    arg_pert = (y - y_pert) / delta
    zeta = -1.0 / (delta * np.cosh(arg_pert)**2)

    return zeta


def random_turbulence(nx: int, ny: int,
                      Lx: float, Ly: float,
                      k_peak: float = 5.0,
                      amplitude: float = 0.1,
                      seed: Optional[int] = None) -> np.ndarray:
    """Random turbulent initial condition with specified spectrum.

    Args:
        nx, ny: Grid size
        Lx, Ly: Domain size
        k_peak: Peak wavenumber of energy spectrum
        amplitude: Overall amplitude scaling
        seed: Random seed for reproducibility

    Returns:
        Random vorticity field
    """
    if seed is not None:
        np.random.seed(seed)

    dx = Lx / nx
    dy = Ly / ny

    # Wave numbers
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # Energy spectrum peaked at k_peak
    E_k = K**4 * np.exp(-(K / k_peak)**2)

    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, (nx, ny))

    # Construct in Fourier space
    zeta_hat = np.sqrt(E_k) * np.exp(1j * phases)
    zeta_hat[0, 0] = 0  # Zero mean

    zeta = amplitude * np.real(ifft2(zeta_hat))

    return zeta
