"""Spectral diagnostics for HW turbulence validation."""

import numpy as np


def _radial_bins(nx: int, ny: int, Lx: float, Ly: float) -> tuple:
    """Build isotropic radial bins and k magnitude grid."""
    kx = np.fft.fftfreq(nx, Lx / nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, Ly / ny) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    n_bins = min(nx, ny) // 2
    k_max = np.max(K)
    k_bins = np.linspace(0.0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    return K, k_bins, k_centers


def energy_spectrum(vx: np.ndarray, vy: np.ndarray, Lx: float, Ly: float) -> tuple:
    """Compute isotropic kinetic energy spectrum E(k).

    Args:
        vx, vy: Velocity components on grid
        Lx, Ly: Domain sizes

    Returns:
        (k_centers, E_k)
    """
    nx, ny = vx.shape
    vx_hat = np.fft.fft2(vx)
    vy_hat = np.fft.fft2(vy)

    # Parseval normalization for unnormalized FFT
    energy_2d = 0.5 * (np.abs(vx_hat)**2 + np.abs(vy_hat)**2) / (nx * ny)**2

    K, k_bins, k_centers = _radial_bins(nx, ny, Lx, Ly)
    E_k = np.zeros_like(k_centers)
    for i in range(len(k_centers)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        E_k[i] = np.sum(energy_2d[mask])

    return k_centers, E_k


def density_flux_spectrum(vx: np.ndarray, n: np.ndarray, Lx: float, Ly: float) -> tuple:
    """Compute isotropic density flux co-spectrum F(k).

    Args:
        vx: Radial velocity component
        n: Density field
        Lx, Ly: Domain sizes

    Returns:
        (k_centers, F_k)
    """
    nx, ny = vx.shape
    vx_hat = np.fft.fft2(vx)
    n_hat = np.fft.fft2(n)

    cospec_2d = 0.5 * np.real(vx_hat * np.conj(n_hat)) / (nx * ny)**2

    K, k_bins, k_centers = _radial_bins(nx, ny, Lx, Ly)
    F_k = np.zeros_like(k_centers)
    for i in range(len(k_centers)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        F_k[i] = np.sum(cospec_2d[mask])

    return k_centers, F_k


def fit_power_law(k: np.ndarray, spectrum: np.ndarray, k_min: float, k_max: float) -> tuple:
    """Fit a power law spectrum ~ k^slope over [k_min, k_max]."""
    mask = (k >= k_min) & (k <= k_max) & (spectrum > 0.0)
    if not np.any(mask):
        return np.nan, np.nan
    log_k = np.log(k[mask])
    log_s = np.log(spectrum[mask])
    slope, intercept = np.polyfit(log_k, log_s, 1)
    return slope, intercept
