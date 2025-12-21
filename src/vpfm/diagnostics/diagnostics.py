"""Diagnostics for VPFM-Plasma.

Computes conserved quantities and structure metrics.
"""

import numpy as np
from numpy.fft import fft2, fftfreq
from ..core.grid import Grid


def compute_diagnostics(grid: Grid) -> dict:
    """Compute physical diagnostics from grid fields.

    Args:
        grid: Grid with q, phi fields

    Returns:
        Dictionary with diagnostic values:
        - energy: 0.5 * integral(|nabla(phi)|^2)
        - enstrophy: 0.5 * integral(zeta^2)
        - pot_enstrophy: 0.5 * integral(q^2)
        - max_q: maximum |q|
        - mean_q: mean q
    """
    dx, dy = grid.dx, grid.dy
    dA = dx * dy

    # Energy: E = 0.5 * integral(|nabla(phi)|^2)
    # Use central differences for efficiency
    dphi_dx = (np.roll(grid.phi, -1, axis=0) - np.roll(grid.phi, 1, axis=0)) / (2 * dx)
    dphi_dy = (np.roll(grid.phi, -1, axis=1) - np.roll(grid.phi, 1, axis=1)) / (2 * dy)
    energy = 0.5 * np.sum(dphi_dx**2 + dphi_dy**2) * dA

    # Vorticity: zeta = nabla^2(phi) = q + phi for Hasegawa-Mima
    zeta = grid.q + grid.phi

    # Enstrophy: Z = 0.5 * integral(zeta^2)
    enstrophy = 0.5 * np.sum(zeta**2) * dA

    # Potential enstrophy: Q = 0.5 * integral(q^2)
    pot_enstrophy = 0.5 * np.sum(grid.q**2) * dA

    return {
        'energy': energy,
        'enstrophy': enstrophy,
        'pot_enstrophy': pot_enstrophy,
        'max_q': np.abs(grid.q).max(),
        'mean_q': grid.q.mean(),
    }


def compute_energy_spectral(grid: Grid) -> float:
    """Compute energy using spectral derivatives (more accurate).

    Args:
        grid: Grid with phi field

    Returns:
        Energy = 0.5 * integral(|nabla(phi)|^2)
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dA = dx * dy

    # Wave numbers
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    phi_hat = fft2(grid.phi)

    # |nabla(phi)|^2 in Fourier space: k^2 |phi_hat|^2
    K2 = KX**2 + KY**2
    grad_phi_sq_hat = K2 * np.abs(phi_hat)**2

    # Parseval's theorem: integral = sum of Fourier modes / N^2
    energy = 0.5 * np.sum(grad_phi_sq_hat) / (nx * ny)

    return energy


def find_vortex_centroid(q_field: np.ndarray, grid: Grid) -> tuple:
    """Find the centroid of the vorticity field.

    Useful for tracking vortex drift in test cases.

    Args:
        q_field: Potential vorticity field
        grid: Grid for coordinates

    Returns:
        (x_centroid, y_centroid) position of weighted centroid
    """
    # Use absolute value for centroid calculation
    q_abs = np.abs(q_field)
    total = np.sum(q_abs)

    if total < 1e-10:
        return grid.Lx / 2, grid.Ly / 2

    x_centroid = np.sum(grid.X * q_abs) / total
    y_centroid = np.sum(grid.Y * q_abs) / total

    return x_centroid, y_centroid


def find_vortex_peak(q_field: np.ndarray, grid: Grid) -> tuple:
    """Find the peak location of the vorticity field.

    Args:
        q_field: Potential vorticity field
        grid: Grid for coordinates

    Returns:
        (x_peak, y_peak, q_peak) location and value of maximum
    """
    idx = np.unravel_index(np.argmax(np.abs(q_field)), q_field.shape)
    x_peak = grid.X[idx]
    y_peak = grid.Y[idx]
    q_peak = q_field[idx]

    return x_peak, y_peak, q_peak


def compute_spectrum(field: np.ndarray, grid: Grid) -> tuple:
    """Compute 1D power spectrum of a field.

    Args:
        field: 2D field array
        grid: Grid for wave numbers

    Returns:
        (k, E_k) wave number array and energy spectrum
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    # Wave numbers
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # FFT
    field_hat = fft2(field)
    power = np.abs(field_hat)**2 / (nx * ny)**2

    # Bin by |k|
    k_max = np.max(K)
    n_bins = min(nx, ny) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    E_k = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        E_k[i] = np.sum(power[mask])

    return k_centers, E_k


def compare_structure_preservation(vpfm_q: np.ndarray,
                                   fd_q: np.ndarray,
                                   initial_q: np.ndarray) -> dict:
    """Compare structure preservation between VPFM and FD methods.

    Args:
        vpfm_q: VPFM potential vorticity field
        fd_q: Finite difference potential vorticity field
        initial_q: Initial potential vorticity field

    Returns:
        Dictionary with comparison metrics
    """
    # Correlation with initial condition
    vpfm_corr = np.corrcoef(vpfm_q.flatten(), initial_q.flatten())[0, 1]
    fd_corr = np.corrcoef(fd_q.flatten(), initial_q.flatten())[0, 1]

    # Peak preservation
    initial_peak = np.abs(initial_q).max()
    vpfm_peak = np.abs(vpfm_q).max() / initial_peak if initial_peak > 0 else 1.0
    fd_peak = np.abs(fd_q).max() / initial_peak if initial_peak > 0 else 1.0

    # High-k spectral content
    def high_k_energy(q):
        q_hat = fft2(q)
        nx, ny = q.shape
        kx = fftfreq(nx) * 2 * np.pi
        ky = fftfreq(ny) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        high_k_mask = K > K.max() / 2
        return np.sum(np.abs(q_hat[high_k_mask])**2)

    initial_highk = high_k_energy(initial_q)
    if initial_highk > 1e-10:
        vpfm_highk = high_k_energy(vpfm_q) / initial_highk
        fd_highk = high_k_energy(fd_q) / initial_highk
    else:
        vpfm_highk = 1.0
        fd_highk = 1.0

    return {
        'vpfm_correlation': vpfm_corr,
        'fd_correlation': fd_corr,
        'vpfm_peak_ratio': vpfm_peak,
        'fd_peak_ratio': fd_peak,
        'vpfm_highk_ratio': vpfm_highk,
        'fd_highk_ratio': fd_highk,
    }
