"""Velocity field computation for VPFM-Plasma.

Computes E×B velocity from electrostatic potential using spectral derivatives.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from .grid import Grid


def compute_velocity(grid: Grid):
    """Compute E×B velocity from potential using spectral derivatives.

    The E×B velocity is:
        vx = -d(phi)/dy
        vy = +d(phi)/dx

    This ensures the velocity field is exactly divergence-free.

    Args:
        grid: Grid with phi field set. Updates vx, vy in place.
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    # Wave numbers
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # FFT of potential
    phi_hat = fft2(grid.phi)

    # Spectral derivatives: d/dx -> i*kx, d/dy -> i*ky
    dphi_dx_hat = 1j * KX * phi_hat
    dphi_dy_hat = 1j * KY * phi_hat

    dphi_dx = np.real(ifft2(dphi_dx_hat))
    dphi_dy = np.real(ifft2(dphi_dy_hat))

    # E×B velocity: v = z_hat × nabla(phi)
    grid.vx = -dphi_dy  # -d(phi)/dy
    grid.vy = dphi_dx   # +d(phi)/dx


def compute_velocity_gradient(grid: Grid):
    """Compute velocity gradient tensor using spectral derivatives.

    Computes:
        dvx/dx, dvx/dy
        dvy/dx, dvy/dy

    These are needed for Jacobian evolution: dJ/dt = -J · nabla(v)

    Args:
        grid: Grid with vx, vy fields set. Updates gradient fields in place.
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    # Wave numbers
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # FFT of velocity components
    vx_hat = fft2(grid.vx)
    vy_hat = fft2(grid.vy)

    # Spectral derivatives
    grid.dvx_dx = np.real(ifft2(1j * KX * vx_hat))
    grid.dvx_dy = np.real(ifft2(1j * KY * vx_hat))
    grid.dvy_dx = np.real(ifft2(1j * KX * vy_hat))
    grid.dvy_dy = np.real(ifft2(1j * KY * vy_hat))


def compute_velocity_central_diff(grid: Grid):
    """Compute E×B velocity using central differences (alternative).

    Less accurate than spectral but faster for small grids.

    Args:
        grid: Grid with phi field set. Updates vx, vy in place.
    """
    dx, dy = grid.dx, grid.dy

    # Central differences with periodic BC (using numpy roll)
    dphi_dx = (np.roll(grid.phi, -1, axis=0) - np.roll(grid.phi, 1, axis=0)) / (2 * dx)
    dphi_dy = (np.roll(grid.phi, -1, axis=1) - np.roll(grid.phi, 1, axis=1)) / (2 * dy)

    grid.vx = -dphi_dy
    grid.vy = dphi_dx


def compute_velocity_gradient_central_diff(grid: Grid):
    """Compute velocity gradient using central differences.

    Args:
        grid: Grid with vx, vy fields set. Updates gradient fields in place.
    """
    dx, dy = grid.dx, grid.dy

    # Central differences
    grid.dvx_dx = (np.roll(grid.vx, -1, axis=0) - np.roll(grid.vx, 1, axis=0)) / (2 * dx)
    grid.dvx_dy = (np.roll(grid.vx, -1, axis=1) - np.roll(grid.vx, 1, axis=1)) / (2 * dy)
    grid.dvy_dx = (np.roll(grid.vy, -1, axis=0) - np.roll(grid.vy, 1, axis=0)) / (2 * dx)
    grid.dvy_dy = (np.roll(grid.vy, -1, axis=1) - np.roll(grid.vy, 1, axis=1)) / (2 * dy)


def max_velocity(grid: Grid) -> float:
    """Get maximum velocity magnitude on grid.

    Args:
        grid: Grid with velocity fields

    Returns:
        Maximum |v| on the grid
    """
    v_mag = np.sqrt(grid.vx**2 + grid.vy**2)
    return np.max(v_mag)
