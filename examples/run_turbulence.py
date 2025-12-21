#!/usr/bin/env python3
"""Run decaying turbulence test case.

This test verifies energy and enstrophy conservation in a
turbulent flow with random initial conditions.

Success criteria:
- Energy conserved to < 1% over 1000 steps
- Enstrophy conserved to < 1% over 1000 steps
- Inverse cascade visible (energy moves to large scales)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vpfm import Simulation, random_turbulence
from vpfm.diagnostics.diagnostics import compute_spectrum
from baseline.finite_diff import FiniteDifferenceSimulation
from scipy.interpolate import RegularGridInterpolator


def run_comparison(nx=128, n_steps=1000, dt=0.005):
    """Run VPFM vs FD comparison for decaying turbulence.

    Args:
        nx: Grid resolution
        n_steps: Number of time steps
        dt: Time step
    """
    ny = nx
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Turbulence parameters
    k_peak = 5.0
    amplitude = 0.2
    seed = 42

    print("=" * 60)
    print("Decaying Turbulence Test Case")
    print("=" * 60)
    print(f"Grid: {nx}x{ny}, Domain: {Lx:.2f}x{Ly:.2f}")
    print(f"Peak wavenumber: k_peak = {k_peak}")
    print(f"Amplitude: {amplitude}")
    print(f"Steps: {n_steps}, dt: {dt}")
    print()

    # Generate initial condition on grid
    q_init = random_turbulence(nx, ny, Lx, Ly, k_peak, amplitude, seed)

    # Create interpolator for VPFM particle initialization
    x = np.linspace(Lx/(2*nx), Lx - Lx/(2*nx), nx)
    y = np.linspace(Ly/(2*ny), Ly - Ly/(2*ny), ny)
    interp = RegularGridInterpolator((x, y), q_init, bounds_error=False, fill_value=0)

    def ic_interp(px, py):
        points = np.column_stack([px, py])
        return interp(points)

    def ic_grid(gx, gy):
        # For FD, return the pre-computed grid
        return q_init

    # --- VPFM Simulation ---
    print("Running VPFM simulation...")
    vpfm = Simulation(nx, ny, Lx, Ly, dt=dt)
    vpfm.set_initial_condition(ic_interp)

    # Store initial spectrum
    k_init, Ek_init = compute_spectrum(q_init, vpfm.grid)

    vpfm.run(n_steps, diag_interval=10, verbose=True)

    # Final spectrum
    k_final, Ek_vpfm = compute_spectrum(vpfm.grid.q, vpfm.grid)

    print()

    # --- FD Simulation ---
    print("Running Finite Difference simulation...")
    fd = FiniteDifferenceSimulation(nx, ny, Lx, Ly, dt=dt)
    fd.set_initial_condition(ic_grid)
    fd.run(n_steps, diag_interval=10, scheme='upwind', verbose=True)

    # Final spectrum
    _, Ek_fd = compute_spectrum(fd.q, vpfm.grid)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    # Conservation metrics
    E_vpfm = np.array(vpfm.history['energy'])
    E_fd = np.array(fd.history['energy'])
    Z_vpfm = np.array(vpfm.history['enstrophy'])
    Z_fd = np.array(fd.history['enstrophy'])

    E_error_vpfm = np.abs(E_vpfm[-1] - E_vpfm[0]) / E_vpfm[0] * 100
    E_error_fd = np.abs(E_fd[-1] - E_fd[0]) / E_fd[0] * 100
    Z_error_vpfm = np.abs(Z_vpfm[-1] - Z_vpfm[0]) / Z_vpfm[0] * 100
    Z_error_fd = np.abs(Z_fd[-1] - Z_fd[0]) / Z_fd[0] * 100

    print(f"\nEnergy Conservation Error:")
    print(f"  VPFM: {E_error_vpfm:.4f}%")
    print(f"  FD:   {E_error_fd:.4f}%")

    print(f"\nEnstrophy Conservation Error:")
    print(f"  VPFM: {Z_error_vpfm:.4f}%")
    print(f"  FD:   {Z_error_fd:.4f}%")

    # Peak preservation
    max_q_vpfm = np.array(vpfm.history['max_q'])
    max_q_fd = np.array(fd.history['max_q'])

    print(f"\nPeak Vorticity Ratio (final/initial):")
    print(f"  VPFM: {max_q_vpfm[-1]/max_q_vpfm[0]:.4f}")
    print(f"  FD:   {max_q_fd[-1]/max_q_fd[0]:.4f}")

    # High-k content preservation
    high_k_mask = k_init > k_peak
    if np.any(high_k_mask):
        highk_init = np.sum(Ek_init[high_k_mask])
        highk_vpfm = np.sum(Ek_vpfm[high_k_mask])
        highk_fd = np.sum(Ek_fd[high_k_mask])

        print(f"\nHigh-k (k > {k_peak}) Content Ratio:")
        print(f"  VPFM: {highk_vpfm/highk_init:.4f}")
        print(f"  FD:   {highk_fd/highk_init:.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Initial condition
    im0 = axes[0, 0].contourf(vpfm.grid.X, vpfm.grid.Y, q_init, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('Initial q')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0, 0])

    # VPFM final
    vmax = np.abs(q_init).max()
    im1 = axes[0, 1].contourf(vpfm.grid.X, vpfm.grid.Y, vpfm.grid.q,
                              levels=np.linspace(-vmax, vmax, 51), cmap='RdBu_r')
    axes[0, 1].set_title(f'VPFM (t={vpfm.time:.1f})')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 1])

    # FD final
    im2 = axes[0, 2].contourf(vpfm.grid.X, vpfm.grid.Y, fd.q,
                              levels=np.linspace(-vmax, vmax, 51), cmap='RdBu_r')
    axes[0, 2].set_title(f'FD Upwind (t={fd.time:.1f})')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 2])

    # Energy spectrum
    axes[1, 0].loglog(k_init, Ek_init, 'k-', label='Initial', linewidth=2)
    axes[1, 0].loglog(k_final, Ek_vpfm, 'b-', label='VPFM', linewidth=2)
    axes[1, 0].loglog(k_final, Ek_fd, 'r--', label='FD', linewidth=2)
    axes[1, 0].set_xlabel('k')
    axes[1, 0].set_ylabel('E(k)')
    axes[1, 0].set_title('Energy Spectrum')
    axes[1, 0].legend()
    axes[1, 0].grid(True, which='both', alpha=0.3)

    # Energy history
    t_vpfm = np.array(vpfm.history['time'])
    t_fd = np.array(fd.history['time'])

    axes[1, 1].plot(t_vpfm, E_vpfm / E_vpfm[0], 'b-', label='VPFM', linewidth=2)
    axes[1, 1].plot(t_fd, E_fd / E_fd[0], 'r--', label='FD', linewidth=2)
    axes[1, 1].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('E(t)/E(0)')
    axes[1, 1].set_title('Energy Conservation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Enstrophy history
    axes[1, 2].plot(t_vpfm, Z_vpfm / Z_vpfm[0], 'b-', label='VPFM', linewidth=2)
    axes[1, 2].plot(t_fd, Z_fd / Z_fd[0], 'r--', label='FD', linewidth=2)
    axes[1, 2].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Z(t)/Z(0)')
    axes[1, 2].set_title('Enstrophy Conservation')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "turbulence_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    run_comparison(nx=128, n_steps=1000, dt=0.005)
