#!/usr/bin/env python3
"""Run Lamb-Oseen vortex test case.

This test verifies that a single Gaussian vortex doesn't drift
or decay spuriously over time.

Success criteria:
- Vortex centroid drift < 0.1 grid cells over 1000 steps
- Peak vorticity decay < 5%
- Energy conserved to < 1%
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpfm.simulation import Simulation, lamb_oseen
from vpfm.diagnostics import find_vortex_centroid, find_vortex_peak
from baseline.finite_diff import FiniteDifferenceSimulation


def run_comparison(nx=128, n_steps=1000, dt=0.01):
    """Run VPFM vs FD comparison for Lamb-Oseen vortex.

    Args:
        nx: Grid resolution
        n_steps: Number of time steps
        dt: Time step
    """
    ny = nx
    Lx, Ly = 2 * np.pi, 2 * np.pi

    # Vortex parameters
    x0, y0 = Lx / 2, Ly / 2
    Gamma = 2 * np.pi
    r0 = 0.5

    def ic(x, y):
        return lamb_oseen(x, y, x0, y0, Gamma, r0)

    print("=" * 60)
    print("Lamb-Oseen Vortex Test Case")
    print("=" * 60)
    print(f"Grid: {nx}x{ny}, Domain: {Lx:.2f}x{Ly:.2f}")
    print(f"Steps: {n_steps}, dt: {dt}")
    print()

    # --- VPFM Simulation ---
    print("Running VPFM simulation...")
    vpfm = Simulation(nx, ny, Lx, Ly, dt=dt)
    vpfm.set_initial_condition(ic)

    # Store initial state
    q_init = vpfm.grid.q.copy()
    x_init, y_init = find_vortex_centroid(q_init, vpfm.grid)
    peak_init = np.abs(q_init).max()
    E_init = vpfm.history['energy'][-1] if vpfm.history['energy'] else None

    vpfm.run(n_steps, diag_interval=10, verbose=True)

    # Final state
    x_final_vpfm, y_final_vpfm = find_vortex_centroid(vpfm.grid.q, vpfm.grid)
    peak_final_vpfm = np.abs(vpfm.grid.q).max()

    print()

    # --- FD Simulation ---
    print("Running Finite Difference simulation...")
    fd = FiniteDifferenceSimulation(nx, ny, Lx, Ly, dt=dt)
    fd.set_initial_condition(ic)
    fd.run(n_steps, diag_interval=10, scheme='upwind', verbose=True)

    # Final state
    x_final_fd, y_final_fd = find_vortex_centroid(fd.q, vpfm.grid)
    peak_final_fd = np.abs(fd.q).max()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    # Centroid drift
    drift_vpfm = np.sqrt((x_final_vpfm - x_init)**2 + (y_final_vpfm - y_init)**2)
    drift_fd = np.sqrt((x_final_fd - x_init)**2 + (y_final_fd - y_init)**2)

    print(f"\nCentroid Drift (grid cells):")
    print(f"  VPFM: {drift_vpfm / vpfm.grid.dx:.4f}")
    print(f"  FD:   {drift_fd / vpfm.grid.dx:.4f}")

    # Peak preservation
    peak_ratio_vpfm = peak_final_vpfm / peak_init
    peak_ratio_fd = peak_final_fd / peak_init

    print(f"\nPeak Vorticity Ratio (final/initial):")
    print(f"  VPFM: {peak_ratio_vpfm:.4f} ({(1-peak_ratio_vpfm)*100:.1f}% decay)")
    print(f"  FD:   {peak_ratio_fd:.4f} ({(1-peak_ratio_fd)*100:.1f}% decay)")

    # Energy conservation
    E_vpfm = np.array(vpfm.history['energy'])
    E_fd = np.array(fd.history['energy'])

    E_error_vpfm = np.abs(E_vpfm[-1] - E_vpfm[0]) / E_vpfm[0] * 100
    E_error_fd = np.abs(E_fd[-1] - E_fd[0]) / E_fd[0] * 100

    print(f"\nEnergy Conservation Error:")
    print(f"  VPFM: {E_error_vpfm:.4f}%")
    print(f"  FD:   {E_error_fd:.4f}%")

    # Improvement factor
    if peak_ratio_fd > 0:
        improvement = (1 - peak_ratio_fd) / max(1 - peak_ratio_vpfm, 1e-10)
        print(f"\nVPFM improvement factor (peak preservation): {improvement:.1f}x")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Initial condition
    im0 = axes[0, 0].contourf(vpfm.grid.X, vpfm.grid.Y, q_init, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title('Initial q')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0, 0])

    # VPFM final
    im1 = axes[0, 1].contourf(vpfm.grid.X, vpfm.grid.Y, vpfm.grid.q, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title(f'VPFM (t={vpfm.time:.1f})')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 1])

    # FD final
    im2 = axes[0, 2].contourf(vpfm.grid.X, vpfm.grid.Y, fd.q, levels=50, cmap='RdBu_r')
    axes[0, 2].set_title(f'FD Upwind (t={fd.time:.1f})')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 2])

    # Cross-section comparison
    mid_y = ny // 2
    x_line = vpfm.grid.x

    axes[1, 0].plot(x_line, q_init[:, mid_y], 'k-', label='Initial', linewidth=2)
    axes[1, 0].plot(x_line, vpfm.grid.q[:, mid_y], 'b-', label='VPFM', linewidth=2)
    axes[1, 0].plot(x_line, fd.q[:, mid_y], 'r--', label='FD', linewidth=2)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('q')
    axes[1, 0].set_title('Cross-section at y=Ly/2')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

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

    # Max vorticity history
    max_q_vpfm = np.array(vpfm.history['max_q'])
    max_q_fd = np.array(fd.history['max_q'])

    axes[1, 2].plot(t_vpfm, max_q_vpfm / max_q_vpfm[0], 'b-', label='VPFM', linewidth=2)
    axes[1, 2].plot(t_fd, max_q_fd / max_q_fd[0], 'r--', label='FD', linewidth=2)
    axes[1, 2].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('max|q|(t) / max|q|(0)')
    axes[1, 2].set_title('Peak Vorticity Preservation')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('lamb_oseen_comparison.png', dpi=150)
    print(f"\nPlot saved to lamb_oseen_comparison.png")
    plt.show()


if __name__ == '__main__':
    run_comparison(nx=128, n_steps=1000, dt=0.01)
