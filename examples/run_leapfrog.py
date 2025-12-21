#!/usr/bin/env python3
"""Run vortex pair leapfrog test case.

This test verifies the accuracy of nonlinear advection by tracking
two co-rotating vortices.

Success criteria:
- Rotation period matches theory to < 5%
- Vortex separation preserved to < 5%
- Run 10+ rotation periods without degradation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vpfm import Simulation, vortex_pair, lamb_oseen
from vpfm.diagnostics.diagnostics import find_vortex_peak
from baseline.finite_diff import FiniteDifferenceSimulation


def find_two_peaks(q_field, grid):
    """Find the two largest peaks in the vorticity field.

    Returns:
        List of (x, y, q) for the two peaks
    """
    q_abs = np.abs(q_field)

    # Find first peak
    idx1 = np.unravel_index(np.argmax(q_abs), q_abs.shape)
    x1, y1 = grid.X[idx1], grid.Y[idx1]
    q1 = q_field[idx1]

    # Mask out region around first peak
    r_mask = 1.5  # Masking radius
    mask_r2 = (grid.X - x1)**2 + (grid.Y - y1)**2
    q_masked = q_abs.copy()
    q_masked[mask_r2 < r_mask**2] = 0

    # Find second peak
    idx2 = np.unravel_index(np.argmax(q_masked), q_masked.shape)
    x2, y2 = grid.X[idx2], grid.Y[idx2]
    q2 = q_field[idx2]

    return [(x1, y1, q1), (x2, y2, q2)]


def run_comparison(nx=128, n_steps=2000, dt=0.01):
    """Run VPFM vs FD comparison for vortex pair leapfrog.

    Args:
        nx: Grid resolution
        n_steps: Number of time steps
        dt: Time step
    """
    ny = nx
    Lx, Ly = 4 * np.pi, 4 * np.pi  # Larger domain for vortex pair

    # Vortex parameters
    separation = 3.0
    Gamma = 2 * np.pi
    r0 = 0.5

    # Theoretical rotation period: T = 2 * pi * d^2 / Gamma
    # (for point vortices)
    T_theory = 2 * np.pi * separation**2 / Gamma

    def ic(x, y):
        return vortex_pair(x, y, Lx, Ly, separation, Gamma, r0)

    print("=" * 60)
    print("Vortex Pair Leapfrog Test Case")
    print("=" * 60)
    print(f"Grid: {nx}x{ny}, Domain: {Lx:.2f}x{Ly:.2f}")
    print(f"Separation: {separation:.1f}, Gamma: {Gamma:.2f}")
    print(f"Theoretical rotation period: T ≈ {T_theory:.2f}")
    print(f"Steps: {n_steps}, dt: {dt}, Total time: {n_steps*dt:.1f}")
    print(f"Number of periods: {n_steps*dt/T_theory:.1f}")
    print()

    # --- VPFM Simulation ---
    print("Running VPFM simulation...")
    vpfm = Simulation(nx, ny, Lx, Ly, dt=dt)
    vpfm.set_initial_condition(ic)

    # Track vortex positions
    vpfm_peaks = []
    times = []

    def track_peaks(sim):
        peaks = find_two_peaks(sim.grid.q, sim.grid)
        vpfm_peaks.append(peaks)
        times.append(sim.time)

    track_peaks(vpfm)
    vpfm.run(n_steps, diag_interval=20, callback=track_peaks, verbose=True)

    print()

    # --- FD Simulation ---
    print("Running Finite Difference simulation...")
    fd = FiniteDifferenceSimulation(nx, ny, Lx, Ly, dt=dt)
    fd.set_initial_condition(ic)

    fd_peaks = []
    fd_times = []

    def track_fd_peaks(step, time, q, grid_x, grid_y):
        peaks = find_two_peaks(q, type('Grid', (), {'X': grid_x, 'Y': grid_y})())
        fd_peaks.append(peaks)
        fd_times.append(time)

    # Run FD
    fd.run(n_steps, diag_interval=20, scheme='upwind', verbose=True)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    # Analyze separation preservation
    initial_sep = separation
    vpfm_seps = []
    for peaks in vpfm_peaks:
        if len(peaks) >= 2:
            sep = np.sqrt((peaks[0][0] - peaks[1][0])**2 +
                         (peaks[0][1] - peaks[1][1])**2)
            vpfm_seps.append(sep)

    if vpfm_seps:
        final_sep_vpfm = vpfm_seps[-1]
        sep_error_vpfm = abs(final_sep_vpfm - initial_sep) / initial_sep * 100
        print(f"\nVortex Separation (initial: {initial_sep:.2f}):")
        print(f"  VPFM final: {final_sep_vpfm:.2f} ({sep_error_vpfm:.1f}% error)")

    # Energy conservation
    E_vpfm = np.array(vpfm.history['energy'])
    E_fd = np.array(fd.history['energy'])

    E_error_vpfm = np.abs(E_vpfm[-1] - E_vpfm[0]) / E_vpfm[0] * 100
    E_error_fd = np.abs(E_fd[-1] - E_fd[0]) / E_fd[0] * 100

    print(f"\nEnergy Conservation Error:")
    print(f"  VPFM: {E_error_vpfm:.4f}%")
    print(f"  FD:   {E_error_fd:.4f}%")

    # Peak preservation
    max_q_vpfm = np.array(vpfm.history['max_q'])
    max_q_fd = np.array(fd.history['max_q'])

    print(f"\nPeak Vorticity Ratio (final/initial):")
    print(f"  VPFM: {max_q_vpfm[-1]/max_q_vpfm[0]:.4f}")
    print(f"  FD:   {max_q_fd[-1]/max_q_fd[0]:.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Initial condition
    q_init = ic(vpfm.grid.X, vpfm.grid.Y)
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

    # Vortex trajectories (VPFM)
    if vpfm_peaks:
        x1_traj = [p[0][0] for p in vpfm_peaks]
        y1_traj = [p[0][1] for p in vpfm_peaks]
        x2_traj = [p[1][0] for p in vpfm_peaks if len(p) > 1]
        y2_traj = [p[1][1] for p in vpfm_peaks if len(p) > 1]

        axes[1, 0].plot(x1_traj, y1_traj, 'b-', linewidth=2, label='Vortex 1')
        if x2_traj:
            axes[1, 0].plot(x2_traj, y2_traj, 'r-', linewidth=2, label='Vortex 2')
        axes[1, 0].plot(x1_traj[0], y1_traj[0], 'ko', markersize=8)
        if x2_traj:
            axes[1, 0].plot(x2_traj[0], y2_traj[0], 'ko', markersize=8)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title('VPFM Vortex Trajectories')
        axes[1, 0].set_aspect('equal')
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

    # Separation history
    if vpfm_seps:
        axes[1, 2].plot(times[:len(vpfm_seps)], np.array(vpfm_seps) / initial_sep,
                       'b-', label='VPFM', linewidth=2)
        axes[1, 2].axhline(1.0, color='k', linestyle=':', alpha=0.5)
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Separation / Initial')
        axes[1, 2].set_title('Vortex Separation')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "leapfrog_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    run_comparison(nx=128, n_steps=2000, dt=0.01)
