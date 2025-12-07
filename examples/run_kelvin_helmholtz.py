#!/usr/bin/env python3
"""Kelvin-Helmholtz instability simulation.

This example demonstrates the VPFM method on a classic fluid dynamics
benchmark: the Kelvin-Helmholtz instability arising from a shear layer.

The simulation shows vortex rollup as the instability develops, with
VPFM preserving the vortex structures much better than finite-difference
methods would.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpfm import Simulation, kelvin_helmholtz


def main():
    # Domain parameters
    nx, ny = 128, 128
    Lx, Ly = 4.0, 4.0

    # Time parameters
    dt = 0.01
    n_steps = 500
    diag_interval = 50

    print("=" * 60)
    print("Kelvin-Helmholtz Instability Simulation")
    print("=" * 60)
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: {Lx} x {Ly}")
    print(f"dt: {dt}, steps: {n_steps}")
    print()

    # Create simulation with adaptive timestep
    sim = Simulation(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt,
        kernel_order='quadratic',
        track_hessian=True,
        use_gradient_p2g=True,
        adaptive_dt=True,
        cfl_number=0.5,
    )

    # Set initial condition
    def ic(x, y):
        return kelvin_helmholtz(x, y, Lx, Ly,
                               shear_width=0.05,
                               perturbation=0.02,
                               k_mode=2)

    sim.set_initial_condition(ic)

    # Store snapshots for visualization
    snapshots = [(0.0, sim.grid.q.copy())]

    def callback(sim):
        if len(sim.history['time']) % 2 == 0:  # Every other diagnostic
            snapshots.append((sim.time, sim.grid.q.copy()))

    # Run simulation
    print("Running simulation...")
    sim.run(n_steps, diag_interval=diag_interval, callback=callback, verbose=True)
    print()

    # Results
    print("Results:")
    print(f"  Final time: {sim.time:.2f}")
    print(f"  Total reinitializations: {sim.history['reinit_count']}")

    energies = np.array(sim.history['energy'])
    if len(energies) > 1:
        E0 = energies[0]
        energy_error = np.abs(energies[-1] - E0) / E0 * 100
        print(f"  Energy error: {energy_error:.2f}%")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select snapshots to plot
    n_plots = min(6, len(snapshots))
    indices = np.linspace(0, len(snapshots)-1, n_plots, dtype=int)

    for idx, (ax, snap_idx) in enumerate(zip(axes, indices)):
        t, q = snapshots[snap_idx]
        im = ax.imshow(q.T, origin='lower', extent=[0, Lx, 0, Ly],
                      cmap='RdBu_r', vmin=-q.max(), vmax=q.max())
        ax.set_title(f't = {t:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='Vorticity')

    plt.suptitle('Kelvin-Helmholtz Instability (VPFM)', fontsize=14)
    plt.tight_layout()
    plt.savefig('kelvin_helmholtz.png', dpi=150)
    print("\nPlot saved to kelvin_helmholtz.png")

    # Plot energy evolution
    fig, ax = plt.subplots(figsize=(8, 5))
    times = np.array(sim.history['time'])
    ax.plot(times, energies / energies[0], 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('E(t) / E(0)')
    ax.set_title('Energy Conservation in KH Instability')
    ax.set_ylim(0.9, 1.1)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kelvin_helmholtz_energy.png', dpi=150)
    print("Energy plot saved to kelvin_helmholtz_energy.png")

    plt.show()


if __name__ == '__main__':
    main()
