#!/usr/bin/env python3
"""Run Hasegawa-Wakatani turbulence simulation.

This demonstrates the full drift-wave turbulence physics with:
- Resistive coupling between density and vorticity
- Curvature drive (interchange instability)
- Zonal flow generation
- Sheath damping (parallel losses)

The simulation seeds a random perturbation and watches the instability
grow, saturate, and generate turbulent transport.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from vpfm import Simulation
from vpfm.physics.hasegawa_wakatani import hw_random_perturbation
from vpfm.diagnostics.flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics
from scipy.interpolate import RegularGridInterpolator


def run_hw_turbulence(nx=128, n_steps=5000, dt=0.02):
    """Run Hasegawa-Wakatani turbulence simulation.

    Args:
        nx: Grid resolution
        n_steps: Number of timesteps
        dt: Time step
    """
    ny = nx
    Lx, Ly = 40 * np.pi, 40 * np.pi  # Large domain for turbulence

    # Physics parameters (typical HW values)
    alpha = 0.5      # Moderate adiabaticity (not too HM-like)
    kappa = 0.05     # Curvature drive
    mu = 1e-4        # Hyperviscosity
    D = 1e-4         # Density diffusion
    nu_sheath = 0.01 # Sheath damping

    print("=" * 60)
    print("Hasegawa-Wakatani Turbulence Simulation")
    print("=" * 60)
    print(f"Grid: {nx}×{ny}, Domain: {Lx:.1f}×{Ly:.1f} ρ_s")
    print(f"α={alpha}, κ={kappa}, μ={mu:.0e}, ν_sh={nu_sheath}")
    print(f"Steps: {n_steps}, dt: {dt}")
    print()

    # Initialize simulation
    sim = Simulation(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt,
        alpha=alpha, kappa=kappa, mu=mu, D=D, nu_sheath=nu_sheath,
    )

    # Random initial perturbation
    zeta_init = hw_random_perturbation(nx, ny, Lx, Ly, k_peak=3.0, amplitude=0.01)

    # Create interpolator for particle initialization
    x = np.linspace(Lx/(2*nx), Lx - Lx/(2*nx), nx)
    y = np.linspace(Ly/(2*ny), Ly - Ly/(2*ny), ny)
    interp = RegularGridInterpolator((x, y), zeta_init, bounds_error=False, fill_value=0)

    def ic_func(px, py):
        points = np.column_stack([px % Lx, py % Ly])
        return interp(points)

    sim.set_initial_condition_hw(ic_func)

    # Set up virtual probe at mid-domain
    probe = VirtualProbe(x_pos=Lx/2, y_range=(0, Ly), sample_rate=10)

    # Set up blob detector
    blob_detector = BlobDetector(threshold_sigma=2.0, min_size=4)

    # Storage for visualization
    snapshots = {'time': [], 'n': [], 'phi': [], 'zeta': []}
    snapshot_interval = n_steps // 5

    print("Running simulation...")
    print("-" * 60)

    for step in range(n_steps):
        sim.step_hw()

        needs_refresh = (
            step % probe.sample_rate == 0
            or step % 100 == 0
            or step % snapshot_interval == 0
            or step == n_steps - 1
        )
        if needs_refresh:
            sim._refresh_grid_fields_hw()

        # Probe measurement
        if step % probe.sample_rate == 0:
            probe.measure(sim.time, sim.n_grid, sim.grid.vx,
                         sim.grid.x, sim.grid.y)

        # Diagnostics
        if step % 100 == 0 or step == n_steps - 1:
            diag = sim.compute_hw_diagnostics()
            sim.history['time'].append(sim.time)
            sim.history['energy'].append(diag['energy'])
            sim.history['enstrophy'].append(diag['enstrophy'])
            sim.history['density_variance'].append(diag['density_variance'])
            sim.history['particle_flux'].append(diag['particle_flux'])
            sim.history['zonal_energy'].append(diag['zonal_energy'])
            sim.history['max_q'].append(diag['max_vorticity'])
            sim.history['max_jacobian_dev'].append(sim.integrator.estimate_error(sim.flow_map))
            print(f"Step {step:5d}, t={sim.time:6.1f}, "
                  f"E={diag['energy']:.3e}, "
                  f"Γ={diag['particle_flux']:+.2e}, "
                  f"ZF={diag['zonal_energy']:.2e}")

        # Snapshots
        if step % snapshot_interval == 0 or step == n_steps - 1:
            snapshots['time'].append(sim.time)
            snapshots['n'].append(sim.n_grid.copy())
            snapshots['phi'].append(sim.grid.phi.copy())
            snapshots['zeta'].append(sim.grid.q.copy())

    print("-" * 60)
    print("Simulation complete.")
    print()

    # Compute flux statistics
    flux_stats = probe.compute_statistics()
    print("=" * 60)
    print("Flux Statistics (Virtual Probe)")
    print("=" * 60)
    print(f"Mean particle flux:   Γ = {flux_stats.mean:+.3e}")
    print(f"Fluctuation level:    σ = {flux_stats.std:.3e}")
    print(f"Skewness:             S = {flux_stats.skewness:+.2f}")
    print(f"Kurtosis (excess):    K = {flux_stats.kurtosis:+.2f}")
    print(f"Intermittency:        I = {flux_stats.intermittency:.1%}")
    print()

    # Experimental comparison context
    print("Typical experimental values (MAST-U, ASDEX-U):")
    print("  Skewness: 0.5 - 2.0 (positive, bursty outward)")
    print("  Kurtosis: 1 - 10 (heavy tails)")
    print()

    # Detect blobs in final state
    blobs = blob_detector.detect(sim.n_grid, sim.grid.x, sim.grid.y)
    print(f"Detected {len(blobs)} blob structures in final state")
    if blobs:
        sizes = [b['size'] for b in blobs]
        amps = [b['amplitude'] for b in blobs]
        print(f"  Mean blob size: {np.mean(sizes):.2f} ρ_s")
        print(f"  Mean amplitude: {np.mean(amps):.3f}")
    print()

    # Zonal flow analysis
    print("=" * 60)
    print("Zonal Flow Analysis")
    print("=" * 60)
    zf_energy = np.array(sim.history['zonal_energy'])
    total_energy = np.array(sim.history['energy'])
    zf_fraction = zf_energy / np.maximum(total_energy, 1e-10)
    print(f"Final ZF energy fraction: {zf_fraction[-1]:.1%}")
    print(f"Max ZF energy fraction:   {np.max(zf_fraction):.1%}")
    print()

    # --- Plotting ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Row 1: Snapshots of density
    n_snaps = len(snapshots['time'])
    for i, idx in enumerate([0, n_snaps//2, -1]):
        im = axes[0, i].contourf(sim.grid.X, sim.grid.Y,
                                  snapshots['n'][idx], levels=50, cmap='RdBu_r')
        axes[0, i].set_title(f"n (t={snapshots['time'][idx]:.0f})")
        axes[0, i].set_aspect('equal')
        plt.colorbar(im, ax=axes[0, i])

    # Row 2: Time series
    times = np.array(sim.history['time'])

    # Energy and enstrophy
    axes[1, 0].semilogy(times, sim.history['energy'], 'b-', label='Energy')
    axes[1, 0].semilogy(times, sim.history['enstrophy'], 'r-', label='Enstrophy')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('E, Z')
    axes[1, 0].set_title('Energy & Enstrophy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Particle flux
    axes[1, 1].plot(times, sim.history['particle_flux'], 'g-')
    axes[1, 1].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Γ')
    axes[1, 1].set_title('Radial Particle Flux')
    axes[1, 1].grid(True)

    # Zonal flow energy
    axes[1, 2].plot(times, zf_fraction, 'm-')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('ZF / Total')
    axes[1, 2].set_title('Zonal Flow Fraction')
    axes[1, 2].grid(True)

    # Row 3: Final state analysis
    # Zonal flow profile (y-averaged phi)
    zonal_profile = np.mean(sim.grid.phi, axis=1)
    axes[2, 0].plot(sim.grid.x, zonal_profile, 'b-', linewidth=2)
    axes[2, 0].set_xlabel('x (radial)')
    axes[2, 0].set_ylabel('<φ>_y')
    axes[2, 0].set_title('Zonal Flow Profile')
    axes[2, 0].grid(True)

    # Flux time series from probe
    t_probe, flux_probe = probe.get_time_series()
    if len(t_probe) > 0:
        axes[2, 1].plot(t_probe, flux_probe, 'g-', alpha=0.7)
        axes[2, 1].axhline(flux_stats.mean, color='r', linestyle='--', label='Mean')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Γ at probe')
        axes[2, 1].set_title('Probe Flux Signal')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

    # Flux PDF
    all_flux = np.concatenate(probe.flux_series) if probe.flux_series else np.array([0])
    if len(all_flux) > 100:
        axes[2, 2].hist(all_flux, bins=50, density=True, alpha=0.7, color='g')
        axes[2, 2].axvline(0, color='k', linestyle=':')
        axes[2, 2].set_xlabel('Flux')
        axes[2, 2].set_ylabel('PDF')
        axes[2, 2].set_title(f'Flux PDF (S={flux_stats.skewness:.2f})')
        axes[2, 2].grid(True)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hasegawa_wakatani_turbulence.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    run_hw_turbulence(nx=128, n_steps=5000, dt=0.02)
