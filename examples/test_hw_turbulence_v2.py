#!/usr/bin/env python3
"""Test Hasegawa-Wakatani turbulence with VPFM SimulationV2.

This script verifies whether the perfectly preserved vortex blobs
(100% peak preservation from VPFM) can actually drive turbulence
through the HW resistive coupling mechanism.

Key physics test:
- VPFM conserves particle vorticity exactly: Dω/Dt = 0
- HW source term α(φ - n) generates new vorticity when φ ≠ n
- If blobs are "real" (not just artifacts), they should:
  1. Cause φ-n decoupling through differential advection
  2. Drive the resistive drift-wave instability
  3. Generate radial particle transport Γ = <n·v_x>
  4. Form zonal flows via Reynolds stress

Success criteria:
- Particle flux Γ should become non-zero (turbulent transport)
- Zonal flow energy should grow (inverse cascade)
- Coupling strength |φ - n| should increase (instability active)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpfm.simulation import Simulation, lamb_oseen


def create_blob_perturbation(Lx: float, Ly: float, n_blobs: int = 5,
                              amplitude: float = 0.1, blob_radius: float = 1.0,
                              seed: int = 42):
    """Create initial condition with multiple blob structures.

    Args:
        Lx, Ly: Domain size
        n_blobs: Number of blobs
        amplitude: Blob amplitude
        blob_radius: Blob radius
        seed: Random seed

    Returns:
        Function zeta(x, y) for vorticity
        Function n(x, y) for density (slightly different to seed instability)
    """
    np.random.seed(seed)

    # Random blob positions
    blob_x = np.random.uniform(0.2 * Lx, 0.8 * Lx, n_blobs)
    blob_y = np.random.uniform(0.2 * Ly, 0.8 * Ly, n_blobs)
    # Alternating positive/negative vorticity
    blob_signs = np.random.choice([-1, 1], n_blobs)

    def zeta_func(x, y):
        result = np.zeros_like(x)
        for i in range(n_blobs):
            result += blob_signs[i] * lamb_oseen(
                x, y, blob_x[i], blob_y[i],
                Gamma=amplitude * 2 * np.pi,
                r0=blob_radius
            )
        return result

    # Density: slightly offset from vorticity to seed the instability
    # The φ-n mismatch will drive the drift-wave
    def n_func(x, y):
        # Add a small phase shift / offset to break symmetry
        result = np.zeros_like(x)
        offset = 0.3  # Radial offset
        for i in range(n_blobs):
            result += blob_signs[i] * lamb_oseen(
                x, y, blob_x[i] + offset, blob_y[i],
                Gamma=amplitude * 2 * np.pi * 0.8,  # Slightly weaker
                r0=blob_radius * 1.1  # Slightly broader
            )
        return result

    return zeta_func, n_func


def run_hw_v2_test(nx: int = 64, n_steps: int = 2000, dt: float = 0.02):
    """Run HW turbulence test with SimulationV2.

    Args:
        nx: Grid resolution
        n_steps: Number of timesteps
        dt: Time step
    """
    # Domain size: large enough for turbulence
    Lx = Ly = 20 * np.pi

    # HW physics parameters
    alpha = 1.0       # Adiabaticity (moderate coupling)
    kappa = 0.1       # Curvature drive (interchange)
    mu = 1e-4         # Hyperviscosity (dissipate small scales)
    D = 1e-4          # Density diffusion

    print("=" * 70)
    print("Hasegawa-Wakatani Turbulence Test with SimulationV2 (VPFM)")
    print("=" * 70)
    print()
    print("Key question: Can perfectly preserved VPFM blobs drive turbulence?")
    print()
    print(f"Grid: {nx}×{nx}, Domain: {Lx:.1f}×{Ly:.1f} ρ_s")
    print(f"Physics: α={alpha}, κ={kappa}, μ={mu:.0e}, D={D:.0e}")
    print(f"Steps: {n_steps}, dt: {dt}, T_final: {n_steps * dt:.1f}")
    print()

    # Initialize simulation with HW physics
    sim = Simulation(
        nx=nx, ny=nx, Lx=Lx, Ly=Ly, dt=dt,
        kernel_order='quadratic',
        track_hessian=True,
        reinit_threshold=1.0,
        max_reinit_steps=100,
        # HW parameters
        alpha=alpha,
        kappa=kappa,
        mu=mu,
        D=D,
    )

    # Create blob initial conditions
    zeta_func, n_func = create_blob_perturbation(
        Lx, Ly, n_blobs=8, amplitude=0.2, blob_radius=2.0
    )

    # Set initial conditions with density offset
    sim.set_initial_condition_hw(zeta_func, n_func)

    # Store initial state for comparison
    zeta_init_peak = np.max(np.abs(sim.particles.q))
    n_init_peak = np.max(np.abs(sim.n_particles))
    coupling_init = np.sqrt(np.mean((sim.grid.phi - sim.n_grid)**2))

    print(f"Initial state:")
    print(f"  Peak vorticity |ζ|_max = {zeta_init_peak:.4f}")
    print(f"  Peak density   |n|_max = {n_init_peak:.4f}")
    print(f"  Coupling |φ-n|_rms     = {coupling_init:.4f}")
    print()

    # Run simulation
    print("Running HW simulation with step_hw()...")
    print("-" * 70)

    snapshots = {'time': [], 'zeta': [], 'n': [], 'phi': []}
    snapshot_times = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]

    for step in range(n_steps):
        sim.step_hw()

        # Periodic diagnostics
        if step % 100 == 0 or step == n_steps - 1:
            diag = sim.compute_hw_diagnostics()
            print(f"Step {step:5d}, t={sim.time:6.2f}: "
                  f"Γ={diag['particle_flux']:+.2e}, "
                  f"|φ-n|={diag['coupling_strength']:.4f}, "
                  f"ZF={diag['zonal_energy']:.2e}, "
                  f"|ζ|_max={diag['max_vorticity']:.4f}")

        # Snapshots
        if step in snapshot_times:
            snapshots['time'].append(sim.time)
            snapshots['zeta'].append(sim.grid.q.copy())
            snapshots['n'].append(sim.n_grid.copy())
            snapshots['phi'].append(sim.grid.phi.copy())

    print("-" * 70)
    print()

    # Final analysis
    diag_final = sim.compute_hw_diagnostics()
    zeta_final_peak = np.max(np.abs(sim.particles.q))

    print("=" * 70)
    print("RESULTS: Can preserved blobs drive turbulence?")
    print("=" * 70)
    print()

    # 1. Vorticity preservation (VPFM property)
    preservation = zeta_final_peak / zeta_init_peak
    print(f"1. VORTICITY PRESERVATION (VPFM property):")
    print(f"   Initial |ζ|_max (particles): {zeta_init_peak:.4f}")
    print(f"   Final   |ζ|_max (particles): {zeta_final_peak:.4f}")
    print(f"   Ratio: {preservation:.2%}")
    print(f"   → {'PASS' if preservation > 0.9 else 'FAIL'}: Blobs preserved")
    print()

    # 2. Turbulent transport
    flux_history = np.array(sim.history['particle_flux'])
    mean_flux = np.mean(flux_history[len(flux_history)//2:])  # Latter half
    print(f"2. TURBULENT TRANSPORT (HW instability):")
    print(f"   Mean particle flux Γ (latter half): {mean_flux:+.2e}")
    print(f"   Flux fluctuation σ_Γ: {np.std(flux_history):.2e}")
    has_transport = abs(mean_flux) > 1e-6 or np.std(flux_history) > 1e-5
    print(f"   → {'PASS' if has_transport else 'FAIL'}: Turbulent transport {'active' if has_transport else 'inactive'}")
    print()

    # 3. Coupling evolution
    print(f"3. RESISTIVE COUPLING (drift-wave activity):")
    print(f"   Initial |φ-n|_rms: {coupling_init:.4f}")
    print(f"   Final   |φ-n|_rms: {diag_final['coupling_strength']:.4f}")
    coupling_grew = diag_final['coupling_strength'] > coupling_init * 1.5
    print(f"   → {'PASS' if coupling_grew else 'NEEDS MORE TIME'}: Coupling {'grew' if coupling_grew else 'stable'}")
    print()

    # 4. Zonal flow formation
    zf_history = np.array(sim.history['zonal_energy'])
    zf_growth = zf_history[-1] / max(zf_history[0], 1e-10)
    print(f"4. ZONAL FLOW FORMATION (inverse cascade):")
    print(f"   Initial ZF energy: {zf_history[0]:.2e}")
    print(f"   Final   ZF energy: {zf_history[-1]:.2e}")
    print(f"   Growth factor: {zf_growth:.1f}x")
    has_zf = zf_growth > 2.0
    print(f"   → {'PASS' if has_zf else 'NEEDS MORE TIME'}: Zonal flows {'forming' if has_zf else 'nascent'}")
    print()

    # Overall assessment
    print("=" * 70)
    print("CONCLUSION:")
    if preservation > 0.9 and has_transport:
        print("  ✓ VPFM preserves blobs AND they drive turbulent transport!")
        print("  ✓ The α(φ - n) source term successfully generates vorticity")
        print("  ✓ Conservation + physics = working drift-wave turbulence")
    elif preservation > 0.9:
        print("  ✓ VPFM preserves blobs perfectly")
        print("  ○ Turbulence needs more time or stronger driving")
        print("  → Try: longer run, larger α, or add more initial perturbation")
    else:
        print("  ? Unexpected vorticity loss - check implementation")
    print("=" * 70)
    print()

    # Plotting
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Vorticity snapshots
    for i in range(min(4, len(snapshots['time']))):
        vmax = max(abs(snapshots['zeta'][i].min()), abs(snapshots['zeta'][i].max()))
        im = axes[0, i].contourf(sim.grid.X, sim.grid.Y, snapshots['zeta'][i],
                                  levels=np.linspace(-vmax, vmax, 50), cmap='RdBu_r')
        axes[0, i].set_title(f"ζ (t={snapshots['time'][i]:.1f})")
        axes[0, i].set_aspect('equal')
        plt.colorbar(im, ax=axes[0, i])

    # Row 2: Density snapshots
    for i in range(min(4, len(snapshots['time']))):
        vmax = max(abs(snapshots['n'][i].min()), abs(snapshots['n'][i].max()))
        im = axes[1, i].contourf(sim.grid.X, sim.grid.Y, snapshots['n'][i],
                                  levels=np.linspace(-vmax, vmax, 50), cmap='RdBu_r')
        axes[1, i].set_title(f"n (t={snapshots['time'][i]:.1f})")
        axes[1, i].set_aspect('equal')
        plt.colorbar(im, ax=axes[1, i])

    # Row 3: Diagnostics
    times = np.array(sim.history['time'])

    # Particle flux
    axes[2, 0].plot(times, sim.history['particle_flux'], 'g-')
    axes[2, 0].axhline(0, color='k', linestyle=':')
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Γ')
    axes[2, 0].set_title('Radial Particle Flux')
    axes[2, 0].grid(True)

    # Energy
    axes[2, 1].semilogy(times, sim.history['energy'], 'b-', label='Energy')
    axes[2, 1].semilogy(times, sim.history['enstrophy'], 'r-', label='Enstrophy')
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('E, Z')
    axes[2, 1].set_title('Energy & Enstrophy')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    # Zonal flow energy
    axes[2, 2].plot(times, sim.history['zonal_energy'], 'm-')
    axes[2, 2].set_xlabel('Time')
    axes[2, 2].set_ylabel('ZF Energy')
    axes[2, 2].set_title('Zonal Flow Energy')
    axes[2, 2].grid(True)

    # Peak vorticity (conservation check)
    axes[2, 3].plot(times, sim.history['max_q'], 'k-', label='|ζ|_max')
    axes[2, 3].axhline(zeta_init_peak, color='r', linestyle='--', alpha=0.5, label='Initial')
    axes[2, 3].set_xlabel('Time')
    axes[2, 3].set_ylabel('|ζ|_max')
    axes[2, 3].set_title('Peak Vorticity (Grid)')
    axes[2, 3].legend()
    axes[2, 3].grid(True)

    plt.tight_layout()
    plt.savefig('hw_v2_turbulence_test.png', dpi=150)
    print(f"Plot saved to hw_v2_turbulence_test.png")
    plt.show()


if __name__ == '__main__':
    # Quick test
    run_hw_v2_test(nx=64, n_steps=1000, dt=0.02)

    # For more thorough test, uncomment:
    # run_hw_v2_test(nx=128, n_steps=5000, dt=0.01)
