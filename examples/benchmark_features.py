#!/usr/bin/env python3
"""Comprehensive benchmark of VPFM features.

Benchmarks:
1. Standard VPFM vs Finite Difference (conservation, peak preservation)
2. Flow map methods: standard vs dual-scale
3. P2G methods: standard vs gradient-enhanced
4. Time stepping: fixed vs adaptive
5. Hasegawa-Wakatani physics validation

Generates benchmark plots for README.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpfm import Simulation, lamb_oseen, kelvin_helmholtz, random_turbulence
from vpfm.flow_map import DualScaleFlowMapIntegrator, FlowMapIntegrator
from vpfm.grid import Grid
from vpfm.particles import ParticleSystem
from vpfm.poisson import solve_poisson_hm
from vpfm.velocity import compute_velocity, compute_velocity_gradient
from baseline.finite_diff import FiniteDifferenceSimulation


def benchmark_vpfm_vs_fd():
    """Compare VPFM vs Finite Difference on conservation metrics."""
    print("\n" + "=" * 60)
    print("Benchmark 1: VPFM vs Finite Difference")
    print("=" * 60)

    nx, ny = 64, 64
    Lx, Ly = 2 * np.pi, 2 * np.pi
    dt = 0.01
    n_steps = 300

    # VPFM simulation
    print("\nRunning VPFM simulation...")
    vpfm_sim = Simulation(nx, ny, Lx, Ly, dt=dt, kernel_order='quadratic')

    def ic(x, y):
        return lamb_oseen(x, y, Lx/2, Ly/2, Gamma=2*np.pi, r0=0.5)

    vpfm_sim.set_initial_condition(ic)
    vpfm_initial_peak = np.abs(vpfm_sim.grid.q).max()
    vpfm_sim.run(n_steps, diag_interval=10, verbose=False)

    vpfm_energies = np.array(vpfm_sim.history['energy'])
    vpfm_enstrophies = np.array(vpfm_sim.history['enstrophy'])
    vpfm_final_peak = np.abs(vpfm_sim.grid.q).max()

    # FD Upwind simulation
    print("Running FD Upwind simulation...")
    fd_sim = FiniteDifferenceSimulation(nx, ny, Lx, Ly, dt=dt)
    fd_sim.set_initial_condition(ic)
    fd_initial_peak = np.abs(fd_sim.q).max()
    fd_sim.run(n_steps, diag_interval=10, scheme='upwind', verbose=False)

    fd_energies = np.array(fd_sim.history['energy'])
    fd_enstrophies = np.array(fd_sim.history['enstrophy'])
    fd_final_peak = np.abs(fd_sim.q).max()

    # Results
    vpfm_energy_error = abs(vpfm_energies[-1] - vpfm_energies[0]) / vpfm_energies[0] * 100
    vpfm_enstrophy_error = abs(vpfm_enstrophies[-1] - vpfm_enstrophies[0]) / vpfm_enstrophies[0] * 100
    vpfm_peak_ratio = vpfm_final_peak / vpfm_initial_peak * 100

    fd_energy_error = abs(fd_energies[-1] - fd_energies[0]) / fd_energies[0] * 100
    fd_enstrophy_error = abs(fd_enstrophies[-1] - fd_enstrophies[0]) / fd_enstrophies[0] * 100
    fd_peak_ratio = fd_final_peak / fd_initial_peak * 100

    results = {
        'vpfm': {
            'energy_error': vpfm_energy_error,
            'enstrophy_error': vpfm_enstrophy_error,
            'peak_preservation': vpfm_peak_ratio,
            'energies': vpfm_energies,
            'enstrophies': vpfm_enstrophies,
            'times': np.array(vpfm_sim.history['time']),
        },
        'fd': {
            'energy_error': fd_energy_error,
            'enstrophy_error': fd_enstrophy_error,
            'peak_preservation': fd_peak_ratio,
            'energies': fd_energies,
            'enstrophies': fd_enstrophies,
            'times': np.array(fd_sim.history['time']),
        }
    }

    print(f"\nResults (t={n_steps * dt:.1f}):")
    print(f"  {'Metric':<25} {'VPFM':<15} {'FD Upwind':<15} {'VPFM Advantage'}")
    print(f"  {'-'*70}")
    print(f"  {'Peak preservation':<25} {vpfm_peak_ratio:.1f}%{'':<10} {fd_peak_ratio:.1f}%{'':<10} {vpfm_peak_ratio/fd_peak_ratio:.1f}x")
    print(f"  {'Energy error':<25} {vpfm_energy_error:.2f}%{'':<10} {fd_energy_error:.2f}%{'':<10} {fd_energy_error/vpfm_energy_error:.1f}x")
    print(f"  {'Enstrophy error':<25} {vpfm_enstrophy_error:.2f}%{'':<10} {fd_enstrophy_error:.2f}%{'':<10} {fd_enstrophy_error/vpfm_enstrophy_error:.1f}x")

    return results


def benchmark_flow_map_methods():
    """Compare standard vs dual-scale flow map integrators."""
    print("\n" + "=" * 60)
    print("Benchmark 2: Standard vs Dual-Scale Flow Maps")
    print("=" * 60)

    nx, ny = 32, 32
    Lx, Ly = 2 * np.pi, 2 * np.pi
    dt = 0.01
    n_steps = 200

    grid = Grid(nx, ny, Lx, Ly)

    # Setup vortex flow
    x = np.linspace(grid.dx/2, Lx - grid.dx/2, nx)
    y = np.linspace(grid.dy/2, Ly - grid.dy/2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid.q = lamb_oseen(X, Y, Lx/2, Ly/2, Gamma=2*np.pi, r0=0.5)
    grid.phi = solve_poisson_hm(grid.q, Lx, Ly)
    compute_velocity(grid)
    compute_velocity_gradient(grid)

    # Standard flow map
    print("\nTesting standard flow map...")
    particles_std = ParticleSystem.from_grid(grid, particles_per_cell=1)
    integrator_std = FlowMapIntegrator(grid, kernel_order='quadratic', track_hessian=True)
    flow_map_std = integrator_std.initialize_flow_map(particles_std.n_particles)

    std_errors = []
    std_reinits = 0
    for step in range(n_steps):
        integrator_std.step(particles_std, flow_map_std, dt)
        error = integrator_std.estimate_error(flow_map_std)
        std_errors.append(error)
        if integrator_std.should_reinitialize(flow_map_std, threshold=0.5, max_steps=50):
            integrator_std.reinitialize(particles_std, flow_map_std, grid.q)
            std_reinits += 1

    # Dual-scale flow map
    print("Testing dual-scale flow map...")
    grid.phi = solve_poisson_hm(grid.q, Lx, Ly)
    compute_velocity(grid)
    compute_velocity_gradient(grid)

    particles_dual = ParticleSystem.from_grid(grid, particles_per_cell=1)
    integrator_dual = DualScaleFlowMapIntegrator(grid, kernel_order='quadratic',
                                                   n_L=100, n_S=20)
    flow_map_dual = integrator_dual.initialize_flow_map(particles_dual.n_particles)

    dual_errors = []
    dual_short_reinits = 0
    dual_long_reinits = 0
    for step in range(n_steps):
        integrator_dual.step(particles_dual, flow_map_dual, dt)
        error = integrator_dual.estimate_long_error(flow_map_dual)
        dual_errors.append(error)
        if integrator_dual.should_reinit_short(flow_map_dual):
            integrator_dual.reinit_short(particles_dual, flow_map_dual, grid.q)
            dual_short_reinits += 1
        if integrator_dual.should_reinit_long(flow_map_dual):
            integrator_dual.reinit_long(particles_dual, flow_map_dual, grid.q)
            dual_long_reinits += 1

    results = {
        'standard': {
            'errors': std_errors,
            'reinits': std_reinits,
            'avg_error': np.mean(std_errors),
            'max_error': np.max(std_errors),
        },
        'dual_scale': {
            'errors': dual_errors,
            'short_reinits': dual_short_reinits,
            'long_reinits': dual_long_reinits,
            'avg_error': np.mean(dual_errors),
            'max_error': np.max(dual_errors),
        }
    }

    print(f"\nResults ({n_steps} steps):")
    print(f"  {'Method':<20} {'Avg Error':<15} {'Max Error':<15} {'Reinits'}")
    print(f"  {'-'*60}")
    print(f"  {'Standard':<20} {results['standard']['avg_error']:.4f}{'':<10} {results['standard']['max_error']:.4f}{'':<10} {std_reinits}")
    print(f"  {'Dual-Scale':<20} {results['dual_scale']['avg_error']:.4f}{'':<10} {results['dual_scale']['max_error']:.4f}{'':<10} {dual_short_reinits}S/{dual_long_reinits}L")

    return results


def benchmark_adaptive_timestep():
    """Compare fixed vs adaptive time stepping."""
    print("\n" + "=" * 60)
    print("Benchmark 3: Fixed vs Adaptive Time Stepping")
    print("=" * 60)

    nx, ny = 64, 64
    Lx, Ly = 4.0, 4.0
    n_steps = 200

    # Fixed timestep
    print("\nRunning fixed timestep (dt=0.01)...")
    sim_fixed = Simulation(nx, ny, Lx, Ly, dt=0.01,
                           kernel_order='quadratic', adaptive_dt=False)

    def ic(x, y):
        return kelvin_helmholtz(x, y, Lx, Ly, shear_width=0.1, perturbation=0.05)

    sim_fixed.set_initial_condition(ic)

    t0 = time.perf_counter()
    sim_fixed.run(n_steps, diag_interval=20, verbose=False)
    fixed_time = time.perf_counter() - t0

    fixed_energies = np.array(sim_fixed.history['energy'])
    fixed_energy_error = abs(fixed_energies[-1] - fixed_energies[0]) / fixed_energies[0] * 100

    # Adaptive timestep
    print("Running adaptive timestep...")
    sim_adaptive = Simulation(nx, ny, Lx, Ly, dt=0.02,  # Start with larger dt
                              kernel_order='quadratic', adaptive_dt=True, cfl_number=0.5)
    sim_adaptive.set_initial_condition(ic)

    t0 = time.perf_counter()
    sim_adaptive.run(n_steps, diag_interval=20, verbose=False)
    adaptive_time = time.perf_counter() - t0

    adaptive_energies = np.array(sim_adaptive.history['energy'])
    adaptive_energy_error = abs(adaptive_energies[-1] - adaptive_energies[0]) / adaptive_energies[0] * 100

    results = {
        'fixed': {
            'wall_time': fixed_time,
            'energy_error': fixed_energy_error,
            'final_time': sim_fixed.time,
            'energies': fixed_energies,
            'times': np.array(sim_fixed.history['time']),
        },
        'adaptive': {
            'wall_time': adaptive_time,
            'energy_error': adaptive_energy_error,
            'final_time': sim_adaptive.time,
            'energies': adaptive_energies,
            'times': np.array(sim_adaptive.history['time']),
        }
    }

    print(f"\nResults ({n_steps} steps):")
    print(f"  {'Method':<15} {'Wall Time':<15} {'Sim Time':<15} {'Energy Error'}")
    print(f"  {'-'*55}")
    print(f"  {'Fixed dt':<15} {fixed_time:.2f}s{'':<10} {sim_fixed.time:.2f}{'':<10} {fixed_energy_error:.2f}%")
    print(f"  {'Adaptive dt':<15} {adaptive_time:.2f}s{'':<10} {sim_adaptive.time:.2f}{'':<10} {adaptive_energy_error:.2f}%")

    return results


def benchmark_gradient_p2g():
    """Compare standard vs gradient-enhanced P2G."""
    print("\n" + "=" * 60)
    print("Benchmark 4: Standard vs Gradient-Enhanced P2G")
    print("=" * 60)

    nx, ny = 64, 64
    Lx, Ly = 2 * np.pi, 2 * np.pi
    dt = 0.01
    n_steps = 150

    # Standard P2G
    print("\nRunning standard P2G...")
    sim_std = Simulation(nx, ny, Lx, Ly, dt=dt,
                         kernel_order='quadratic', use_gradient_p2g=False)

    def ic(x, y):
        return lamb_oseen(x, y, Lx/2, Ly/2, Gamma=2*np.pi, r0=0.3)

    sim_std.set_initial_condition(ic)
    initial_peak_std = np.abs(sim_std.grid.q).max()

    t0 = time.perf_counter()
    sim_std.run(n_steps, diag_interval=15, verbose=False)
    std_time = time.perf_counter() - t0

    final_peak_std = np.abs(sim_std.grid.q).max()
    std_peak_ratio = final_peak_std / initial_peak_std * 100

    # Gradient-enhanced P2G
    print("Running gradient-enhanced P2G...")
    sim_grad = Simulation(nx, ny, Lx, Ly, dt=dt,
                          kernel_order='quadratic', use_gradient_p2g=True)
    sim_grad.set_initial_condition(ic)
    initial_peak_grad = np.abs(sim_grad.grid.q).max()

    t0 = time.perf_counter()
    sim_grad.run(n_steps, diag_interval=15, verbose=False)
    grad_time = time.perf_counter() - t0

    final_peak_grad = np.abs(sim_grad.grid.q).max()
    grad_peak_ratio = final_peak_grad / initial_peak_grad * 100

    results = {
        'standard': {
            'wall_time': std_time,
            'peak_preservation': std_peak_ratio,
            'energies': np.array(sim_std.history['energy']),
            'times': np.array(sim_std.history['time']),
        },
        'gradient': {
            'wall_time': grad_time,
            'peak_preservation': grad_peak_ratio,
            'energies': np.array(sim_grad.history['energy']),
            'times': np.array(sim_grad.history['time']),
        }
    }

    print(f"\nResults ({n_steps} steps):")
    print(f"  {'Method':<20} {'Peak Preserved':<18} {'Wall Time'}")
    print(f"  {'-'*50}")
    print(f"  {'Standard P2G':<20} {std_peak_ratio:.1f}%{'':<12} {std_time:.2f}s")
    print(f"  {'Gradient P2G':<20} {grad_peak_ratio:.1f}%{'':<12} {grad_time:.2f}s")

    return results


def benchmark_hw_physics():
    """Benchmark Hasegawa-Wakatani physics."""
    print("\n" + "=" * 60)
    print("Benchmark 5: Hasegawa-Wakatani Turbulence")
    print("=" * 60)

    from vpfm.hasegawa_wakatani import HWSimulation, hw_random_perturbation

    nx, ny = 64, 64
    Lx, Ly = 20 * np.pi, 20 * np.pi
    n_steps = 200

    print("\nRunning HW simulation (α=0.5, κ=0.1)...")
    sim = HWSimulation(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=0.05,
        alpha=0.5, kappa=0.1, mu=1e-3, D=1e-3,
        particles_per_cell=1
    )

    perturbation = hw_random_perturbation(nx, ny, Lx, Ly, k_peak=5.0, amplitude=0.1, seed=42)

    def ic(x, y):
        from scipy.interpolate import RegularGridInterpolator
        x_grid = np.linspace(sim.grid.dx/2, Lx - sim.grid.dx/2, nx)
        y_grid = np.linspace(sim.grid.dy/2, Ly - sim.grid.dy/2, ny)
        interp = RegularGridInterpolator((x_grid, y_grid), perturbation,
                                         bounds_error=False, fill_value=0)
        points = np.column_stack([x, y])
        return interp(points)

    sim.set_initial_condition(ic)

    t0 = time.perf_counter()
    sim.run(n_steps, diag_interval=20, verbose=False)
    hw_time = time.perf_counter() - t0

    results = {
        'wall_time': hw_time,
        'times': np.array(sim.history['time']),
        'energy': np.array(sim.history['energy']),
        'enstrophy': np.array(sim.history['enstrophy']),
        'particle_flux': np.array(sim.history['particle_flux']),
        'zonal_energy': np.array(sim.history['zonal_energy']),
        'density_variance': np.array(sim.history['density_variance']),
    }

    print(f"\nResults ({n_steps} steps, t={sim.time:.1f}):")
    print(f"  Wall time: {hw_time:.2f}s")
    print(f"  Final energy: {results['energy'][-1]:.4f}")
    print(f"  Final zonal energy: {results['zonal_energy'][-1]:.4e}")
    print(f"  Mean particle flux: {np.mean(results['particle_flux']):.4e}")

    return results


def generate_plots(results):
    """Generate benchmark plots for README."""
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))

    # 1. VPFM vs FD comparison
    ax1 = fig.add_subplot(2, 3, 1)
    vpfm_data = results['vpfm_vs_fd']['vpfm']
    fd_data = results['vpfm_vs_fd']['fd']
    ax1.plot(vpfm_data['times'], vpfm_data['energies'] / vpfm_data['energies'][0],
             'b-', linewidth=2, label='VPFM')
    ax1.plot(fd_data['times'], fd_data['energies'] / fd_data['energies'][0],
             'r--', linewidth=2, label='FD Upwind')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('E(t) / E(0)')
    ax1.set_title('Energy Conservation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.05)

    # 2. Flow map error comparison
    ax2 = fig.add_subplot(2, 3, 2)
    std_errors = results['flow_maps']['standard']['errors']
    dual_errors = results['flow_maps']['dual_scale']['errors']
    steps = np.arange(len(std_errors))
    ax2.semilogy(steps, std_errors, 'b-', linewidth=1.5, label='Standard', alpha=0.7)
    ax2.semilogy(steps, dual_errors, 'g-', linewidth=1.5, label='Dual-Scale', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Flow Map Error')
    ax2.set_title('Flow Map Methods')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Adaptive timestep energy
    ax3 = fig.add_subplot(2, 3, 3)
    fixed_data = results['adaptive']['fixed']
    adaptive_data = results['adaptive']['adaptive']
    ax3.plot(fixed_data['times'], fixed_data['energies'] / fixed_data['energies'][0],
             'b-', linewidth=2, label='Fixed dt')
    ax3.plot(adaptive_data['times'], adaptive_data['energies'] / adaptive_data['energies'][0],
             'g-', linewidth=2, label='Adaptive dt')
    ax3.set_xlabel('Simulation Time')
    ax3.set_ylabel('E(t) / E(0)')
    ax3.set_title('Adaptive Time Stepping')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. P2G comparison
    ax4 = fig.add_subplot(2, 3, 4)
    std_p2g = results['gradient_p2g']['standard']
    grad_p2g = results['gradient_p2g']['gradient']
    ax4.plot(std_p2g['times'], std_p2g['energies'] / std_p2g['energies'][0],
             'b-', linewidth=2, label='Standard P2G')
    ax4.plot(grad_p2g['times'], grad_p2g['energies'] / grad_p2g['energies'][0],
             'g-', linewidth=2, label='Gradient P2G')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('E(t) / E(0)')
    ax4.set_title('P2G Methods')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. HW Zonal Flow
    ax5 = fig.add_subplot(2, 3, 5)
    hw_data = results['hw_physics']
    ax5.plot(hw_data['times'], hw_data['zonal_energy'], 'b-', linewidth=2)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Zonal Flow Energy')
    ax5.set_title('HW: Zonal Flow Generation')
    ax5.grid(True, alpha=0.3)

    # 6. HW Particle Flux
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(hw_data['times'], hw_data['particle_flux'], 'r-', linewidth=2)
    ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Particle Flux Γ')
    ax6.set_title('HW: Particle Flux')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("Saved: benchmark_results.png")

    # Generate summary bar chart
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Peak preservation comparison
    ax = axes[0]
    methods = ['VPFM', 'FD Upwind']
    peaks = [results['vpfm_vs_fd']['vpfm']['peak_preservation'],
             results['vpfm_vs_fd']['fd']['peak_preservation']]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(methods, peaks, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Peak Preservation (%)')
    ax.set_title('Vortex Peak Preservation')
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Conservation comparison
    ax = axes[1]
    metrics = ['Energy\nError', 'Enstrophy\nError']
    vpfm_vals = [results['vpfm_vs_fd']['vpfm']['energy_error'],
                 results['vpfm_vs_fd']['vpfm']['enstrophy_error']]
    fd_vals = [results['vpfm_vs_fd']['fd']['energy_error'],
               results['vpfm_vs_fd']['fd']['enstrophy_error']]
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, vpfm_vals, width, label='VPFM', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, fd_vals, width, label='FD Upwind', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Error (%)')
    ax.set_title('Conservation Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # P2G comparison
    ax = axes[2]
    methods = ['Standard\nP2G', 'Gradient\nP2G']
    peaks = [results['gradient_p2g']['standard']['peak_preservation'],
             results['gradient_p2g']['gradient']['peak_preservation']]
    colors = ['#3498db', '#9b59b6']
    bars = ax.bar(methods, peaks, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Peak Preservation (%)')
    ax.set_title('P2G Method Comparison')
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: benchmark_summary.png")

    plt.close('all')


def print_summary_table(results):
    """Print markdown summary table for README."""
    print("\n" + "=" * 60)
    print("Summary Table (for README)")
    print("=" * 60)

    vpfm = results['vpfm_vs_fd']['vpfm']
    fd = results['vpfm_vs_fd']['fd']
    std_fm = results['flow_maps']['standard']
    dual_fm = results['flow_maps']['dual_scale']
    std_p2g = results['gradient_p2g']['standard']
    grad_p2g = results['gradient_p2g']['gradient']

    print("""
### Benchmark Results

| Metric | VPFM | FD Upwind | VPFM Advantage |
|--------|------|-----------|----------------|
| Peak preservation | {:.1f}% | {:.1f}% | **{:.1f}x** |
| Energy error | {:.2f}% | {:.2f}% | **{:.1f}x** |
| Enstrophy error | {:.2f}% | {:.2f}% | **{:.1f}x** |

### Flow Map Methods

| Method | Avg Error | Max Error | Reinits |
|--------|-----------|-----------|---------|
| Standard | {:.4f} | {:.4f} | {} |
| Dual-Scale | {:.4f} | {:.4f} | {}S/{}L |

### P2G Transfer Methods

| Method | Peak Preservation |
|--------|-------------------|
| Standard P2G | {:.1f}% |
| Gradient-Enhanced P2G | {:.1f}% |
""".format(
        vpfm['peak_preservation'], fd['peak_preservation'],
        vpfm['peak_preservation'] / fd['peak_preservation'],
        vpfm['energy_error'], fd['energy_error'],
        fd['energy_error'] / vpfm['energy_error'] if vpfm['energy_error'] > 0 else 1,
        vpfm['enstrophy_error'], fd['enstrophy_error'],
        fd['enstrophy_error'] / vpfm['enstrophy_error'] if vpfm['enstrophy_error'] > 0 else 1,
        std_fm['avg_error'], std_fm['max_error'], std_fm['reinits'],
        dual_fm['avg_error'], dual_fm['max_error'],
        dual_fm['short_reinits'], dual_fm['long_reinits'],
        std_p2g['peak_preservation'],
        grad_p2g['peak_preservation'],
    ))


def main():
    print("=" * 60)
    print("VPFM Comprehensive Feature Benchmark")
    print("=" * 60)

    results = {}

    # Run all benchmarks
    results['vpfm_vs_fd'] = benchmark_vpfm_vs_fd()
    results['flow_maps'] = benchmark_flow_map_methods()
    results['adaptive'] = benchmark_adaptive_timestep()
    results['gradient_p2g'] = benchmark_gradient_p2g()
    results['hw_physics'] = benchmark_hw_physics()

    # Generate plots
    generate_plots(results)

    # Print summary
    print_summary_table(results)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
