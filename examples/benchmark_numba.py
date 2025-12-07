#!/usr/bin/env python3
"""Benchmark Numba-optimized VPFM implementation.

Compares performance of key operations:
- P2G transfers (kernels.py)
- G2P transfers (kernels.py)
- Flow map evolution (flow_map.py)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpfm import Simulation, lamb_oseen


def benchmark_simulation_step(nx: int = 128, n_warmup: int = 5, n_runs: int = 50):
    """Benchmark a single simulation step."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Simulation Step (nx={nx})")
    print(f"{'='*60}")

    Lx = Ly = 2 * np.pi
    dt = 0.01

    sim = Simulation(
        nx=nx, ny=nx, Lx=Lx, Ly=Ly, dt=dt,
        kernel_order='quadratic',
        track_hessian=True,
    )

    def ic(x, y):
        return lamb_oseen(x, y, np.pi, np.pi, Gamma=2*np.pi, r0=0.5)

    sim.set_initial_condition(ic)

    n_particles = sim.particles.n_particles
    print(f"Grid: {nx}x{nx}, Particles: {n_particles}")

    # Warmup (includes JIT compilation)
    print(f"\nWarmup ({n_warmup} steps)...")
    t0 = time.perf_counter()
    for _ in range(n_warmup):
        sim.advance()
    warmup_time = time.perf_counter() - t0
    print(f"Warmup time: {warmup_time:.3f}s (includes JIT compilation)")

    # Benchmark
    print(f"\nBenchmark ({n_runs} steps)...")
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sim.advance()
    bench_time = time.perf_counter() - t0

    avg_time = bench_time / n_runs * 1000  # ms
    steps_per_sec = n_runs / bench_time

    print(f"Total time: {bench_time:.3f}s")
    print(f"Average step time: {avg_time:.2f} ms")
    print(f"Steps per second: {steps_per_sec:.1f}")

    return avg_time


def benchmark_hw_step(nx: int = 64, n_warmup: int = 5, n_runs: int = 50):
    """Benchmark Hasegawa-Wakatani step."""
    print(f"\n{'='*60}")
    print(f"Benchmarking HW Step (nx={nx})")
    print(f"{'='*60}")

    Lx = Ly = 20 * np.pi
    dt = 0.02

    sim = Simulation(
        nx=nx, ny=nx, Lx=Lx, Ly=Ly, dt=dt,
        kernel_order='quadratic',
        track_hessian=True,
        alpha=1.0,
        kappa=0.1,
        mu=1e-4,
        D=1e-4,
    )

    def zeta_ic(x, y):
        return lamb_oseen(x, y, 10*np.pi, 10*np.pi, Gamma=0.5, r0=2.0)

    def n_ic(x, y):
        return lamb_oseen(x, y, 10*np.pi + 0.5, 10*np.pi, Gamma=0.4, r0=2.5)

    sim.set_initial_condition_hw(zeta_ic, n_ic)

    n_particles = sim.particles.n_particles
    print(f"Grid: {nx}x{nx}, Particles: {n_particles}")

    # Warmup
    print(f"\nWarmup ({n_warmup} steps)...")
    t0 = time.perf_counter()
    for _ in range(n_warmup):
        sim.step_hw()
    warmup_time = time.perf_counter() - t0
    print(f"Warmup time: {warmup_time:.3f}s (includes JIT compilation)")

    # Benchmark
    print(f"\nBenchmark ({n_runs} steps)...")
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sim.step_hw()
    bench_time = time.perf_counter() - t0

    avg_time = bench_time / n_runs * 1000
    steps_per_sec = n_runs / bench_time

    print(f"Total time: {bench_time:.3f}s")
    print(f"Average step time: {avg_time:.2f} ms")
    print(f"Steps per second: {steps_per_sec:.1f}")

    return avg_time


def benchmark_scaling():
    """Benchmark scaling with grid size."""
    print(f"\n{'='*60}")
    print("Scaling Benchmark")
    print(f"{'='*60}")

    sizes = [32, 64, 128, 256]
    times = []

    for nx in sizes:
        avg_time = benchmark_simulation_step(nx, n_warmup=3, n_runs=20)
        times.append(avg_time)

    print(f"\n{'='*60}")
    print("Scaling Summary")
    print(f"{'='*60}")
    print(f"{'Grid':<10} {'Particles':<12} {'Time (ms)':<12} {'Relative':<10}")
    print("-" * 44)

    base_time = times[0]
    for nx, t in zip(sizes, times):
        n_particles = nx * nx
        relative = t / base_time
        print(f"{nx}x{nx:<6} {n_particles:<12} {t:<12.2f} {relative:.1f}x")


if __name__ == '__main__':
    print("VPFM Numba Performance Benchmark")
    print("=" * 60)

    # Single step benchmarks
    benchmark_simulation_step(nx=64, n_warmup=5, n_runs=50)
    benchmark_simulation_step(nx=128, n_warmup=5, n_runs=50)
    benchmark_hw_step(nx=64, n_warmup=5, n_runs=50)

    # Scaling benchmark
    benchmark_scaling()
