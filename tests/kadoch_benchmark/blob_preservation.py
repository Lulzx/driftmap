#!/usr/bin/env python3
"""Compare blob peak preservation for VPFM vs FD baseline."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vpfm import Simulation, lamb_oseen
from baseline.finite_diff import FiniteDifferenceSimulation


def _run_vpfm(nx: int, n_steps: int, dt: float, sample_interval: int,
             particles_per_cell: int, reinit_threshold: float, max_reinit_steps: int):
    Lx = Ly = 2 * np.pi
    sim = Simulation(
        nx=nx, ny=nx, Lx=Lx, Ly=Ly, dt=dt,
        kernel_order="quadratic",
        particles_per_cell=particles_per_cell,
        reinit_threshold=reinit_threshold,
        max_reinit_steps=max_reinit_steps,
    )

    def ic(x, y):
        return lamb_oseen(x, y, Lx / 2, Ly / 2, Gamma=2 * np.pi, r0=0.5)

    sim.set_initial_condition(ic)
    peaks = [np.max(np.abs(sim.grid.q))]
    times = [0.0]

    for step in range(n_steps):
        sim.advance()
        if (step + 1) % sample_interval == 0 or step == n_steps - 1:
            sim._refresh_grid_fields_hm()
            peaks.append(np.max(np.abs(sim.grid.q)))
            times.append(sim.time)

    return np.asarray(times), np.asarray(peaks)


def _run_fd(nx: int, n_steps: int, dt: float, sample_interval: int, scheme: str):
    Lx = Ly = 2 * np.pi
    sim = FiniteDifferenceSimulation(nx, nx, Lx, Ly, dt=dt)

    def ic(x, y):
        return lamb_oseen(x, y, Lx / 2, Ly / 2, Gamma=2 * np.pi, r0=0.5)

    sim.set_initial_condition(ic)
    peaks = [np.max(np.abs(sim.q))]
    times = [0.0]

    for step in range(n_steps):
        sim.advance(scheme=scheme)
        if (step + 1) % sample_interval == 0 or step == n_steps - 1:
            peaks.append(np.max(np.abs(sim.q)))
            times.append(sim.time)

    return np.asarray(times), np.asarray(peaks)


def _time_to_threshold(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Return the first time where values fall below threshold."""
    mask = values < threshold
    if not np.any(mask):
        return float(times[-1])
    return float(times[np.argmax(mask)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--scheme", choices=["upwind", "central"], default="upwind")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--particles-per-cell", type=int, default=1)
    parser.add_argument("--reinit-threshold", type=float, default=2.0)
    parser.add_argument("--max-reinit-steps", type=int, default=200)
    args = parser.parse_args()

    print("Running blob preservation benchmark...")
    print(f"Grid: {args.nx}x{args.nx}, steps: {args.n_steps}, dt: {args.dt}")
    print(f"FD scheme: {args.scheme}, sample interval: {args.sample_interval}")
    print(f"VPFM ppc: {args.particles_per_cell}, reinit threshold: {args.reinit_threshold}")

    t_vpfm, peaks_vpfm = _run_vpfm(
        args.nx,
        args.n_steps,
        args.dt,
        args.sample_interval,
        args.particles_per_cell,
        args.reinit_threshold,
        args.max_reinit_steps,
    )
    t_fd, peaks_fd = _run_fd(args.nx, args.n_steps, args.dt, args.sample_interval, args.scheme)

    peaks_vpfm_norm = peaks_vpfm / peaks_vpfm[0]
    peaks_fd_norm = peaks_fd / peaks_fd[0]

    t90_vpfm = _time_to_threshold(t_vpfm, peaks_vpfm_norm, args.threshold)
    t90_fd = _time_to_threshold(t_fd, peaks_fd_norm, args.threshold)

    print("\nTime to preservation threshold:")
    print(f"  VPFM: {t90_vpfm:.2f}")
    print(f"  FD:   {t90_fd:.2f}")
    if t90_fd > 0:
        print(f"  Ratio: {t90_vpfm / t90_fd:.2f}x")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_vpfm, peaks_vpfm_norm, "b-", linewidth=2, label="VPFM")
    ax.plot(t_fd, peaks_fd_norm, "r--", linewidth=2, label=f"FD ({args.scheme})")
    ax.axhline(args.threshold, color="gray", linestyle=":", label=f"{args.threshold:.0%} threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Peak / Initial Peak")
    ax.set_title("Blob Peak Preservation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).resolve().parents[2] / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "blob_preservation_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
