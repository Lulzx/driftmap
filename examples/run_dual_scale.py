#!/usr/bin/env python3
"""Dual-scale flow map example.

Runs a small HM case with dual-scale flow maps enabled and saves a snapshot
plus the flow-map error history to assets/images.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vpfm import Simulation, lamb_oseen


def main():
    nx, ny = 128, 128
    Lx = Ly = 2 * np.pi
    dt = 0.01
    n_steps = 200

    sim = Simulation(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        kernel_order="quadratic",
        track_hessian=True,
        use_gradient_p2g=True,
        use_dual_scale=True,
        dual_scale_nL=100,
        dual_scale_nS=20,
        dual_scale_error_long=3.0,
        dual_scale_error_short=1.0,
    )

    def ic(x, y):
        return lamb_oseen(x, y, Lx / 2, Ly / 2, Gamma=2 * np.pi, r0=0.5)

    sim.set_initial_condition(ic)
    sim.run(n_steps, diag_interval=10, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes

    q = sim.grid.q
    im = ax0.imshow(q.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="RdBu_r")
    ax0.set_title("Dual-Scale Vorticity Snapshot")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    plt.colorbar(im, ax=ax0, label="q")

    times = np.array(sim.history["time"])
    errors = np.array(sim.history["max_jacobian_dev"])
    ax1.plot(times, errors, "k-", linewidth=2)
    ax1.set_title("Flow-Map Error (Dual-Scale)")
    ax1.set_xlabel("time")
    ax1.set_ylabel("max |J - I|")
    ax1.grid(True, alpha=0.3)

    output_dir = ROOT / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dual_scale_example.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
