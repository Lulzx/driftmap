#!/usr/bin/env python3
"""3D VPFM demo.

Runs a short 3D simulation and saves a mid-plane slice to assets/images.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vpfm import Simulation3D, gaussian_blob_3d


def main():
    nx, ny, nz = 32, 32, 16
    Lx = Ly = Lz = 2 * np.pi
    dt = 0.01
    n_steps = 20

    sim = Simulation3D(
        nx=nx,
        ny=ny,
        nz=nz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        dt=dt,
        particles_per_cell=1,
        cs=1.0,
    )

    def ic(x, y, z):
        return gaussian_blob_3d(
            x, y, z,
            x0=Lx / 2, y0=Ly / 2, z0=Lz / 2,
            amplitude=1.0, rx=0.6, ry=0.6, rz=0.6,
        )

    sim.set_initial_condition(ic)
    for _ in range(n_steps):
        sim.advance()

    z_idx = nz // 2
    q_slice = sim.grid.q[:, :, z_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(q_slice.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="RdBu_r")
    ax.set_title("3D Vorticity Slice (z = mid-plane)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="q")

    output_dir = ROOT / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "simulation3d_slice.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
