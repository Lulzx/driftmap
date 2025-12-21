#!/usr/bin/env python3
"""Reproduce Kadoch et al. (PoP 29, 102301, 2022) spectral scalings."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from vpfm import Simulation
from vpfm.diagnostics.spectral import energy_spectrum, density_flux_spectrum, fit_power_law
from config import CONFIGS


def _make_grid_interpolator(field: np.ndarray, Lx: float, Ly: float):
    """Build a periodic interpolator from a grid field."""
    nx, ny = field.shape
    x = np.linspace(Lx / (2 * nx), Lx - Lx / (2 * nx), nx)
    y = np.linspace(Ly / (2 * ny), Ly - Ly / (2 * ny), ny)
    interp = RegularGridInterpolator((x, y), field, bounds_error=False, fill_value=0.0)

    def ic(px, py):
        points = np.column_stack([px % Lx, py % Ly])
        return interp(points)

    return ic


def run_to_steady_state(config: dict,
                        n_eddy_times: int = 50,
                        seed: int = 42,
                        particles_per_cell: int = 1,
                        reinit_threshold: float = 2.0,
                        max_reinit_steps: int = 200,
                        use_gradient_p2g: bool = True,
                        backend: str = "cpu") -> Simulation:
    """Run to a quasi-stationary state."""
    sim = Simulation(
        nx=config["nx"],
        ny=config["ny"],
        Lx=config["Lx"],
        Ly=config["Ly"],
        dt=config["dt"],
        alpha=config["alpha"],
        kappa=config["kappa"],
        nu=config["nu"],
        D=config["D"],
        modified_hw=config["modified_hw"],
        particles_per_cell=particles_per_cell,
        reinit_threshold=reinit_threshold,
        max_reinit_steps=max_reinit_steps,
        use_gradient_p2g=use_gradient_p2g,
        backend=backend,
    )

    rng = np.random.default_rng(seed)
    zeta_init = rng.normal(scale=0.1, size=(config["nx"], config["ny"]))
    n_init = rng.normal(scale=0.1, size=(config["nx"], config["ny"]))

    zeta_ic = _make_grid_interpolator(zeta_init, config["Lx"], config["Ly"])
    n_ic = _make_grid_interpolator(n_init, config["Lx"], config["Ly"])
    sim.set_initial_condition_hw(zeta_ic, n_ic)

    tau_k = 0.4
    n_steps = int(n_eddy_times * tau_k / config["dt"])
    sim.run_hw(n_steps=n_steps, diag_interval=max(n_steps // 20, 1), verbose=True)
    return sim


def collect_statistics(sim: Simulation, config: dict, n_samples: int = 25, sample_interval: int = 200):
    """Collect time-averaged spectra in the stationary regime."""
    E_k_samples = []
    F_k_samples = []

    for _ in range(n_samples):
        for _ in range(sample_interval):
            sim.step_hw()
        sim._refresh_grid_fields_hw()

        k, E_k = energy_spectrum(sim.grid.vx, sim.grid.vy, config["Lx"], config["Ly"])
        _, F_k = density_flux_spectrum(sim.grid.vx, sim.n_grid, config["Lx"], config["Ly"])

        E_k_samples.append(E_k)
        F_k_samples.append(np.abs(F_k))

    return k, np.mean(E_k_samples, axis=0), np.mean(F_k_samples, axis=0)


def validate_spectra(k, E_k, F_k, config_name: str, k_min: float, k_max: float) -> bool:
    """Check spectral scalings against Kadoch targets."""
    E_slope, _ = fit_power_law(k, E_k, k_min, k_max)
    F_slope, _ = fit_power_law(k, F_k, k_min, k_max)

    print(f"\n=== {config_name} ===")
    print(f"Energy spectrum slope: {E_slope:.2f} (target: -4.0)")
    print(f"Density flux slope:    {F_slope:.2f} (target: -3.5)")

    E_pass = -4.5 < E_slope < -3.5
    F_pass = -4.0 < F_slope < -3.0
    print(f"Energy spectrum: {'PASS' if E_pass else 'FAIL'}")
    print(f"Density flux:    {'PASS' if F_pass else 'FAIL'}")
    return bool(E_pass and F_pass)


def _apply_overrides(config: dict, nx: Optional[int], ny: Optional[int]) -> dict:
    """Return a copy of config with optional nx/ny overrides."""
    updated = dict(config)
    if nx is not None:
        updated["nx"] = nx
    if ny is not None:
        updated["ny"] = ny
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", help="Subset of config names to run")
    parser.add_argument("--n-eddy-times", type=int, default=50)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--particles-per-cell", type=int, default=1)
    parser.add_argument("--reinit-threshold", type=float, default=2.0)
    parser.add_argument("--max-reinit-steps", type=int, default=200)
    parser.add_argument("--no-gradient-p2g", action="store_true")
    parser.add_argument("--k-min", type=float, default=1.0)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "mlx"])
    parser.add_argument("--fast", action="store_true", help="Use a lightweight config for quick checks")
    args = parser.parse_args()

    if args.fast:
        args.n_eddy_times = 5 if args.n_eddy_times == 50 else args.n_eddy_times
        args.samples = 5 if args.samples == 25 else args.samples
        args.sample_interval = 50 if args.sample_interval == 200 else args.sample_interval
        args.nx = 64 if args.nx is None else args.nx
        args.ny = 64 if args.ny is None else args.ny

    results = {}

    config_names = args.configs or list(CONFIGS.keys())
    for name in config_names:
        config = CONFIGS[name]
        config = _apply_overrides(config, args.nx, args.ny)
        print(f"\n{'=' * 60}")
        print(f"Running {name}")
        print(f"{'=' * 60}")

        sim = run_to_steady_state(
            config,
            n_eddy_times=args.n_eddy_times,
            particles_per_cell=args.particles_per_cell,
            reinit_threshold=args.reinit_threshold,
            max_reinit_steps=args.max_reinit_steps,
            use_gradient_p2g=not args.no_gradient_p2g,
            backend=args.backend,
        )
        k, E_k, F_k = collect_statistics(sim, config,
                                         n_samples=args.samples,
                                         sample_interval=args.sample_interval)
        passed = validate_spectra(k, E_k, F_k, name, args.k_min, args.k_max)

        results[name] = {"k": k, "E_k": E_k, "F_k": F_k, "passed": passed}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, data in results.items():
        axes[0].loglog(data["k"], data["E_k"], label=name)
        axes[1].loglog(data["k"], data["F_k"], label=name)

    k_ref = np.array([1.0, 10.0])
    axes[0].loglog(k_ref, 1e-2 * k_ref**(-4.0), "k--", label="k^-4")
    axes[1].loglog(k_ref, 1e-3 * k_ref**(-3.5), "k--", label="k^-7/2")

    axes[0].set_xlabel("k")
    axes[0].set_ylabel("E(k)")
    axes[0].set_title("Energy Spectrum")
    axes[0].legend()

    axes[1].set_xlabel("k")
    axes[1].set_ylabel("F(k)")
    axes[1].set_title("Density Flux Spectrum")
    axes[1].legend()

    output_dir = Path(__file__).resolve().parents[2] / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "kadoch_benchmark_spectra.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
