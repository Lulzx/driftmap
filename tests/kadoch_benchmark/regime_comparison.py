#!/usr/bin/env python3
"""Find regimes where VPFM outperforms Arakawa."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vpfm import Simulation, random_turbulence
from vpfm.diagnostics.diagnostics import compute_diagnostics
from vpfm.diagnostics.flow_topology import compute_weiss_field, compute_residence_times
from baseline.finite_diff import FiniteDifferenceSimulation


def _bilinear_sample(field, x, y, Lx, Ly):
    field = np.asarray(field)
    x = np.asarray(x) % Lx
    y = np.asarray(y) % Ly
    nx, ny = field.shape
    dx = Lx / nx
    dy = Ly / ny

    x_idx = x / dx
    y_idx = y / dy
    i = np.floor(x_idx).astype(int) % nx
    j = np.floor(y_idx).astype(int) % ny
    fx = x_idx - np.floor(x_idx)
    fy = y_idx - np.floor(y_idx)
    ip1 = (i + 1) % nx
    jp1 = (j + 1) % ny

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    return (w00 * field[i, j] +
            w10 * field[ip1, j] +
            w01 * field[i, jp1] +
            w11 * field[ip1, jp1])


def _grid_interpolator(field, Lx, Ly):
    def ic(x, y):
        return _bilinear_sample(field, x, y, Lx, Ly)
    return ic


def _estimate_eddy_time(grid):
    diag = compute_diagnostics(grid)
    area = grid.Lx * grid.Ly
    zeta_rms = np.sqrt(2.0 * diag["enstrophy"] / area)
    return 1.0 / max(zeta_rms, 1e-8)


def _central_gradients(field, dx, dy):
    dfx = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dx)
    dfy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dy)
    return dfx, dfy


def _advect_positions_rk4(x, y, vx, vy, Lx, Ly, dt):
    k1x = _bilinear_sample(vx, x, y, Lx, Ly)
    k1y = _bilinear_sample(vy, x, y, Lx, Ly)

    x2 = (x + 0.5 * dt * k1x) % Lx
    y2 = (y + 0.5 * dt * k1y) % Ly
    k2x = _bilinear_sample(vx, x2, y2, Lx, Ly)
    k2y = _bilinear_sample(vy, x2, y2, Lx, Ly)

    x3 = (x + 0.5 * dt * k2x) % Lx
    y3 = (y + 0.5 * dt * k2y) % Ly
    k3x = _bilinear_sample(vx, x3, y3, Lx, Ly)
    k3y = _bilinear_sample(vy, x3, y3, Lx, Ly)

    x4 = (x + dt * k3x) % Lx
    y4 = (y + dt * k3y) % Ly
    k4x = _bilinear_sample(vx, x4, y4, Lx, Ly)
    k4y = _bilinear_sample(vy, x4, y4, Lx, Ly)

    x_new = (x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)) % Lx
    y_new = (y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)) % Ly
    return x_new, y_new


def _gaussian_blob(x, y, x0, y0, amplitude, sigma):
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma ** 2))


def _advance_vpfm_with_shear(sim, shear_rate, shear_profile):
    sim._p2g()
    phi_mx = sim._solve_poisson_hm()
    sim._compute_velocity_fields(phi_mx)

    if shear_rate != 0.0:
        sim.grid.vy = sim.grid.vy + shear_rate * shear_profile
        sim.grid.dvy_dx = sim.grid.dvy_dx + shear_rate

    dt_step = sim.compute_adaptive_dt() if sim.adaptive_dt else sim.dt
    if sim.use_dual_scale:
        sim.integrator.step(sim.particles, sim.flow_map, dt_step)
        if sim.use_gradient_p2g:
            sim.integrator.evolve_particle_gradients_rk4(sim.particles, dt_step)
    else:
        sim.integrator.step(
            sim.particles,
            sim.flow_map,
            dt_step,
            update_gradients=sim.use_gradient_p2g,
        )

    sim.time += dt_step
    sim.step += 1
    sim._maybe_reinitialize_hm()


def _fd_rhs_with_shear(sim, shear_rate, shear_profile):
    dq_dt = sim._arakawa_advection()
    if shear_rate != 0.0:
        _, dq_dy = _central_gradients(sim.q, sim.dx, sim.dy)
        dq_dt -= shear_rate * shear_profile * dq_dy
    if sim.nu > 0.0:
        dq_dt = dq_dt + sim.nu * sim._laplacian(sim.q)
    return dq_dt


def _advance_fd_with_shear(sim, shear_rate, shear_profile):
    sim._solve_poisson()
    sim._compute_velocity()

    k1 = _fd_rhs_with_shear(sim, shear_rate, shear_profile)
    q_temp = sim.q + 0.5 * sim.dt * k1

    sim.q = q_temp
    sim._solve_poisson()
    sim._compute_velocity()

    k2 = _fd_rhs_with_shear(sim, shear_rate, shear_profile)
    sim.q = sim.q - 0.5 * sim.dt * k1 + sim.dt * k2

    sim._solve_poisson()
    sim._compute_velocity()
    sim.time += sim.dt
    sim.step += 1


def _run_until_time(sim, total_time, advance_fn):
    while sim.time < total_time - 1e-12:
        advance_fn()


def _time_to_threshold(sim, total_time, sample_dt, advance_fn, refresh_fn, peak_fn, threshold):
    next_sample = 0.0
    t_hit = None
    while sim.time < total_time - 1e-12:
        advance_fn()
        if sim.time + 1e-12 >= next_sample or sim.time >= total_time:
            refresh_fn()
            if t_hit is None and peak_fn() < threshold:
                t_hit = sim.time
            next_sample += sample_dt
    return total_time if t_hit is None else t_hit


def long_time_drift(cfg):
    nx = cfg["nx"]
    Lx = Ly = cfg["Lx"]
    dt_vpfm = cfg["dt"]
    dt_fd = cfg["dt"] * cfg["fd_dt_scale"]

    q0 = random_turbulence(nx, nx, Lx, Ly, k_peak=cfg["k_peak"], amplitude=cfg["amplitude"], seed=cfg["seed"])
    ic = _grid_interpolator(q0, Lx, Ly)

    vpfm = Simulation(
        nx=nx,
        ny=nx,
        Lx=Lx,
        Ly=Ly,
        dt=dt_vpfm,
        particles_per_cell=cfg["particles_per_cell"],
        reinit_threshold=cfg["reinit_threshold"],
        max_reinit_steps=cfg["max_reinit_steps"],
        backend=cfg["backend"],
    )
    vpfm.set_initial_condition(ic)
    vpfm._refresh_grid_fields_hm()
    tau = _estimate_eddy_time(vpfm.grid)
    total_time = cfg["long_time_eddy"] * tau
    sample_dt = total_time / max(cfg["long_time_samples"], 1)

    diag0_v = compute_diagnostics(vpfm.grid)
    times_v = [0.0]
    drift_v = [0.0]
    energy_v = [diag0_v["energy"]]
    enstrophy_v = [diag0_v["enstrophy"]]
    next_sample = sample_dt

    while vpfm.time < total_time - 1e-12:
        vpfm.advance()
        if vpfm.time + 1e-12 >= next_sample or vpfm.time >= total_time:
            vpfm._refresh_grid_fields_hm()
            diag = compute_diagnostics(vpfm.grid)
            e_err = abs(diag["energy"] - diag0_v["energy"]) / (abs(diag0_v["energy"]) + 1e-12)
            z_err = abs(diag["enstrophy"] - diag0_v["enstrophy"]) / (abs(diag0_v["enstrophy"]) + 1e-12)
            drift_v.append(0.5 * (e_err + z_err))
            times_v.append(vpfm.time)
            energy_v.append(diag["energy"])
            enstrophy_v.append(diag["enstrophy"])
            next_sample += sample_dt

    fd = FiniteDifferenceSimulation(nx, nx, Lx, Ly, dt=dt_fd, nu=cfg["fd_nu"])
    fd.set_initial_condition(lambda x, y: q0)
    diag0_fd = compute_diagnostics(_fd_grid_from_sim(fd))

    times_fd = [0.0]
    drift_fd = [0.0]
    energy_fd = [diag0_fd["energy"]]
    enstrophy_fd = [diag0_fd["enstrophy"]]
    next_sample = sample_dt

    while fd.time < total_time - 1e-12:
        fd.advance(scheme="arakawa")
        if fd.time + 1e-12 >= next_sample or fd.time >= total_time:
            diag = compute_diagnostics(_fd_grid_from_sim(fd))
            e_err = abs(diag["energy"] - diag0_fd["energy"]) / (abs(diag0_fd["energy"]) + 1e-12)
            z_err = abs(diag["enstrophy"] - diag0_fd["enstrophy"]) / (abs(diag0_fd["enstrophy"]) + 1e-12)
            drift_fd.append(0.5 * (e_err + z_err))
            times_fd.append(fd.time)
            energy_fd.append(diag["energy"])
            enstrophy_fd.append(diag["enstrophy"])
            next_sample += sample_dt

    mean_v = float(np.mean(drift_v))
    mean_fd = float(np.mean(drift_fd))
    advantage = mean_fd / mean_v if mean_v > 1e-12 else float("inf")

    return {
        "tau_eddy": float(tau),
        "total_time": float(total_time),
        "vpfm": {
            "times": _as_list(times_v),
            "drift": _as_list(drift_v),
            "energy": _as_list(energy_v),
            "enstrophy": _as_list(enstrophy_v),
            "mean_drift": mean_v,
        },
        "arakawa": {
            "times": _as_list(times_fd),
            "drift": _as_list(drift_fd),
            "energy": _as_list(energy_fd),
            "enstrophy": _as_list(enstrophy_fd),
            "mean_drift": mean_fd,
        },
        "advantage": float(advantage),
    }


def sharp_blob_preservation(cfg):
    nx = cfg["nx"]
    Lx = Ly = cfg["Lx"]
    dt_vpfm = cfg["dt"]
    dt_fd = cfg["dt"] * cfg["fd_dt_scale"]
    sigmas = cfg["blob_sigmas"]

    results = {"sigmas": _as_list(sigmas), "vpfm_peaks": [], "arakawa_peaks": []}
    for sigma in sigmas:
        x = (np.arange(nx) + 0.5) * (Lx / nx)
        y = (np.arange(nx) + 0.5) * (Ly / nx)
        X, Y = np.meshgrid(x, y, indexing="ij")
        q0 = _gaussian_blob(X, Y, Lx / 2, Ly / 2, cfg["blob_amplitude"], sigma)
        ic = _grid_interpolator(q0, Lx, Ly)
        initial_peak = float(np.max(np.abs(q0)))

        vpfm = Simulation(
            nx=nx,
            ny=nx,
            Lx=Lx,
            Ly=Ly,
            dt=dt_vpfm,
            particles_per_cell=cfg["particles_per_cell"],
            reinit_threshold=cfg["reinit_threshold"],
            max_reinit_steps=cfg["max_reinit_steps"],
            backend=cfg["backend"],
        )
        vpfm.set_initial_condition(ic)
        vpfm._refresh_grid_fields_hm()
        tau = _estimate_eddy_time(vpfm.grid)
        total_time = cfg["blob_eddy"] * tau

        _run_until_time(vpfm, total_time, vpfm.advance)
        vpfm._refresh_grid_fields_hm()
        vpfm_peak = float(np.max(np.abs(vpfm.grid.q)) / initial_peak)
        results["vpfm_peaks"].append(vpfm_peak)

        fd = FiniteDifferenceSimulation(nx, nx, Lx, Ly, dt=dt_fd, nu=cfg["fd_nu"])
        fd.set_initial_condition(lambda x, y: q0)
        _run_until_time(fd, total_time, lambda: fd.advance(scheme="arakawa"))
        fd_peak = float(np.max(np.abs(fd.q)) / initial_peak)
        results["arakawa_peaks"].append(fd_peak)

    vpfm_mean = float(np.mean(results["vpfm_peaks"]))
    fd_mean = float(np.mean(results["arakawa_peaks"]))
    advantage = vpfm_mean / fd_mean if fd_mean > 1e-12 else float("inf")
    results["vpfm_mean_peak"] = vpfm_mean
    results["arakawa_mean_peak"] = fd_mean
    results["advantage"] = float(advantage)
    return results


def blob_in_shear(cfg):
    nx = cfg["nx"]
    Lx = Ly = cfg["Lx"]
    dt_vpfm = cfg["dt"]
    dt_fd = cfg["dt"] * cfg["fd_dt_scale"]
    shear_rates = cfg["shear_rates"]

    x = (np.arange(nx) + 0.5) * (Lx / nx)
    y = (np.arange(nx) + 0.5) * (Ly / nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    q0 = _gaussian_blob(X, Y, Lx / 2, Ly / 2, cfg["blob_amplitude"], cfg["shear_blob_sigma"])
    ic = _grid_interpolator(q0, Lx, Ly)
    initial_peak = float(np.max(np.abs(q0)))

    results = {"shear_rates": _as_list(shear_rates), "vpfm_t50": [], "arakawa_t50": []}
    for shear_rate in shear_rates:
        vpfm = Simulation(
            nx=nx,
            ny=nx,
            Lx=Lx,
            Ly=Ly,
            dt=dt_vpfm,
            particles_per_cell=cfg["particles_per_cell"],
            reinit_threshold=cfg["reinit_threshold"],
            max_reinit_steps=cfg["max_reinit_steps"],
            backend=cfg["backend"],
        )
        vpfm.set_initial_condition(ic)
        vpfm._refresh_grid_fields_hm()
        tau = _estimate_eddy_time(vpfm.grid)
        total_time = cfg["shear_eddy"] * tau
        sample_dt = total_time / max(cfg["shear_samples"], 1)
        shear_profile = vpfm.grid.X - 0.5 * vpfm.grid.Lx

        def vpfm_advance():
            _advance_vpfm_with_shear(vpfm, shear_rate, shear_profile)

        vpfm_t50 = _time_to_threshold(
            vpfm,
            total_time,
            sample_dt,
            vpfm_advance,
            vpfm._refresh_grid_fields_hm,
            lambda: np.max(np.abs(vpfm.grid.q)) / initial_peak,
            cfg["shear_threshold"],
        )
        results["vpfm_t50"].append(float(vpfm_t50))

        fd = FiniteDifferenceSimulation(nx, nx, Lx, Ly, dt=dt_fd, nu=cfg["fd_nu"])
        fd.set_initial_condition(lambda x, y: q0)
        shear_profile_fd = fd.X - 0.5 * fd.Lx

        def fd_advance():
            _advance_fd_with_shear(fd, shear_rate, shear_profile_fd)

        vfunc = lambda: np.max(np.abs(fd.q)) / initial_peak
        fd_t50 = _time_to_threshold(
            fd,
            total_time,
            sample_dt,
            fd_advance,
            lambda: None,
            vfunc,
            cfg["shear_threshold"],
        )
        results["arakawa_t50"].append(float(fd_t50))

    vpfm_mean = float(np.mean(results["vpfm_t50"]))
    fd_mean = float(np.mean(results["arakawa_t50"]))
    advantage = vpfm_mean / fd_mean if fd_mean > 1e-12 else float("inf")
    results["vpfm_mean_t50"] = vpfm_mean
    results["arakawa_mean_t50"] = fd_mean
    results["advantage"] = float(advantage)
    return results


def lagrangian_statistics(cfg):
    nx = cfg["nx"]
    Lx = Ly = cfg["Lx"]
    dt_vpfm = cfg["dt"]
    dt_fd = cfg["dt"] * cfg["fd_dt_scale"]
    q0 = random_turbulence(nx, nx, Lx, Ly, k_peak=cfg["k_peak"], amplitude=cfg["amplitude"], seed=cfg["seed"])
    ic = _grid_interpolator(q0, Lx, Ly)

    vpfm = Simulation(
        nx=nx,
        ny=nx,
        Lx=Lx,
        Ly=Ly,
        dt=dt_vpfm,
        particles_per_cell=cfg["particles_per_cell"],
        reinit_threshold=cfg["reinit_threshold"],
        max_reinit_steps=cfg["max_reinit_steps"],
        backend=cfg["backend"],
    )
    vpfm.set_initial_condition(ic)
    vpfm._refresh_grid_fields_hm()
    tau = _estimate_eddy_time(vpfm.grid)

    fd = FiniteDifferenceSimulation(nx, nx, Lx, Ly, dt=dt_fd, nu=cfg["fd_nu"])
    fd.set_initial_condition(lambda x, y: q0)

    settle_time = cfg["lagrangian_settle_eddy"] * tau
    _run_until_time(vpfm, settle_time, vpfm.advance)
    _run_until_time(fd, settle_time, lambda: fd.advance(scheme="arakawa"))

    n_tracers = min(cfg["lagrangian_tracers"], nx * nx)
    rng = np.random.default_rng(cfg["seed"])

    vpfm_indices = rng.choice(vpfm.particles.n_particles, size=n_tracers, replace=False)
    x_centers = (np.arange(nx) + 0.5) * (Lx / nx)
    y_centers = (np.arange(nx) + 0.5) * (Ly / nx)
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    coords = np.column_stack([Xc.ravel(), Yc.ravel()])
    tracer_indices = rng.choice(coords.shape[0], size=n_tracers, replace=False)
    fd_x = coords[tracer_indices, 0].copy()
    fd_y = coords[tracer_indices, 1].copy()

    samples_per_eddy = cfg["lagrangian_samples_per_eddy"]
    sample_dt = tau / max(samples_per_eddy, 1)
    total_samples = int(max(cfg["lagrangian_sample_eddy"] * samples_per_eddy, 1))
    label_hist_v = np.zeros((n_tracers, total_samples), dtype=int)
    label_hist_fd = np.zeros((n_tracers, total_samples), dtype=int)

    for s in range(total_samples):
        steps_v = int(np.ceil(sample_dt / vpfm.dt))
        for _ in range(steps_v):
            vpfm.advance()
        vpfm._refresh_grid_fields_hm()
        Q_v, _ = compute_weiss_field(vpfm.grid.vx, vpfm.grid.vy, vpfm.grid.dx, vpfm.grid.dy)
        Q0_v = max(np.std(Q_v), 1e-12)
        q_samples_v = _bilinear_sample(Q_v, vpfm.particles.x[vpfm_indices],
                                       vpfm.particles.y[vpfm_indices], Lx, Ly)
        labels_v = np.zeros_like(q_samples_v, dtype=int)
        labels_v[q_samples_v <= -Q0_v] = -1
        labels_v[q_samples_v >= Q0_v] = 1
        label_hist_v[:, s] = labels_v

        steps_fd = int(np.ceil(sample_dt / fd.dt))
        for _ in range(steps_fd):
            fd.advance(scheme="arakawa")
            fd_x, fd_y = _advect_positions_rk4(fd_x, fd_y, fd.vx, fd.vy, Lx, Ly, fd.dt)
        Q_fd, _ = compute_weiss_field(fd.vx, fd.vy, fd.dx, fd.dy)
        Q0_fd = max(np.std(Q_fd), 1e-12)
        q_samples_fd = _bilinear_sample(Q_fd, fd_x, fd_y, Lx, Ly)
        labels_fd = np.zeros_like(q_samples_fd, dtype=int)
        labels_fd[q_samples_fd <= -Q0_fd] = -1
        labels_fd[q_samples_fd >= Q0_fd] = 1
        label_hist_fd[:, s] = labels_fd

    residence_v = compute_residence_times(label_hist_v, sample_dt)
    residence_fd = compute_residence_times(label_hist_fd, sample_dt)

    fit_v = _fit_residence_models(residence_v)
    fit_fd = _fit_residence_models(residence_fd)

    score_v = float(np.mean([fit_v["elliptic"]["r2"], fit_v["intermediate"]["r2"], fit_v["hyperbolic"]["r2"]]))
    score_fd = float(np.mean([fit_fd["elliptic"]["r2"], fit_fd["intermediate"]["r2"], fit_fd["hyperbolic"]["r2"]]))
    advantage = score_v / score_fd if score_fd > 1e-12 else float("inf")

    return {
        "tau_eddy": float(tau),
        "sample_dt": float(sample_dt),
        "samples": int(total_samples),
        "vpfm": {
            "fit": fit_v,
            "score": score_v,
        },
        "arakawa": {
            "fit": fit_fd,
            "score": score_fd,
        },
        "advantage": float(advantage),
    }


def _fit_residence_models(residence):
    fits = {}
    fits["elliptic"] = _fit_power_law(residence.get(-1, []))
    fits["intermediate"] = _fit_exponential(residence.get(0, []))
    fits["hyperbolic"] = _fit_power_law(residence.get(1, []))
    return fits


def _fit_power_law(samples):
    values = np.asarray(samples, dtype=float)
    if values.size < 10:
        return {"slope": None, "intercept": None, "r2": 0.0, "bins": [], "pdf": []}
    t_min, t_max = np.min(values), np.max(values)
    bins = np.logspace(np.log10(t_min), np.log10(t_max), 15)
    pdf, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = pdf > 0
    if np.sum(mask) < 3:
        return {"slope": None, "intercept": None, "r2": 0.0, "bins": _as_list(centers), "pdf": _as_list(pdf)}
    x = np.log(centers[mask])
    y = np.log(pdf[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = _r2_score(y, y_pred)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "bins": _as_list(centers), "pdf": _as_list(pdf)}


def _fit_exponential(samples):
    values = np.asarray(samples, dtype=float)
    if values.size < 10:
        return {"slope": None, "intercept": None, "r2": 0.0, "bins": [], "pdf": []}
    t_min, t_max = np.min(values), np.max(values)
    bins = np.linspace(t_min, t_max, 15)
    pdf, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = pdf > 0
    if np.sum(mask) < 3:
        return {"slope": None, "intercept": None, "r2": 0.0, "bins": _as_list(centers), "pdf": _as_list(pdf)}
    x = centers[mask]
    y = np.log(pdf[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = _r2_score(y, y_pred)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "bins": _as_list(centers), "pdf": _as_list(pdf)}


def _r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _as_list(values):
    return [float(v) for v in np.asarray(values).ravel()]


def _fd_grid_from_sim(sim):
    class _GridProxy:
        pass
    grid = _GridProxy()
    grid.nx = sim.nx
    grid.ny = sim.ny
    grid.Lx = sim.Lx
    grid.Ly = sim.Ly
    grid.dx = sim.dx
    grid.dy = sim.dy
    grid.q = sim.q
    grid.phi = sim.phi
    grid.X = sim.X
    grid.Y = sim.Y
    return grid


def _plot_results(results, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    drift = results["long_time_drift"]
    axes[0, 0].plot(drift["vpfm"]["times"], drift["vpfm"]["drift"], label="VPFM")
    axes[0, 0].plot(drift["arakawa"]["times"], drift["arakawa"]["drift"], label="Arakawa")
    axes[0, 0].set_title("Long-Time Drift")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Mean Relative Drift")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    blob = results["sharp_blob"]
    axes[0, 1].plot(blob["sigmas"], blob["vpfm_peaks"], "o-", label="VPFM")
    axes[0, 1].plot(blob["sigmas"], blob["arakawa_peaks"], "s--", label="Arakawa")
    axes[0, 1].set_title("Peak Preservation vs Blob Width")
    axes[0, 1].set_xlabel("Sigma")
    axes[0, 1].set_ylabel("Peak / Initial Peak")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    shear = results["shear"]
    axes[1, 0].plot(shear["shear_rates"], shear["vpfm_t50"], "o-", label="VPFM")
    axes[1, 0].plot(shear["shear_rates"], shear["arakawa_t50"], "s--", label="Arakawa")
    axes[1, 0].set_title("Shear: Time to 50% Peak")
    axes[1, 0].set_xlabel("Shear Rate")
    axes[1, 0].set_ylabel("t_50")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    lag = results["lagrangian"]
    regions = ["elliptic", "intermediate", "hyperbolic"]
    vpfm_scores = [lag["vpfm"]["fit"][r]["r2"] for r in regions]
    fd_scores = [lag["arakawa"]["fit"][r]["r2"] for r in regions]
    x = np.arange(len(regions))
    width = 0.35
    axes[1, 1].bar(x - width / 2, vpfm_scores, width, label="VPFM")
    axes[1, 1].bar(x + width / 2, fd_scores, width, label="Arakawa")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["Elliptic", "Intermediate", "Hyperbolic"])
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Residence Time Fit Scores (R2)")
    axes[1, 1].set_ylabel("R2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def _apply_speed_profile(cfg, fast, full):
    if fast:
        cfg.update({
            "nx": 64,
            "dt": 0.02,
            "long_time_eddy": 50,
            "long_time_samples": 50,
            "blob_eddy": 10,
            "shear_eddy": 10,
            "shear_samples": 50,
            "lagrangian_settle_eddy": 5,
            "lagrangian_sample_eddy": 5,
            "lagrangian_samples_per_eddy": 3,
            "lagrangian_tracers": 1024,
        })
    elif full:
        cfg.update({
            "nx": 256,
            "dt": 0.005,
            "long_time_eddy": 500,
            "long_time_samples": 250,
            "blob_eddy": 50,
            "shear_eddy": 50,
            "shear_samples": 200,
            "lagrangian_settle_eddy": 30,
            "lagrangian_sample_eddy": 30,
            "lagrangian_samples_per_eddy": 6,
            "lagrangian_tracers": 16384,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Use a lightweight config")
    parser.add_argument("--full", action="store_true", help="Use a heavier config")
    parser.add_argument("--backend", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.fast and args.full:
        raise SystemExit("Choose only one of --fast or --full.")

    cfg = {
        "nx": 128,
        "Lx": 2 * np.pi,
        "dt": 0.01,
        "fd_dt_scale": 0.25,
        "particles_per_cell": 1,
        "reinit_threshold": 2.0,
        "max_reinit_steps": 200,
        "fd_nu": 0.0,
        "k_peak": 5.0,
        "amplitude": 0.1,
        "seed": args.seed,
        "backend": args.backend,
        "long_time_eddy": 500,
        "long_time_samples": 200,
        "blob_eddy": 50,
        "blob_sigmas": [0.5, 1.0, 2.0, 4.0],
        "blob_amplitude": 1.0,
        "shear_eddy": 50,
        "shear_rates": [0.1, 0.3, 0.5, 0.7, 1.0],
        "shear_samples": 200,
        "shear_threshold": 0.5,
        "shear_blob_sigma": 1.0,
        "lagrangian_settle_eddy": 20,
        "lagrangian_sample_eddy": 20,
        "lagrangian_samples_per_eddy": 5,
        "lagrangian_tracers": 8192,
    }
    _apply_speed_profile(cfg, args.fast, args.full)

    results = {
        "config": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in cfg.items()},
        "long_time_drift": long_time_drift(cfg),
        "sharp_blob": sharp_blob_preservation(cfg),
        "shear": blob_in_shear(cfg),
        "lagrangian": lagrangian_statistics(cfg),
    }

    for key in ["long_time_drift", "sharp_blob", "shear", "lagrangian"]:
        advantage = results[key]["advantage"]
        win = advantage > 1.2
        results[key]["win"] = bool(win)
        print(f"{key}: advantage {advantage:.2f}x -> {'WIN' if win else 'NO WIN'}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "regime_comparison_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {json_path}")

    plot_path = output_dir / "regime_comparison_plots.png"
    _plot_results(results, plot_path)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
