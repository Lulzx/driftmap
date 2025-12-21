"""Linear dispersion validation for HW model."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from vpfm import Simulation


def _hw_dispersion_lambda(kx, ky, alpha, kappa, nu, mu, D, nu_sheath):
    """Return the two HW eigenvalues for the linearized system."""
    k2 = kx**2 + ky**2
    d_zeta = mu * k2**2 + nu * k2 + nu_sheath
    d_n = D * k2

    a = k2
    b = alpha * (1.0 + k2) + k2 * (d_n + d_zeta)
    c = alpha * d_n + alpha * k2 * d_zeta + k2 * d_n * d_zeta + 1j * alpha * kappa * ky

    disc = b**2 - 4.0 * a * c
    sqrt_disc = np.sqrt(disc)
    lam1 = (-b + sqrt_disc) / (2.0 * a)
    lam2 = (-b - sqrt_disc) / (2.0 * a)
    return lam1, lam2


def _measure_growth(sim, kx_mode, ky_mode, n_steps, sample_skip=0.2):
    """Measure growth rate and frequency from a single Fourier mode."""
    nx, ny = sim.grid.nx, sim.grid.ny
    idx_x = kx_mode % nx
    idx_y = ky_mode % ny

    times = []
    amps = []
    phases = []

    for _ in range(n_steps):
        sim.step_hw()
        sim._refresh_grid_fields_hw()

        coeff = np.fft.fft2(sim.grid.phi)[idx_x, idx_y] / (nx * ny)
        times.append(sim.time)
        amps.append(np.abs(coeff))
        phases.append(np.angle(coeff))

    times = np.asarray(times)
    amps = np.asarray(amps)
    phases = np.unwrap(np.asarray(phases))

    start = int(len(times) * sample_skip)
    end = max(start + 5, int(len(times) * (1.0 - sample_skip / 2.0)))
    t_fit = times[start:end]

    gamma = np.polyfit(t_fit, np.log(amps[start:end]), 1)[0]
    omega = np.polyfit(t_fit, phases[start:end], 1)[0]
    return gamma, omega


def _build_initial_conditions(Lx, Ly, kx, ky, phi_amp, ratio):
    """Return zeta and density initial conditions for an eigenmode."""
    k2 = kx**2 + ky**2

    def zeta_ic(px, py):
        phase = kx * px + ky * py
        phi = phi_amp * np.cos(phase)
        return -k2 * phi

    def n_ic(px, py):
        phase = kx * px + ky * py
        return phi_amp * np.real(ratio * np.exp(1j * phase))

    return zeta_ic, n_ic


def _run_mode(nx, ny, Lx, Ly, mode, alpha, kappa, dt, n_steps):
    kx_mode, ky_mode = mode
    kx = 2.0 * np.pi * kx_mode / Lx
    ky = 2.0 * np.pi * ky_mode / Ly

    lam1, lam2 = _hw_dispersion_lambda(kx, ky, alpha, kappa, 0.0, 0.0, 0.0, 0.0)
    lam = lam1 if lam1.real >= lam2.real else lam2
    assert lam.real > 0.05

    ratio = 1.0 + (lam * (kx**2 + ky**2)) / alpha
    zeta_ic, n_ic = _build_initial_conditions(Lx, Ly, kx, ky, 1e-3, ratio)

    sim = Simulation(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        alpha=alpha,
        kappa=kappa,
        nu=0.0,
        mu=0.0,
        D=0.0,
        nu_sheath=0.0,
        kernel_order="quadratic",
        use_gradient_p2g=False,
        particles_per_cell=1,
        reinit_threshold=1e9,
        max_reinit_steps=10**9,
        linear_hw=True,
    )

    sim.set_initial_condition_hw(zeta_ic, n_ic)
    gamma, omega = _measure_growth(sim, kx_mode, ky_mode, n_steps)
    return gamma, omega, lam


def test_hw_linear_dispersion():
    nx = ny = 64
    Lx = Ly = 2.0 * np.pi
    alpha = 1.0
    kappa = 5.0
    dt = 0.01
    n_steps = 200

    modes = [(0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]

    for mode in modes:
        gamma, omega, lam = _run_mode(nx, ny, Lx, Ly, mode, alpha, kappa, dt, n_steps)

        gamma_ref = lam.real
        omega_ref = lam.imag

        gamma_err = abs(gamma - gamma_ref) / abs(gamma_ref)
        omega_err = abs(omega - omega_ref) / max(1e-8, abs(omega_ref))

        assert gamma_err < 0.05
        assert omega_err < 0.05
