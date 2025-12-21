"""Tests for spectral and topology diagnostics utilities."""

import numpy as np

from vpfm.diagnostics.spectral import energy_spectrum, density_flux_spectrum, fit_power_law
from vpfm.diagnostics.flow_topology import compute_weiss_field, partition_flow, compute_residence_times


def test_energy_spectrum_shapes():
    nx = ny = 32
    Lx = Ly = 2 * np.pi
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    vx = np.sin(X) * 0.1
    vy = np.cos(Y) * 0.1
    k, E_k = energy_spectrum(vx, vy, Lx, Ly)

    assert k.shape == E_k.shape
    assert np.all(np.isfinite(E_k))


def test_density_flux_spectrum_shapes():
    nx = ny = 32
    Lx = Ly = 2 * np.pi
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    vx = np.sin(X)
    n = np.cos(Y)
    k, F_k = density_flux_spectrum(vx, n, Lx, Ly)

    assert k.shape == F_k.shape
    assert np.all(np.isfinite(F_k))


def test_fit_power_law():
    k = np.linspace(1.0, 10.0, 50)
    spectrum = k**-4.0
    slope, _ = fit_power_law(k, spectrum, 2.0, 8.0)
    assert np.isfinite(slope)
    assert -4.5 < slope < -3.5


def test_flow_topology_partition():
    nx = ny = 16
    dx = dy = 1.0
    vx = np.random.randn(nx, ny)
    vy = np.random.randn(nx, ny)

    Q, omega = compute_weiss_field(vx, vy, dx, dy)
    labels, Q0 = partition_flow(Q)

    assert Q.shape == omega.shape == labels.shape
    assert Q0 >= 0.0
    assert set(np.unique(labels)).issubset({-1, 0, 1})


def test_residence_times_counts():
    labels = np.array([
        [-1, -1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1],
    ])
    residence = compute_residence_times(labels, dt=0.5)

    assert set(residence.keys()) == {-1, 0, 1}
    assert sum(len(v) for v in residence.values()) == 5
