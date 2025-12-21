"""Smoke tests for MLX backend integration."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from vpfm import (
    Simulation,
    check_mlx_available,
    lamb_oseen,
    InterpolationKernel,
    P2G_bspline,
    P2G_bspline_gradient_enhanced,
    G2P_bspline,
    G2P_bspline_with_gradient,
    P2G_mlx,
    P2G_mlx_gradient_enhanced,
    G2P_mlx,
    G2P_mlx_with_gradient,
)


def test_mlx_backend_hm_step():
    """Ensure MLX backend runs a basic HM step when available."""
    if not check_mlx_available():
        pytest.skip("MLX not available")

    nx = ny = 16
    Lx = Ly = 2 * np.pi
    sim = Simulation(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=0.01,
        backend="mlx",
        use_gradient_p2g=False,
    )

    def ic(x, y):
        return lamb_oseen(x, y, Lx / 2, Ly / 2, Gamma=2 * np.pi, r0=0.5)

    sim.set_initial_condition(ic)
    sim.advance()
    sim._refresh_grid_fields_hm()

    assert np.isfinite(sim.grid.phi).all()
    assert np.isfinite(sim.grid.vx).all()
    assert np.isfinite(sim.grid.vy).all()


def test_mlx_transfers_match_cpu_quadratic():
    """Check MLX P2G/G2P against CPU quadratic B-spline transfers."""
    if not check_mlx_available():
        pytest.skip("MLX not available")

    import mlx.core as mx

    np.random.seed(0)
    nx = ny = 16
    Lx = Ly = 2 * np.pi
    dx = Lx / nx
    dy = Ly / ny
    n_particles = 64

    px = np.random.uniform(0, Lx, n_particles)
    py = np.random.uniform(0, Ly, n_particles)
    pq = np.random.randn(n_particles)
    gx = np.random.randn(n_particles) * 0.1
    gy = np.random.randn(n_particles) * 0.1

    kernel = InterpolationKernel("quadratic")

    q_grid_cpu = P2G_bspline(px, py, pq, nx, ny, dx, dy, kernel)
    q_grid_mx = P2G_mlx(
        mx.array(px.astype(np.float32)),
        mx.array(py.astype(np.float32)),
        mx.array(pq.astype(np.float32)),
        nx, ny, dx, dy,
    )
    q_grid_mx_np = np.array(q_grid_mx)
    assert np.allclose(q_grid_cpu, q_grid_mx_np, rtol=1e-4, atol=1e-5)

    q_grid_cpu_grad = P2G_bspline_gradient_enhanced(
        px, py, pq, gx, gy, nx, ny, dx, dy, kernel
    )
    q_grid_mx_grad = P2G_mlx_gradient_enhanced(
        mx.array(px.astype(np.float32)),
        mx.array(py.astype(np.float32)),
        mx.array(pq.astype(np.float32)),
        mx.array(gx.astype(np.float32)),
        mx.array(gy.astype(np.float32)),
        nx, ny, dx, dy,
    )
    q_grid_mx_grad_np = np.array(q_grid_mx_grad)
    assert np.allclose(q_grid_cpu_grad, q_grid_mx_grad_np, rtol=1e-4, atol=1e-5)

    grid_field = np.random.randn(nx, ny)
    g2p_cpu = G2P_bspline(grid_field, px, py, dx, dy, kernel)
    g2p_mx = G2P_mlx(
        mx.array(grid_field.astype(np.float32)),
        mx.array(px.astype(np.float32)),
        mx.array(py.astype(np.float32)),
        dx, dy,
    )
    g2p_mx_np = np.array(g2p_mx)
    assert np.allclose(g2p_cpu, g2p_mx_np, rtol=1e-4, atol=1e-5)

    val_cpu, gx_cpu, gy_cpu = G2P_bspline_with_gradient(
        grid_field, px, py, dx, dy, kernel
    )
    val_mx, gx_mx, gy_mx = G2P_mlx_with_gradient(
        mx.array(grid_field.astype(np.float32)),
        mx.array(px.astype(np.float32)),
        mx.array(py.astype(np.float32)),
        dx, dy,
    )
    assert np.allclose(val_cpu, np.array(val_mx), rtol=1e-4, atol=1e-5)
    assert np.allclose(gx_cpu, np.array(gx_mx), rtol=1e-4, atol=1e-5)
    assert np.allclose(gy_cpu, np.array(gy_mx), rtol=1e-4, atol=1e-5)
