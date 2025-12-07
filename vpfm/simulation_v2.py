"""Improved VPFM simulation with higher-order methods.

Uses:
- Quadratic/cubic B-spline interpolation
- RK4 Jacobian evolution
- Hessian tracking
- Adaptive reinitialization

This should achieve the target 10x improvement over finite differences.
"""

import numpy as np
from typing import Callable, Optional
from tqdm import tqdm

from .grid import Grid
from .particles import ParticleSystem
from .kernels import InterpolationKernel, P2G_bspline, G2P_bspline
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .flow_map import FlowMapIntegrator, FlowMapState
from .diagnostics import compute_diagnostics


class SimulationV2:
    """Improved VPFM simulation with higher-order accuracy.

    Key improvements over v1:
    1. B-spline interpolation (quadratic by default)
    2. RK4 Jacobian evolution
    3. Hessian tracking for gradient accuracy
    4. Adaptive reinitialization

    Attributes:
        grid: Eulerian grid
        particles: Lagrangian particle system
        flow_map: Flow map state (J, H)
        integrator: Flow map integrator
    """

    def __init__(self,
                 nx: int = 128,
                 ny: int = 128,
                 Lx: float = 2 * np.pi,
                 Ly: float = 2 * np.pi,
                 dt: float = 0.01,
                 particles_per_cell: int = 1,
                 kernel_order: str = 'quadratic',
                 track_hessian: bool = True,
                 reinit_threshold: float = 2.0,
                 max_reinit_steps: int = 200):
        """Initialize improved simulation.

        Args:
            nx, ny: Grid resolution
            Lx, Ly: Domain size
            dt: Time step
            particles_per_cell: Particle density
            kernel_order: 'linear', 'quadratic', or 'cubic'
            track_hessian: Whether to track Hessian
            reinit_threshold: Error threshold for reinitialization
            max_reinit_steps: Maximum steps between reinits
        """
        self.grid = Grid(nx, ny, Lx, Ly)
        self.particles = ParticleSystem.from_grid(self.grid, particles_per_cell)

        self.kernel = InterpolationKernel(kernel_order)
        self.integrator = FlowMapIntegrator(self.grid, kernel_order, track_hessian)
        self.flow_map = self.integrator.initialize_flow_map(self.particles.n_particles)

        self.time = 0.0
        self.step = 0
        self.dt = dt

        self.reinit_threshold = reinit_threshold
        self.max_reinit_steps = max_reinit_steps

        # Diagnostic history
        self.history = {
            'time': [],
            'energy': [],
            'enstrophy': [],
            'pot_enstrophy': [],
            'max_q': [],
            'max_jacobian_dev': [],
            'reinit_count': 0,
        }

    def set_initial_condition(self, q_func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Set initial condition from a function.

        Args:
            q_func: Function q(x, y) returning potential vorticity
        """
        # Set on particles
        self.particles.q = q_func(self.particles.x, self.particles.y)

        # Transfer to grid
        self._p2g()

        # Initial solve
        self.grid.phi = solve_poisson_hm(self.grid.q, self.grid.Lx, self.grid.Ly)
        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

    def _p2g(self):
        """Particle to grid transfer using B-spline kernel."""
        self.grid.q = P2G_bspline(
            self.particles.x, self.particles.y, self.particles.q,
            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy,
            self.kernel
        )

    def _g2p(self):
        """Grid to particle transfer for reinitialization."""
        self.particles.q = G2P_bspline(
            self.grid.q, self.particles.x, self.particles.y,
            self.grid.dx, self.grid.dy, self.kernel
        )

    def reinitialize(self):
        """Reinitialize flow map."""
        # Current grid field is up to date from last P2G
        self.particles.q = self.integrator.reinitialize(
            self.particles, self.flow_map, self.grid.q
        )
        self.history['reinit_count'] += 1

    def advance(self):
        """Advance simulation by one time step."""
        # 1. P2G transfer
        self._p2g()

        # 2. Poisson solve
        self.grid.phi = solve_poisson_hm(self.grid.q, self.grid.Lx, self.grid.Ly)

        # 3. Compute velocity and gradients
        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

        # 4. Flow map evolution (Jacobian, Hessian, positions)
        self.integrator.step(self.particles, self.flow_map, self.dt)

        # 5. Update time
        self.time += self.dt
        self.step += 1

        # 6. Check for reinitialization
        if self.integrator.should_reinitialize(
            self.flow_map, self.reinit_threshold, self.max_reinit_steps
        ):
            self.reinitialize()

    def run(self,
            n_steps: int,
            diag_interval: int = 10,
            callback: Optional[Callable] = None,
            verbose: bool = True,
            progress: bool = True):
        """Run simulation for specified number of steps.

        Args:
            n_steps: Number of time steps
            diag_interval: Steps between diagnostics
            callback: Optional callback(sim) called at diag_interval
            verbose: Print progress
            progress: Show tqdm progress bar
        """
        iterator = tqdm(range(n_steps), desc="V2 Simulation", disable=not progress)
        for i in iterator:
            self.advance()

            if (self.step % diag_interval == 0) or (i == n_steps - 1):
                diag = compute_diagnostics(self.grid)
                max_jac = self.integrator.estimate_error(self.flow_map)

                self.history['time'].append(self.time)
                self.history['energy'].append(diag['energy'])
                self.history['enstrophy'].append(diag['enstrophy'])
                self.history['pot_enstrophy'].append(diag['pot_enstrophy'])
                self.history['max_q'].append(diag['max_q'])
                self.history['max_jacobian_dev'].append(max_jac)

                if verbose and self.step % (10 * diag_interval) == 0:
                    print(f"Step {self.step:5d}, t={self.time:.2f}, "
                          f"E={diag['energy']:.6f}, Z={diag['enstrophy']:.6f}, "
                          f"|J-I|={max_jac:.3f}, reinits={self.history['reinit_count']}")

                if callback is not None:
                    callback(self)

    def get_field(self, name: str) -> np.ndarray:
        """Get a grid field by name."""
        return getattr(self.grid, name)


# Convenience function to compare v1 and v2
def run_comparison_v1_v2(q_func: Callable, nx: int = 128,
                          n_steps: int = 1000, dt: float = 0.01) -> dict:
    """Compare v1 (linear/Euler) with v2 (quadratic/RK4).

    Args:
        q_func: Initial condition function
        nx: Grid resolution
        n_steps: Number of steps
        dt: Time step

    Returns:
        Dictionary with comparison metrics
    """
    from .simulation import Simulation  # v1

    Lx = Ly = 2 * np.pi

    # V1: Original implementation
    sim_v1 = Simulation(nx, nx, Lx, Ly, dt=dt)
    sim_v1.set_initial_condition(q_func)
    # Store initial peak from particles (what VPFM preserves)
    q_init_particles = sim_v1.particles.q.copy()
    peak_init = np.abs(q_init_particles).max()
    E_init_v1 = None

    sim_v1.run(n_steps, diag_interval=n_steps, verbose=False, progress=True)

    E_v1 = np.array(sim_v1.history['energy'])
    # Compare particle peak (both methods store q on particles)
    peak_v1 = np.abs(sim_v1.particles.q).max()

    # V2: Improved implementation
    sim_v2 = SimulationV2(nx, nx, Lx, Ly, dt=dt,
                          kernel_order='quadratic', track_hessian=True)
    sim_v2.set_initial_condition(q_func)

    sim_v2.run(n_steps, diag_interval=n_steps, verbose=False, progress=True)

    E_v2 = np.array(sim_v2.history['energy'])
    # Compare particle peak (VPFM preserves particle values perfectly between reinits)
    peak_v2 = np.abs(sim_v2.particles.q).max()

    # Compute improvement factor
    # V1 peak loss
    v1_loss = 1 - peak_v1 / peak_init
    # V2 peak loss
    v2_loss = 1 - peak_v2 / peak_init

    # Improvement: how much better is V2 at preserving peak?
    # If V2 has less loss, factor > 1
    improvement = v1_loss / max(v2_loss, 1e-10) if v2_loss > 0 else float('inf')

    return {
        'v1_peak_ratio': peak_v1 / peak_init,
        'v2_peak_ratio': peak_v2 / peak_init,
        'v1_energy_error': abs(E_v1[-1] - E_v1[0]) / E_v1[0] if len(E_v1) > 0 else 0,
        'v2_energy_error': abs(E_v2[-1] - E_v2[0]) / E_v2[0] if len(E_v2) > 0 else 0,
        'v2_reinit_count': sim_v2.history['reinit_count'],
        'improvement_factor': improvement,
    }
