"""Main simulation class for VPFM-Plasma.

Orchestrates the VPFM algorithm for Hasegawa-Mima equation.
"""

import numpy as np
from typing import Callable, Optional
from .grid import Grid
from .particles import ParticleSystem
from .transfers import P2G_vectorized, G2P
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .integrator import RK4Integrator, compute_cfl_dt
from .diagnostics import compute_diagnostics


class Simulation:
    """VPFM simulation for Hasegawa-Mima plasma turbulence.

    Implements the main time-stepping loop with:
    - P2G transfer of vorticity
    - Poisson solve for potential
    - Velocity computation
    - Particle advection (RK4)
    - Jacobian evolution
    - Flow map reinitialization

    Attributes:
        grid: Eulerian grid
        particles: Lagrangian particle system
        time: Current simulation time
        step: Current step number
        dt: Time step
    """

    def __init__(self,
                 nx: int = 128,
                 ny: int = 128,
                 Lx: float = 2 * np.pi,
                 Ly: float = 2 * np.pi,
                 dt: float = 0.01,
                 particles_per_cell: int = 1):
        """Initialize simulation.

        Args:
            nx, ny: Grid resolution
            Lx, Ly: Domain size
            dt: Time step
            particles_per_cell: Number of particles per grid cell
        """
        self.grid = Grid(nx, ny, Lx, Ly)
        self.particles = ParticleSystem.from_grid(self.grid, particles_per_cell)
        self.integrator = RK4Integrator(self.grid)

        self.time = 0.0
        self.step = 0
        self.dt = dt

        # Flow map parameters
        self.reinit_interval = 20  # Steps between reinitializations
        self.reinit_threshold = 0.5  # Maximum ||J - I|| before reinit

        # Diagnostic history
        self.history = {
            'time': [],
            'energy': [],
            'enstrophy': [],
            'pot_enstrophy': [],
            'max_q': [],
            'max_jacobian_dev': [],
        }

    def set_initial_condition(self, q_func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Set initial condition from a function.

        Args:
            q_func: Function q(x, y) returning potential vorticity field
        """
        # Set on particles
        self.particles.set_initial_condition(q_func, self.grid)

        # Transfer to grid for initial Poisson solve
        self.grid.q = P2G_vectorized(self.particles, self.grid)

        # Initial Poisson solve and velocity
        self.grid.phi = solve_poisson_hm(self.grid.q, self.grid.Lx, self.grid.Ly)
        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

    def reinitialize_flow_map(self):
        """Reinitialize the flow map.

        Transfers q from particles to grid and back, resets Jacobians.
        """
        # P2G transfer
        self.grid.q = P2G_vectorized(self.particles, self.grid)

        # G2P transfer - update particle vorticity from grid
        self.particles.q = G2P(self.grid, self.particles, self.grid.q)

        # Reset Jacobians to identity
        self.particles.reset_jacobian()

    def advance(self):
        """Advance simulation by one time step."""
        # 1. P2G: Transfer vorticity from particles to grid
        self.grid.q = P2G_vectorized(self.particles, self.grid)

        # 2. Poisson solve: (nabla^2 - 1) phi = -q
        self.grid.phi = solve_poisson_hm(self.grid.q, self.grid.Lx, self.grid.Ly)

        # 3. Compute velocity: v = z_hat × nabla(phi)
        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

        # 4. Advect particles and evolve Jacobians
        self.integrator.step(self.particles, self.dt)

        # 5. Update time
        self.time += self.dt
        self.step += 1

        # 6. Check for flow map reinitialization
        if self.step % self.reinit_interval == 0:
            max_dev = self.particles.max_jacobian_deviation()
            if max_dev > self.reinit_threshold:
                self.reinitialize_flow_map()

    def run(self,
            n_steps: int,
            diag_interval: int = 10,
            callback: Optional[Callable] = None,
            verbose: bool = True):
        """Run simulation for specified number of steps.

        Args:
            n_steps: Number of time steps
            diag_interval: Steps between diagnostic computation
            callback: Optional callback function(sim) called each diag_interval
            verbose: Print progress
        """
        for i in range(n_steps):
            self.advance()

            if (self.step % diag_interval == 0) or (i == n_steps - 1):
                diag = compute_diagnostics(self.grid)
                max_jac = self.particles.max_jacobian_deviation()

                self.history['time'].append(self.time)
                self.history['energy'].append(diag['energy'])
                self.history['enstrophy'].append(diag['enstrophy'])
                self.history['pot_enstrophy'].append(diag['pot_enstrophy'])
                self.history['max_q'].append(diag['max_q'])
                self.history['max_jacobian_dev'].append(max_jac)

                if verbose and self.step % (10 * diag_interval) == 0:
                    print(f"Step {self.step:5d}, t={self.time:.2f}, "
                          f"E={diag['energy']:.6f}, Z={diag['enstrophy']:.6f}, "
                          f"max|J-I|={max_jac:.3f}")

                if callback is not None:
                    callback(self)

    def get_field(self, name: str) -> np.ndarray:
        """Get a grid field by name.

        Args:
            name: Field name ('q', 'phi', 'vx', 'vy')

        Returns:
            Field array
        """
        return getattr(self.grid, name)


def lamb_oseen(x: np.ndarray, y: np.ndarray,
               x0: float, y0: float,
               Gamma: float = 2 * np.pi,
               r0: float = 1.0) -> np.ndarray:
    """Lamb-Oseen (Gaussian) vortex initial condition.

    Args:
        x, y: Coordinate arrays
        x0, y0: Vortex center
        Gamma: Circulation
        r0: Core radius

    Returns:
        Vorticity field zeta = nabla^2(phi)
    """
    r2 = (x - x0)**2 + (y - y0)**2
    zeta = (Gamma / (np.pi * r0**2)) * np.exp(-r2 / r0**2)
    return zeta


def vortex_pair(x: np.ndarray, y: np.ndarray,
                Lx: float, Ly: float,
                separation: float = 3.0,
                Gamma: float = 2 * np.pi,
                r0: float = 1.0) -> np.ndarray:
    """Two co-rotating vortices initial condition.

    Args:
        x, y: Coordinate arrays
        Lx, Ly: Domain size
        separation: Distance between vortex centers
        Gamma: Circulation (same sign for co-rotating)
        r0: Core radius

    Returns:
        Combined vorticity field
    """
    x_center = Lx / 2
    y_center = Ly / 2

    x1 = x_center - separation / 2
    x2 = x_center + separation / 2

    q1 = lamb_oseen(x, y, x1, y_center, Gamma, r0)
    q2 = lamb_oseen(x, y, x2, y_center, Gamma, r0)

    return q1 + q2


def random_turbulence(nx: int, ny: int,
                      Lx: float, Ly: float,
                      k_peak: float = 5.0,
                      amplitude: float = 0.1,
                      seed: Optional[int] = None) -> np.ndarray:
    """Random turbulent initial condition with specified spectrum.

    Args:
        nx, ny: Grid size
        Lx, Ly: Domain size
        k_peak: Peak wavenumber of energy spectrum
        amplitude: Overall amplitude scaling
        seed: Random seed for reproducibility

    Returns:
        Random vorticity field
    """
    if seed is not None:
        np.random.seed(seed)

    from numpy.fft import fft2, ifft2, fftfreq

    dx = Lx / nx
    dy = Ly / ny

    # Wave numbers
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # Energy spectrum peaked at k_peak
    E_k = K**4 * np.exp(-(K / k_peak)**2)

    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, (nx, ny))

    # Construct in Fourier space
    zeta_hat = np.sqrt(E_k) * np.exp(1j * phases)
    zeta_hat[0, 0] = 0  # Zero mean

    zeta = amplitude * np.real(ifft2(zeta_hat))

    return zeta
