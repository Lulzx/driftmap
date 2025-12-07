"""VPFM simulation for plasma turbulence.

Implements the Vortex Particle Flow Map method with:
- Quadratic/cubic B-spline interpolation
- RK4 Jacobian evolution
- Hessian tracking for gradient accuracy
- Adaptive reinitialization

Supports both Hasegawa-Mima (HM) and Hasegawa-Wakatani (HW) physics:
- HM: Pure drift-wave advection with (∇² - 1)φ = -q
- HW: Resistive coupling α(φ - n) drives drift-wave instability
     Curvature drive κ·∂φ/∂y (interchange instability)
     Hyperviscosity μ∇⁴ζ for small-scale dissipation
     Density diffusion D∇²n
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from typing import Callable, Optional
from tqdm import tqdm

from .grid import Grid
from .particles import ParticleSystem
from .kernels import InterpolationKernel, P2G_bspline, G2P_bspline
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .flow_map import FlowMapIntegrator, FlowMapState
from .diagnostics import compute_diagnostics


class Simulation:
    """VPFM simulation for plasma turbulence.

    Features:
    - B-spline interpolation (quadratic by default)
    - RK4 Jacobian evolution
    - Hessian tracking for gradient accuracy
    - Adaptive reinitialization
    - Hasegawa-Mima (advance) and Hasegawa-Wakatani (step_hw) physics

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
                 max_reinit_steps: int = 200,
                 # Hasegawa-Wakatani physics parameters
                 alpha: float = 0.0,      # Adiabaticity (0 = pure HM, >0 = HW)
                 kappa: float = 0.0,      # Curvature drive
                 mu: float = 0.0,         # Hyperviscosity
                 D: float = 0.0,          # Density diffusion
                 nu_sheath: float = 0.0): # Sheath damping
        """Initialize simulation.

        Args:
            nx, ny: Grid resolution
            Lx, Ly: Domain size
            dt: Time step
            particles_per_cell: Particle density
            kernel_order: 'linear', 'quadratic', or 'cubic'
            track_hessian: Whether to track Hessian
            reinit_threshold: Error threshold for reinitialization
            max_reinit_steps: Maximum steps between reinits
            alpha: Adiabaticity parameter (α → ∞ is HM limit, α ~ 1 is HW)
            kappa: Background density gradient / curvature drive
            mu: Hyperviscosity coefficient for ∇⁴ζ
            D: Density diffusivity for ∇²n
            nu_sheath: Sheath damping rate
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

        # Hasegawa-Wakatani physics parameters
        self.alpha = alpha
        self.kappa = kappa
        self.mu = mu
        self.D = D
        self.nu_sheath = nu_sheath

        # Density field on particles (for HW coupling)
        self.n_particles = np.zeros(self.particles.n_particles)
        self.n_grid = np.zeros((nx, ny))

        # Precompute wave numbers for spectral operators
        kx = fftfreq(nx, Lx / nx) * 2 * np.pi
        ky = fftfreq(ny, Ly / ny) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.K4 = self.K2**2

        # Diagnostic history
        self.history = {
            'time': [],
            'energy': [],
            'enstrophy': [],
            'pot_enstrophy': [],
            'max_q': [],
            'max_jacobian_dev': [],
            'reinit_count': 0,
            # HW-specific diagnostics
            'density_variance': [],
            'particle_flux': [],
            'zonal_energy': [],
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

    # ========================================================================
    # Hasegawa-Mima (pure advection) methods
    # ========================================================================

    def advance(self):
        """Advance simulation by one time step (Hasegawa-Mima physics)."""
        # 1. P2G transfer
        self._p2g()

        # 2. Poisson solve: (∇² - 1)φ = -q
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
        """Run Hasegawa-Mima simulation.

        Args:
            n_steps: Number of time steps
            diag_interval: Steps between diagnostics
            callback: Optional callback(sim) called at diag_interval
            verbose: Print progress
            progress: Show tqdm progress bar
        """
        iterator = tqdm(range(n_steps), desc="Simulation", disable=not progress)
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

    # ========================================================================
    # Hasegawa-Wakatani (turbulence-driving) methods
    # ========================================================================

    def set_initial_condition_hw(self,
                                  zeta_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                  n_func: Optional[Callable] = None):
        """Set initial conditions for HW simulation.

        Args:
            zeta_func: Function ζ(x, y) for initial vorticity
            n_func: Function n(x, y) for initial density (defaults to zeta_func)
        """
        # Set vorticity on particles
        self.particles.q = zeta_func(self.particles.x, self.particles.y)

        # Set density on particles (default to vorticity if not specified)
        if n_func is None:
            n_func = zeta_func
        self.n_particles = n_func(self.particles.x, self.particles.y)

        # Transfer to grid
        self._p2g()
        self._p2g_density()

        # Initial solve (standard Poisson for HW: ∇²φ = ζ)
        self._solve_poisson_hw()
        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

    def _p2g_density(self):
        """Transfer density from particles to grid."""
        self.n_grid = P2G_bspline(
            self.particles.x, self.particles.y, self.n_particles,
            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy,
            self.kernel
        )

    def _solve_poisson_hw(self):
        """Solve Poisson equation for HW: ∇²φ = ζ.

        Unlike HM which uses (∇² - 1)φ = -ζ, HW uses the standard
        Poisson equation since ζ = ∇²φ directly.
        """
        zeta_hat = fft2(self.grid.q)

        # Avoid division by zero at k=0
        K2_safe = self.K2.copy()
        K2_safe[0, 0] = 1.0

        phi_hat = -zeta_hat / K2_safe
        phi_hat[0, 0] = 0  # Zero mean potential

        self.grid.phi = np.real(ifft2(phi_hat))

    def _compute_hw_sources(self) -> tuple:
        """Compute Hasegawa-Wakatani source terms.

        The HW equations are:
            ∂ζ/∂t + {φ, ζ} = α(φ - n) - μ∇⁴ζ - ν_sheath·ζ
            ∂n/∂t + {φ, n} = α(φ - n) - κ·∂φ/∂y - D∇²n

        The advection {φ, ζ} and {φ, n} are handled by particle motion.
        This method computes the source terms on the RHS.

        Returns:
            (S_zeta, S_n): Source terms for vorticity and density on grid
        """
        # Resistive coupling: α(φ - n)
        # This is the key term that drives the drift-wave instability
        coupling = self.alpha * (self.grid.phi - self.n_grid)

        S_zeta = coupling.copy()
        S_n = coupling.copy()

        # Hyperviscosity: -μ∇⁴ζ (dissipates small scales)
        if self.mu > 1e-10:
            zeta_hat = fft2(self.grid.q)
            hyperviscosity = np.real(ifft2(-self.mu * self.K4 * zeta_hat))
            S_zeta += hyperviscosity

        # Sheath damping: -ν_sheath·ζ
        if self.nu_sheath > 1e-10:
            S_zeta -= self.nu_sheath * self.grid.q

        # Curvature drive: -κ·∂φ/∂y (interchange instability)
        if abs(self.kappa) > 1e-10:
            phi_hat = fft2(self.grid.phi)
            dphi_dy = np.real(ifft2(1j * self.KY * phi_hat))
            S_n -= self.kappa * dphi_dy

        # Density diffusion: D∇²n (not -D∇²n since we move D to RHS)
        if self.D > 1e-10:
            n_hat = fft2(self.n_grid)
            diffusion = np.real(ifft2(-self.D * self.K2 * n_hat))
            S_n += diffusion

        return S_zeta, S_n

    def _interpolate_to_particles(self, grid_field: np.ndarray) -> np.ndarray:
        """Interpolate a grid field to particle positions."""
        return G2P_bspline(
            grid_field, self.particles.x, self.particles.y,
            self.grid.dx, self.grid.dy, self.kernel
        )

    def step_hw(self):
        """Advance one timestep with Hasegawa-Wakatani physics.

        This is the key method that tests whether perfectly preserved
        vortex blobs can drive turbulence through the HW instability.

        The algorithm:
        1. P2G transfer of vorticity and density
        2. Solve Poisson for potential
        3. Compute HW source terms: α(φ - n), curvature, dissipation
        4. Update particle strengths from sources (dq/dt, dn/dt)
        5. Advect particles with ExB velocity (preserves values exactly)
        6. Evolve flow map Jacobian
        7. Check for reinitialization

        Physics note:
        - In pure advection (step 5), particle values are EXACTLY preserved
        - The HW source terms (step 4) are what DRIVE turbulence
        - α(φ - n) ≠ 0 when density-potential coupling breaks down
        - This generates vorticity even as advection conserves it
        """
        # 1. P2G transfer
        self._p2g()
        self._p2g_density()

        # 2. Poisson solve for potential
        self._solve_poisson_hw()

        # 3. Compute velocity and gradients
        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

        # 4. Compute HW source terms on grid
        S_zeta, S_n = self._compute_hw_sources()

        # 5. Update particle strengths from grid sources
        # Interpolate source terms to particle positions
        dq_dt = self._interpolate_to_particles(S_zeta)
        dn_dt = self._interpolate_to_particles(S_n)

        self.particles.q += self.dt * dq_dt
        self.n_particles += self.dt * dn_dt

        # 6. Flow map evolution (Jacobian, Hessian, positions)
        # This advects particles - vorticity transport is EXACT here
        self.integrator.step(self.particles, self.flow_map, self.dt)

        # 7. Update time
        self.time += self.dt
        self.step += 1

        # 8. Check for reinitialization
        if self.integrator.should_reinitialize(
            self.flow_map, self.reinit_threshold, self.max_reinit_steps
        ):
            self._reinitialize_hw()

    def _reinitialize_hw(self):
        """Reinitialize flow map for HW simulation.

        Must handle both vorticity and density fields.
        """
        # Update grid fields
        self._p2g()
        self._p2g_density()

        # Reinitialize vorticity
        self.particles.q = self.integrator.reinitialize(
            self.particles, self.flow_map, self.grid.q
        )

        # Reinitialize density
        self.n_particles = self._interpolate_to_particles(self.n_grid)

        self.history['reinit_count'] += 1

    def compute_hw_diagnostics(self) -> dict:
        """Compute Hasegawa-Wakatani specific diagnostics.

        Returns:
            Dictionary with HW diagnostics:
            - energy: Kinetic energy ∫|∇φ|²dx
            - enstrophy: ∫ζ²dx
            - density_variance: ∫n²dx
            - particle_flux: Γ = <n·vx> (radial transport)
            - zonal_energy: Energy in ky=0 modes (zonal flows)
            - coupling_strength: |φ - n| (measures instability activity)
        """
        dx, dy = self.grid.dx, self.grid.dy
        dA = dx * dy

        # Standard energy and enstrophy
        diag = compute_diagnostics(self.grid)

        # Density variance
        density_var = 0.5 * np.sum(self.n_grid**2) * dA

        # Particle flux: Γ = <ñ·ṽx> (radial particle transport)
        # This is the key observable for HW turbulence
        particle_flux = np.mean(self.n_grid * self.grid.vx)

        # Zonal flow energy (ky = 0 modes)
        phi_hat = fft2(self.grid.phi)
        zonal_mask = np.abs(self.KY) < 1e-10
        zonal_energy = np.sum(np.abs(phi_hat[zonal_mask])**2)
        zonal_energy /= (self.grid.nx * self.grid.ny)**2

        # Coupling strength: measures how much φ and n have decoupled
        # When this is large, the instability is active
        coupling_strength = np.sqrt(np.mean((self.grid.phi - self.n_grid)**2))

        return {
            'energy': diag['energy'],
            'enstrophy': diag['enstrophy'],
            'density_variance': density_var,
            'particle_flux': particle_flux,
            'zonal_energy': zonal_energy,
            'coupling_strength': coupling_strength,
            'max_vorticity': np.max(np.abs(self.grid.q)),
            'max_density': np.max(np.abs(self.n_grid)),
        }

    def run_hw(self,
               n_steps: int,
               diag_interval: int = 10,
               callback: Optional[Callable] = None,
               verbose: bool = True,
               progress: bool = True):
        """Run Hasegawa-Wakatani simulation.

        Args:
            n_steps: Number of time steps
            diag_interval: Steps between diagnostics
            callback: Optional callback(sim) called at diag_interval
            verbose: Print progress
            progress: Show tqdm progress bar
        """
        iterator = tqdm(range(n_steps), desc="HW Simulation", disable=not progress)
        for i in iterator:
            self.step_hw()

            if (self.step % diag_interval == 0) or (i == n_steps - 1):
                diag = self.compute_hw_diagnostics()
                max_jac = self.integrator.estimate_error(self.flow_map)

                self.history['time'].append(self.time)
                self.history['energy'].append(diag['energy'])
                self.history['enstrophy'].append(diag['enstrophy'])
                self.history['density_variance'].append(diag['density_variance'])
                self.history['particle_flux'].append(diag['particle_flux'])
                self.history['zonal_energy'].append(diag['zonal_energy'])
                self.history['max_q'].append(diag['max_vorticity'])
                self.history['max_jacobian_dev'].append(max_jac)

                if verbose and self.step % (10 * diag_interval) == 0:
                    print(f"Step {self.step:5d}, t={self.time:.2f}, "
                          f"E={diag['energy']:.4f}, Γ={diag['particle_flux']:.2e}, "
                          f"ZF={diag['zonal_energy']:.2e}, |φ-n|={diag['coupling_strength']:.3f}")

                if callback is not None:
                    callback(self)

    def get_field(self, name: str) -> np.ndarray:
        """Get a grid field by name."""
        return getattr(self.grid, name)


# Backward compatibility alias
SimulationV2 = Simulation


# =============================================================================
# Helper functions for initial conditions
# =============================================================================

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
