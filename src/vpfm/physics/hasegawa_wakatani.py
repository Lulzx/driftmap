"""Hasegawa-Wakatani model implementation for VPFM-Plasma.

Extends the Hasegawa-Mima model with density coupling:

    ∂ζ/∂t + {φ, ζ} = α(φ - n) + μ∇⁴ζ - ν_sheath·ζ
    ∂n/∂t + {φ, n} = α(φ - n) - κ·∂φ/∂y + D∇²n

Where:
    ζ = ∇²φ is the vorticity
    φ is the electrostatic potential
    n is the density perturbation
    α is the adiabaticity parameter (parallel resistivity)
    κ is the background density gradient (curvature drive)
    μ is the hyperviscosity coefficient
    D is the density diffusivity
    ν_sheath is the sheath damping rate

The α(φ - n) term couples vorticity and density, driving the resistive
drift-wave instability. This is the "baroclinic source" in the vortex analogy.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from typing import Optional, Callable

from ..core.grid import Grid
from ..core.particles import ParticleSystem
from ..core.transfers import P2G_vectorized, G2P
from ..numerics.poisson import solve_poisson_hm
from ..numerics.velocity import compute_velocity, compute_velocity_gradient
from ..core.integrator import RK4Integrator
from ..diagnostics.diagnostics import compute_diagnostics


class DensityParticles:
    """Lagrangian density tracer particles.

    Separate from vortex particles to allow different particle counts
    and potentially different interpolation strategies.
    """

    def __init__(self, n_particles: int):
        self.n_particles = n_particles
        self.x = np.zeros(n_particles)
        self.y = np.zeros(n_particles)
        self.n = np.zeros(n_particles)  # Density perturbation

    @classmethod
    def from_grid(cls, grid: Grid, particles_per_cell: int = 1) -> "DensityParticles":
        """Seed density particles uniformly."""
        n_particles = grid.nx * grid.ny * particles_per_cell
        dp = cls(n_particles)

        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        dp.x = X.flatten()
        dp.y = Y.flatten()

        return dp

    def wrap_positions(self, Lx: float, Ly: float):
        """Apply periodic BCs."""
        self.x = self.x % Lx
        self.y = self.y % Ly


class HWSimulation:
    """Hasegawa-Wakatani simulation using VPFM.

    This extends the basic VPFM with:
    1. Density particles coupled to vorticity
    2. Resistive coupling term α(φ - n)
    3. Curvature drive κ·∂φ/∂y
    4. Sheath damping ν_sheath·ζ
    5. Hyperviscosity and diffusion
    """

    def __init__(self,
                 nx: int = 128,
                 ny: int = 128,
                 Lx: float = 20 * np.pi,  # Larger domain for HW
                 Ly: float = 20 * np.pi,
                 dt: float = 0.01,
                 # Physics parameters
                 alpha: float = 1.0,      # Adiabaticity
                 kappa: float = 0.05,     # Curvature drive
                 mu: float = 1e-4,        # Hyperviscosity
                 nu: float = 0.0,         # Viscosity (Laplacian)
                 D: float = 1e-4,         # Density diffusion
                 nu_sheath: float = 0.0,  # Sheath damping
                 particles_per_cell: int = 1):
        """Initialize Hasegawa-Wakatani simulation.

        Args:
            nx, ny: Grid resolution
            Lx, Ly: Domain size (in units of ρ_s)
            dt: Time step
            alpha: Adiabaticity parameter (α → ∞ is HM limit)
            kappa: Background density gradient / curvature drive
            mu: Hyperviscosity coefficient
            nu: Viscosity coefficient
            D: Density diffusivity
            nu_sheath: Sheath damping rate for parallel losses
            particles_per_cell: Particle density
        """
        self.grid = Grid(nx, ny, Lx, Ly)
        self.vortex_particles = ParticleSystem.from_grid(self.grid, particles_per_cell)
        self.density_particles = DensityParticles.from_grid(self.grid, particles_per_cell)
        self.integrator = RK4Integrator(self.grid)

        # Physics parameters
        self.alpha = alpha
        self.kappa = kappa
        self.mu = mu
        self.nu = nu
        self.D = D
        self.nu_sheath = nu_sheath

        self.time = 0.0
        self.step = 0
        self.dt = dt

        # Flow map parameters
        self.reinit_interval = 20
        self.reinit_threshold = 0.5

        # Grid fields for density
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
            'density_variance': [],
            'particle_flux': [],
            'zonal_energy': [],
        }

    def set_initial_condition(self,
                              zeta_func: Callable,
                              n_func: Optional[Callable] = None):
        """Set initial conditions for vorticity and density.

        Args:
            zeta_func: Function ζ(x, y) for initial vorticity
            n_func: Function n(x, y) for initial density (defaults to ζ)
        """
        # Set vorticity on particles
        self.vortex_particles.q = zeta_func(self.vortex_particles.x,
                                             self.vortex_particles.y)

        # Set density (default to vorticity if not specified)
        if n_func is None:
            n_func = zeta_func
        self.density_particles.n = n_func(self.density_particles.x,
                                          self.density_particles.y)

        # Transfer to grid
        self._transfer_to_grid()
        self._solve_fields()

    def _transfer_to_grid(self):
        """Transfer particle quantities to grid."""
        # Vorticity
        self.grid.q = P2G_vectorized(self.vortex_particles, self.grid)

        # Density (reuse P2G with density particles)
        # Create temporary particle system for density transfer
        temp_ps = ParticleSystem(self.density_particles.n_particles)
        temp_ps.x = self.density_particles.x
        temp_ps.y = self.density_particles.y
        temp_ps.q = self.density_particles.n
        self.n_grid = P2G_vectorized(temp_ps, self.grid)

    def _solve_fields(self):
        """Solve for potential and velocity."""
        # For HW, we solve ∇²φ = ζ (not HM modified Poisson)
        # Using standard Poisson since ζ = ∇²φ directly
        zeta_hat = fft2(self.grid.q)

        # Avoid division by zero
        K2_safe = self.K2.copy()
        K2_safe[0, 0] = 1.0

        phi_hat = -zeta_hat / K2_safe
        phi_hat[0, 0] = 0  # Zero mean

        self.grid.phi = np.real(ifft2(phi_hat))

        compute_velocity(self.grid)
        compute_velocity_gradient(self.grid)

    def _compute_coupling_source(self) -> tuple:
        """Compute the α(φ - n) coupling term.

        Returns:
            (S_zeta, S_n) source terms for vorticity and density
        """
        coupling = self.alpha * (self.grid.phi - self.n_grid)

        S_zeta = coupling      # Source for vorticity equation
        S_n = coupling         # Source for density equation (same term)

        return S_zeta, S_n

    def _compute_curvature_drive(self) -> np.ndarray:
        """Compute the curvature drive term κ·∂φ/∂y.

        This drives the interchange instability.
        """
        phi_hat = fft2(self.grid.phi)
        dphi_dy = np.real(ifft2(1j * self.KY * phi_hat))

        return -self.kappa * dphi_dy

    def _apply_hyperviscosity(self, field: np.ndarray, coeff: float) -> np.ndarray:
        """Apply hyperviscosity: -μ∇⁴ field.

        Dissipates small scales while preserving large-scale dynamics.
        """
        if coeff < 1e-10:
            return np.zeros_like(field)

        field_hat = fft2(field)
        dissipation = np.real(ifft2(-coeff * self.K4 * field_hat))

        return dissipation

    def _apply_diffusion(self, field: np.ndarray, coeff: float) -> np.ndarray:
        """Apply standard diffusion: D∇² field."""
        if coeff < 1e-10:
            return np.zeros_like(field)

        field_hat = fft2(field)
        diffusion = np.real(ifft2(-coeff * self.K2 * field_hat))

        return diffusion

    def _apply_sheath_damping(self, field: np.ndarray) -> np.ndarray:
        """Apply sheath damping: -ν_sheath · field.

        Models parallel losses to the divertor.
        """
        return -self.nu_sheath * field

    def advance(self):
        """Advance simulation by one timestep."""
        # 1. Transfer particles to grid
        self._transfer_to_grid()

        # 2. Solve for potential and velocity
        self._solve_fields()

        # 3. Compute source terms on grid
        S_zeta, S_n = self._compute_coupling_source()
        curvature = self._compute_curvature_drive()

        # Add dissipation terms
        S_zeta += self._apply_hyperviscosity(self.grid.q, self.mu)
        S_zeta += self._apply_diffusion(self.grid.q, self.nu)
        S_zeta += self._apply_sheath_damping(self.grid.q)

        S_n += curvature
        S_n += self._apply_diffusion(self.n_grid, self.D)

        # 4. Update particle strengths from grid sources
        # Interpolate source terms to particle positions
        dq_dt = G2P(self.grid, self.vortex_particles, S_zeta)
        self.vortex_particles.q += self.dt * dq_dt

        temp_ps = ParticleSystem(self.density_particles.n_particles)
        temp_ps.x = self.density_particles.x
        temp_ps.y = self.density_particles.y
        dn_dt = G2P(self.grid, temp_ps, S_n)
        self.density_particles.n += self.dt * dn_dt

        # 5. Advect particles
        self.integrator.advect_particles(self.vortex_particles, self.dt)

        # Advect density particles with same velocity
        # (they move with ExB drift)
        x0 = self.density_particles.x.copy()
        y0 = self.density_particles.y.copy()

        temp_ps.x = x0
        temp_ps.y = y0
        vx = G2P(self.grid, temp_ps, self.grid.vx)
        vy = G2P(self.grid, temp_ps, self.grid.vy)

        self.density_particles.x = (x0 + self.dt * vx) % self.grid.Lx
        self.density_particles.y = (y0 + self.dt * vy) % self.grid.Ly

        # 6. Evolve Jacobians
        self.integrator.evolve_jacobian_euler(self.vortex_particles, self.dt)

        # 7. Update time
        self.time += self.dt
        self.step += 1

        # 8. Reinitialization check
        if self.step % self.reinit_interval == 0:
            if self.vortex_particles.max_jacobian_deviation() > self.reinit_threshold:
                self._reinitialize()

    def _reinitialize(self):
        """Reinitialize flow map."""
        # Transfer to grid
        self._transfer_to_grid()

        # Transfer back to particles
        self.vortex_particles.q = G2P(self.grid, self.vortex_particles, self.grid.q)

        temp_ps = ParticleSystem(self.density_particles.n_particles)
        temp_ps.x = self.density_particles.x
        temp_ps.y = self.density_particles.y
        self.density_particles.n = G2P(self.grid, temp_ps, self.n_grid)

        # Reset Jacobians
        self.vortex_particles.reset_jacobian()

    def compute_diagnostics(self) -> dict:
        """Compute HW-specific diagnostics."""
        dx, dy = self.grid.dx, self.grid.dy
        dA = dx * dy

        # Standard energy
        dphi_dx = (np.roll(self.grid.phi, -1, axis=0) -
                   np.roll(self.grid.phi, 1, axis=0)) / (2 * dx)
        dphi_dy = (np.roll(self.grid.phi, -1, axis=1) -
                   np.roll(self.grid.phi, 1, axis=1)) / (2 * dy)
        energy = 0.5 * np.sum(dphi_dx**2 + dphi_dy**2) * dA

        # Enstrophy
        enstrophy = 0.5 * np.sum(self.grid.q**2) * dA

        # Density variance
        density_var = 0.5 * np.sum(self.n_grid**2) * dA

        # Particle flux: Γ = <ñ ṽ_x> (radial flux)
        particle_flux = np.mean(self.n_grid * self.grid.vx)

        # Zonal flow energy (k_y = 0 modes of φ)
        phi_hat = fft2(self.grid.phi)
        zonal_mask = np.abs(self.KY) < 1e-10
        zonal_energy = np.sum(np.abs(phi_hat[zonal_mask])**2) / (self.grid.nx * self.grid.ny)**2

        return {
            'energy': energy,
            'enstrophy': enstrophy,
            'density_variance': density_var,
            'particle_flux': particle_flux,
            'zonal_energy': zonal_energy,
        }

    def run(self, n_steps: int, diag_interval: int = 10, verbose: bool = True):
        """Run simulation."""
        for i in range(n_steps):
            self.advance()

            if (self.step % diag_interval == 0) or (i == n_steps - 1):
                diag = self.compute_diagnostics()

                self.history['time'].append(self.time)
                self.history['energy'].append(diag['energy'])
                self.history['enstrophy'].append(diag['enstrophy'])
                self.history['density_variance'].append(diag['density_variance'])
                self.history['particle_flux'].append(diag['particle_flux'])
                self.history['zonal_energy'].append(diag['zonal_energy'])

                if verbose and self.step % (10 * diag_interval) == 0:
                    print(f"Step {self.step:5d}, t={self.time:.2f}, "
                          f"E={diag['energy']:.4f}, Γ={diag['particle_flux']:.4e}, "
                          f"ZF={diag['zonal_energy']:.4e}")


def hw_random_perturbation(nx: int, ny: int, Lx: float, Ly: float,
                           k_peak: float = 3.0, amplitude: float = 0.01,
                           seed: int = 42) -> np.ndarray:
    """Generate random perturbation for HW instability seeding.

    Small-amplitude noise to seed the instability.
    """
    np.random.seed(seed)

    dx, dy = Lx / nx, Ly / ny
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # Spectrum peaked at k_peak
    E_k = K**2 * np.exp(-(K / k_peak)**2)

    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, (nx, ny))

    field_hat = np.sqrt(E_k) * np.exp(1j * phases)
    field_hat[0, 0] = 0

    return amplitude * np.real(ifft2(field_hat))
