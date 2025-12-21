"""Finite difference baseline solver for VPFM comparison.

Implements standard upwind/central difference advection for
Hasegawa-Mima equation to compare against VPFM.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from typing import Callable, Optional


class FiniteDifferenceSimulation:
    """Finite difference solver for Hasegawa-Mima equation.

    Supports upwind, central, and Arakawa advection schemes for comparison
    against VPFM.

    Equation: dq/dt + {phi, q} = 0
    where q = nabla^2(phi) - phi

    Attributes:
        nx, ny: Grid resolution
        Lx, Ly: Domain size
        dx, dy: Grid spacing
        nu: Viscosity coefficient for ∇²q
        q: Potential vorticity field
        phi: Electrostatic potential
        time: Current simulation time
        step: Current step number
    """

    def __init__(self,
                 nx: int = 128,
                 ny: int = 128,
                 Lx: float = 2 * np.pi,
                 Ly: float = 2 * np.pi,
                 dt: float = 0.01,
                 nu: float = 0.0):
        """Initialize simulation.

        Args:
            nx, ny: Grid resolution
            Lx, Ly: Domain size
            dt: Time step
            nu: Viscosity coefficient for ∇²q
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.nu = nu

        # Fields
        self.q = np.zeros((nx, ny))
        self.phi = np.zeros((nx, ny))
        self.vx = np.zeros((nx, ny))
        self.vy = np.zeros((nx, ny))

        # Grid coordinates
        x = np.linspace(self.dx/2, Lx - self.dx/2, nx)
        y = np.linspace(self.dy/2, Ly - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        self.time = 0.0
        self.step = 0

        # Diagnostic history
        self.history = {
            'time': [],
            'energy': [],
            'enstrophy': [],
            'pot_enstrophy': [],
            'max_q': [],
        }

    def set_initial_condition(self, q_func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Set initial condition from a function.

        Args:
            q_func: Function q(x, y) returning potential vorticity
        """
        self.q = q_func(self.X, self.Y)
        self._solve_poisson()
        self._compute_velocity()

    def _solve_poisson(self):
        """Solve Hasegawa-Mima Poisson equation: (nabla^2 - 1) phi = -q"""
        kx = fftfreq(self.nx, self.dx) * 2 * np.pi
        ky = fftfreq(self.ny, self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        q_hat = fft2(self.q)
        K2 = KX**2 + KY**2 + 1

        phi_hat = q_hat / K2
        phi_hat[0, 0] = 0

        self.phi = np.real(ifft2(phi_hat))

    def _compute_velocity(self):
        """Compute E×B velocity from potential."""
        # Spectral derivatives
        kx = fftfreq(self.nx, self.dx) * 2 * np.pi
        ky = fftfreq(self.ny, self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        phi_hat = fft2(self.phi)

        dphi_dx = np.real(ifft2(1j * KX * phi_hat))
        dphi_dy = np.real(ifft2(1j * KY * phi_hat))

        self.vx = -dphi_dy
        self.vy = dphi_dx

    def _upwind_advection(self) -> np.ndarray:
        """Compute dq/dt using upwind advection.

        Returns:
            Time derivative dq/dt
        """
        dx, dy = self.dx, self.dy

        # Upwind derivatives in x
        dq_dx_minus = (self.q - np.roll(self.q, 1, axis=0)) / dx
        dq_dx_plus = (np.roll(self.q, -1, axis=0) - self.q) / dx

        # Upwind derivatives in y
        dq_dy_minus = (self.q - np.roll(self.q, 1, axis=1)) / dy
        dq_dy_plus = (np.roll(self.q, -1, axis=1) - self.q) / dy

        # Select based on velocity sign
        dq_dx = np.where(self.vx > 0, dq_dx_minus, dq_dx_plus)
        dq_dy = np.where(self.vy > 0, dq_dy_minus, dq_dy_plus)

        # Advection: dq/dt = -v · nabla(q)
        dq_dt = -(self.vx * dq_dx + self.vy * dq_dy)

        return dq_dt

    def _central_advection(self) -> np.ndarray:
        """Compute dq/dt using central differences (Arakawa-like).

        Returns:
            Time derivative dq/dt
        """
        dx, dy = self.dx, self.dy

        # Central differences
        dq_dx = (np.roll(self.q, -1, axis=0) - np.roll(self.q, 1, axis=0)) / (2 * dx)
        dq_dy = (np.roll(self.q, -1, axis=1) - np.roll(self.q, 1, axis=1)) / (2 * dy)

        # Advection
        dq_dt = -(self.vx * dq_dx + self.vy * dq_dy)

        return dq_dt

    def _arakawa_advection(self) -> np.ndarray:
        """Compute dq/dt using the Arakawa Jacobian (energy/enstrophy conserving)."""
        dx, dy = self.dx, self.dy

        def ip(f): return np.roll(f, -1, axis=0)
        def im(f): return np.roll(f, 1, axis=0)
        def jp(f): return np.roll(f, -1, axis=1)
        def jm(f): return np.roll(f, 1, axis=1)

        phi = self.phi
        zeta = self.q

        Jpp = ((ip(phi) - im(phi)) * (jp(zeta) - jm(zeta)) -
               (jp(phi) - jm(phi)) * (ip(zeta) - im(zeta))) / (4 * dx * dy)

        Jpx = (ip(phi) * (ip(jp(zeta)) - ip(jm(zeta))) -
               im(phi) * (im(jp(zeta)) - im(jm(zeta))) -
               jp(phi) * (ip(jp(zeta)) - im(jp(zeta))) +
               jm(phi) * (ip(jm(zeta)) - im(jm(zeta)))) / (4 * dx * dy)

        Jxp = (ip(jp(phi)) * (jp(zeta) - ip(zeta)) -
               im(jm(phi)) * (im(zeta) - jm(zeta)) -
               im(jp(phi)) * (jp(zeta) - im(zeta)) +
               ip(jm(phi)) * (ip(zeta) - jm(zeta))) / (4 * dx * dy)

        return -(Jpp + Jpx + Jxp) / 3

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute Laplacian with periodic boundaries."""
        d2x = (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / (self.dx ** 2)
        d2y = (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / (self.dy ** 2)
        return d2x + d2y

    def advance(self, scheme: str = 'upwind'):
        """Advance simulation by one time step.

        Args:
            scheme: 'upwind', 'central', or 'arakawa'
        """
        # RK2 time stepping
        if scheme == 'upwind':
            advect = self._upwind_advection
        elif scheme == 'central':
            advect = self._central_advection
        elif scheme == 'arakawa':
            advect = self._arakawa_advection
        else:
            raise ValueError(f"Unknown advection scheme: {scheme}")

        # Stage 1
        k1 = advect()
        if self.nu > 0:
            k1 = k1 + self.nu * self._laplacian(self.q)
        q_temp = self.q + 0.5 * self.dt * k1

        # Update fields with intermediate q
        self.q = q_temp
        self._solve_poisson()
        self._compute_velocity()

        # Stage 2
        k2 = advect()
        if self.nu > 0:
            k2 = k2 + self.nu * self._laplacian(self.q)
        self.q = self.q - 0.5 * self.dt * k1 + self.dt * k2

        # Final field update
        self._solve_poisson()
        self._compute_velocity()

        self.time += self.dt
        self.step += 1

    def advance_euler(self, scheme: str = 'upwind'):
        """Advance using simple Euler (for comparison).

        Args:
            scheme: 'upwind', 'central', or 'arakawa'
        """
        if scheme == 'upwind':
            dq_dt = self._upwind_advection()
        elif scheme == 'central':
            dq_dt = self._central_advection()
        elif scheme == 'arakawa':
            dq_dt = self._arakawa_advection()
        else:
            raise ValueError(f"Unknown advection scheme: {scheme}")

        if self.nu > 0:
            dq_dt = dq_dt + self.nu * self._laplacian(self.q)
        self.q += self.dt * dq_dt
        self._solve_poisson()
        self._compute_velocity()

        self.time += self.dt
        self.step += 1

    def compute_diagnostics(self) -> dict:
        """Compute physical diagnostics.

        Returns:
            Dictionary with energy, enstrophy, etc.
        """
        dA = self.dx * self.dy

        # Energy
        dphi_dx = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2 * self.dx)
        dphi_dy = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2 * self.dy)
        energy = 0.5 * np.sum(dphi_dx**2 + dphi_dy**2) * dA

        # Vorticity
        zeta = self.q + self.phi

        # Enstrophy
        enstrophy = 0.5 * np.sum(zeta**2) * dA

        # Potential enstrophy
        pot_enstrophy = 0.5 * np.sum(self.q**2) * dA

        return {
            'energy': energy,
            'enstrophy': enstrophy,
            'pot_enstrophy': pot_enstrophy,
            'max_q': np.abs(self.q).max(),
        }

    def run(self,
            n_steps: int,
            diag_interval: int = 10,
            scheme: str = 'upwind',
            verbose: bool = True):
        """Run simulation for specified number of steps.

        Args:
            n_steps: Number of time steps
            diag_interval: Steps between diagnostics
            scheme: 'upwind', 'central', or 'arakawa'
            verbose: Print progress
        """
        for i in range(n_steps):
            self.advance(scheme=scheme)

            if (self.step % diag_interval == 0) or (i == n_steps - 1):
                diag = self.compute_diagnostics()

                self.history['time'].append(self.time)
                self.history['energy'].append(diag['energy'])
                self.history['enstrophy'].append(diag['enstrophy'])
                self.history['pot_enstrophy'].append(diag['pot_enstrophy'])
                self.history['max_q'].append(diag['max_q'])

                if verbose and self.step % (10 * diag_interval) == 0:
                    print(f"[FD] Step {self.step:5d}, t={self.time:.2f}, "
                          f"E={diag['energy']:.6f}, Z={diag['enstrophy']:.6f}")
