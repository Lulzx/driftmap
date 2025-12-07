"""VPFM-Plasma: Vortex Particle Flow Maps for Plasma Turbulence Simulation"""

from .grid import Grid
from .particles import ParticleSystem
from .transfers import P2G, G2P
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .integrator import RK4Integrator
from .diagnostics import compute_diagnostics
from .simulation import Simulation

__version__ = "0.1.0"
__all__ = [
    "Grid",
    "ParticleSystem",
    "P2G",
    "G2P",
    "solve_poisson_hm",
    "compute_velocity",
    "compute_velocity_gradient",
    "RK4Integrator",
    "compute_diagnostics",
    "Simulation",
]
