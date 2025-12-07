"""VPFM-Plasma: Vortex Particle Flow Maps for Plasma Turbulence Simulation"""

from .grid import Grid
from .particles import ParticleSystem
from .transfers import P2G, G2P
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .integrator import RK4Integrator
from .diagnostics import compute_diagnostics
from .simulation import Simulation
from .hasegawa_wakatani import HWSimulation, DensityParticles
from .arakawa import arakawa_jacobian, compute_poisson_bracket
from .flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics

__version__ = "0.2.0"
__all__ = [
    # Core components
    "Grid",
    "ParticleSystem",
    "P2G",
    "G2P",
    "solve_poisson_hm",
    "compute_velocity",
    "compute_velocity_gradient",
    "RK4Integrator",
    "compute_diagnostics",
    # Simulations
    "Simulation",          # Hasegawa-Mima
    "HWSimulation",        # Hasegawa-Wakatani
    "DensityParticles",
    # Enstrophy-conserving schemes
    "arakawa_jacobian",
    "compute_poisson_bracket",
    # Diagnostics
    "VirtualProbe",
    "BlobDetector",
    "FluxStatistics",
]
