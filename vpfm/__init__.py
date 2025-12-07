"""VPFM-Plasma: Vortex Particle Flow Maps for Plasma Turbulence Simulation"""

from .grid import Grid
from .particles import ParticleSystem
from .transfers import P2G, G2P
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .integrator import RK4Integrator
from .diagnostics import compute_diagnostics
from .simulation import Simulation
from .simulation_v2 import SimulationV2
from .hasegawa_wakatani import HWSimulation, DensityParticles
from .arakawa import arakawa_jacobian, compute_poisson_bracket
from .flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics
from .kernels import InterpolationKernel, P2G_bspline, G2P_bspline
from .flow_map import FlowMapIntegrator, FlowMapState

__version__ = "0.3.0"
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
    "Simulation",          # Hasegawa-Mima (v1 - linear/Euler)
    "SimulationV2",        # Hasegawa-Mima (v2 - B-spline/RK4)
    "HWSimulation",        # Hasegawa-Wakatani
    "DensityParticles",
    # Higher-order methods
    "InterpolationKernel",
    "P2G_bspline",
    "G2P_bspline",
    "FlowMapIntegrator",
    "FlowMapState",
    # Enstrophy-conserving schemes
    "arakawa_jacobian",
    "compute_poisson_bracket",
    # Diagnostics
    "VirtualProbe",
    "BlobDetector",
    "FluxStatistics",
]
