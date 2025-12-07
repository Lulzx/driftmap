"""VPFM-Plasma: Vortex Particle Flow Maps for Plasma Turbulence Simulation"""

from .grid import Grid
from .particles import ParticleSystem
from .transfers import P2G, G2P
from .poisson import solve_poisson_hm
from .velocity import compute_velocity, compute_velocity_gradient
from .integrator import RK4Integrator
from .diagnostics import compute_diagnostics
from .simulation import (
    Simulation,
    SimulationV2,  # Backward compatibility alias
    lamb_oseen,
    vortex_pair,
    random_turbulence,
)
from .hasegawa_wakatani import HWSimulation, DensityParticles
from .arakawa import arakawa_jacobian, compute_poisson_bracket
from .flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics
from .kernels import InterpolationKernel, P2G_bspline, G2P_bspline
from .flow_map import FlowMapIntegrator, FlowMapState

__version__ = "0.4.0"
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
    # Simulation (unified HM + HW)
    "Simulation",
    "SimulationV2",  # Backward compatibility alias
    # Initial conditions
    "lamb_oseen",
    "vortex_pair",
    "random_turbulence",
    # Legacy HW simulation (deprecated)
    "HWSimulation",
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
