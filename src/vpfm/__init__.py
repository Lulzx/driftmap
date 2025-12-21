"""VPFM-Plasma: Vortex Particle Flow Maps for Plasma Turbulence Simulation"""

from .core.grid import Grid
from .core.particles import ParticleSystem
from .core.transfers import P2G, G2P
from .numerics.poisson import solve_poisson_hm
from .numerics.velocity import compute_velocity, compute_velocity_gradient
from .core.integrator import RK4Integrator
from .diagnostics.diagnostics import compute_diagnostics
from .diagnostics.spectral import energy_spectrum, density_flux_spectrum, fit_power_law
from .diagnostics.flow_topology import compute_weiss_field, partition_flow, compute_residence_times
from .physics.simulation import Simulation, SimulationV2  # Backward compatibility alias
from .physics.initial_conditions import (
    lamb_oseen,
    vortex_pair,
    kelvin_helmholtz,
    random_turbulence,
)
from .physics.hasegawa_wakatani import HWSimulation, DensityParticles
from .physics.arakawa import arakawa_jacobian, compute_poisson_bracket
from .diagnostics.flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics
from .core.kernels import (
    InterpolationKernel, P2G_bspline, G2P_bspline,
    P2G_bspline_gradient_enhanced, G2P_bspline_with_gradient
)
from .core.flow_map import (
    FlowMapIntegrator, FlowMapState,
    DualScaleFlowMapIntegrator, DualScaleFlowMapState,
)

# Backend abstraction (CPU/GPU)
from .backends.backend import (
    get_backend, set_backend, get_backend_name,
    to_cpu, to_gpu, is_gpu_backend, synchronize,
)

# GPU kernels - CUDA (optional, requires CuPy)
from .backends.kernels_gpu import (
    check_gpu_available as check_cuda_available,
    P2G_gpu,
    G2P_gpu,
    jacobian_rhs_gpu,
    rk4_positions_gpu,
)

# GPU kernels - MLX (optional, requires MLX on Apple Silicon)
from .backends.kernels_mlx import (
    check_mlx_available,
    P2G_mlx,
    G2P_mlx,
    jacobian_rhs_mlx,
    rk4_positions_mlx,
    solve_poisson_mlx,
    to_mlx,
    to_numpy,
)

# 3D extension
from .physics.simulation3d import (
    Simulation3D,
    Grid3D,
    Particles3D,
    FlowMap3D,
    gaussian_blob_3d,
    random_turbulence_3d,
)

__version__ = "0.5.0"
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
    "energy_spectrum",
    "density_flux_spectrum",
    "fit_power_law",
    "compute_weiss_field",
    "partition_flow",
    "compute_residence_times",
    # Simulation (unified HM + HW)
    "Simulation",
    "SimulationV2",  # Backward compatibility alias
    # Initial conditions
    "lamb_oseen",
    "vortex_pair",
    "kelvin_helmholtz",
    "random_turbulence",
    # Legacy HW simulation (deprecated)
    "HWSimulation",
    "DensityParticles",
    # Higher-order methods
    "InterpolationKernel",
    "P2G_bspline",
    "G2P_bspline",
    "P2G_bspline_gradient_enhanced",
    "G2P_bspline_with_gradient",
    "FlowMapIntegrator",
    "FlowMapState",
    "DualScaleFlowMapIntegrator",
    "DualScaleFlowMapState",
    # Enstrophy-conserving schemes
    "arakawa_jacobian",
    "compute_poisson_bracket",
    # Diagnostics
    "VirtualProbe",
    "BlobDetector",
    "FluxStatistics",
    # Backend (CPU/GPU)
    "get_backend",
    "set_backend",
    "get_backend_name",
    "to_cpu",
    "to_gpu",
    "is_gpu_backend",
    "synchronize",
    # GPU kernels - CUDA
    "check_cuda_available",
    "P2G_gpu",
    "G2P_gpu",
    "jacobian_rhs_gpu",
    "rk4_positions_gpu",
    # GPU kernels - MLX (Apple Silicon)
    "check_mlx_available",
    "P2G_mlx",
    "G2P_mlx",
    "jacobian_rhs_mlx",
    "rk4_positions_mlx",
    "solve_poisson_mlx",
    "to_mlx",
    "to_numpy",
    # 3D extension
    "Simulation3D",
    "Grid3D",
    "Particles3D",
    "FlowMap3D",
    "gaussian_blob_3d",
    "random_turbulence_3d",
]
