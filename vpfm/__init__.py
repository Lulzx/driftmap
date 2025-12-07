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
    kelvin_helmholtz,
    random_turbulence,
)
from .hasegawa_wakatani import HWSimulation, DensityParticles
from .arakawa import arakawa_jacobian, compute_poisson_bracket
from .flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics
from .kernels import (
    InterpolationKernel, P2G_bspline, G2P_bspline,
    P2G_bspline_gradient_enhanced, G2P_bspline_with_gradient
)
from .flow_map import (
    FlowMapIntegrator, FlowMapState,
    DualScaleFlowMapIntegrator, DualScaleFlowMapState,
)

# Backend abstraction (CPU/GPU)
from .backend import (
    get_backend, set_backend, get_backend_name,
    to_cpu, to_gpu, is_gpu_backend, synchronize,
)

# GPU kernels - CUDA (optional, requires CuPy)
from .kernels_gpu import (
    check_gpu_available as check_cuda_available,
    P2G_gpu,
    G2P_gpu,
    jacobian_rhs_gpu,
    rk4_positions_gpu,
)

# GPU kernels - MLX (optional, requires MLX on Apple Silicon)
from .kernels_mlx import (
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
from .simulation3d import (
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
