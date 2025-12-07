# VPFM-Plasma

**Vortex Particle Flow Maps for Plasma Edge Turbulence Simulation**

An implementation adapting [Vortex Particle Flow Maps](https://arxiv.org/abs/2505.21946) to simulate plasma turbulence in the scrape-off layer (SOL) of tokamak fusion reactors.

## Overview

This project demonstrates the VPFM method's advantages for plasma turbulence simulation:

- **Better structure preservation**: VPFM maintains vortex structures (blobs) longer than finite-difference methods
- **Material conservation**: Exploits the fact that potential vorticity is materially conserved
- **Reduced numerical dissipation**: Lagrangian particles avoid grid-scale diffusion
- **Correct zonal flow physics**: Arakawa scheme conserves both energy and enstrophy

### The Mathematical Isomorphism

The key insight is that **potential vorticity in drift-wave turbulence obeys the same material conservation law as vorticity in incompressible fluids**:

- In 2D incompressible flow: Dω/Dt = 0 (inviscid limit)
- In drift-wave turbulence: D(∇²φ - φ)/Dt ≈ 0 (adiabatic electron limit)

Both are advected by an incompressible velocity field (physical velocity or E×B drift), making VPFM directly applicable.

## Physics Models

### Hasegawa-Mima (Basic)

```
∂q/∂t + {φ, q} = 0
```

where q = ∇²φ - φ is the potential vorticity.

### Hasegawa-Wakatani (Full)

```
∂ζ/∂t + {φ, ζ} = α(φ - n) + μ∇⁴ζ - ν_sheath·ζ
∂n/∂t + {φ, n} = α(φ - n) - κ·∂φ/∂y + D∇²n
```

Features:
- **Resistive coupling α(φ - n)**: Drives the drift-wave instability
- **Curvature drive κ·∂φ/∂y**: Interchange instability
- **Sheath damping ν_sheath**: Parallel losses to divertor
- **Zonal flow generation**: Self-consistent turbulence saturation

## Results

### Benchmark Summary

![Benchmark Results](benchmark_results.png)

### Material Conservation Test

The core advantage of VPFM is **exact material conservation** on Lagrangian particles. Testing with a Lamb-Oseen (Gaussian) vortex:

| Method | Peak @ t=5.0 | Peak @ t=10.0 |
|--------|--------------|---------------|
| **VPFM** | **100.0%** | **100.0%** |
| FD Upwind | 49.3% | 34.0% |
| FD Central | 100.0% | ~100% |

**VPFM achieves perfect material conservation** - particles preserve potential vorticity exactly, while finite-difference upwind loses over 50% of peak vorticity due to numerical diffusion.

### Lamb-Oseen Vortex Test

Single Gaussian vortex stability comparing VPFM vs finite-difference:

![Lamb-Oseen Comparison](lamb_oseen_comparison.png)

| Method | Peak Preservation |
|--------|-------------------|
| VPFM | 100.0% |
| FD Upwind | 49.3% |
| FD Central | 100.0% |

### Decaying Turbulence Test

Random initial condition with energy/enstrophy conservation (t=3.0):

![Turbulence Comparison](turbulence_comparison.png)

| Metric | VPFM | FD Upwind | VPFM Advantage |
|--------|------|-----------|----------------|
| Energy error | 0.21% | 2.76% | **13x better** |
| Enstrophy error | 0.34% | 3.94% | **12x better** |

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.4
- pytest >= 6.0

## Usage

### Hasegawa-Mima (Quick Start)

```python
from vpfm import Simulation, lamb_oseen
import numpy as np

# Create simulation
sim = Simulation(nx=128, ny=128, Lx=2*np.pi, Ly=2*np.pi, dt=0.01)

# Set initial condition (Gaussian vortex)
def ic(x, y):
    return lamb_oseen(x, y, np.pi, np.pi, Gamma=2*np.pi, r0=0.5)

sim.set_initial_condition(ic)
sim.run(n_steps=1000, diag_interval=10, verbose=True)
```

### Higher-Order Methods (SimulationV2)

```python
from vpfm import SimulationV2
import numpy as np

# Create simulation with B-spline kernels and RK4 Jacobian evolution
sim = SimulationV2(
    nx=128, ny=128, Lx=2*np.pi, Ly=2*np.pi, dt=0.01,
    kernel_order='quadratic',  # 'linear', 'quadratic', or 'cubic'
    track_hessian=True,        # Track Hessian for gradient accuracy
    reinit_threshold=2.0,      # ||J-I|| threshold for reinitialization
    max_reinit_steps=200,      # Max steps between reinits
)

sim.set_initial_condition(ic)
sim.run(n_steps=1000, diag_interval=10, verbose=True)
```

### Hasegawa-Wakatani (Full Physics)

```python
from vpfm import HWSimulation
import numpy as np

sim = HWSimulation(
    nx=128, ny=128, Lx=40*np.pi, Ly=40*np.pi, dt=0.02,
    alpha=0.5,      # Adiabaticity
    kappa=0.05,     # Curvature drive
    nu_sheath=0.01, # Sheath damping
)

# Set random perturbation to seed instability
from vpfm.hasegawa_wakatani import hw_random_perturbation
# ... (see examples/run_hasegawa_wakatani.py)
```

### Flux Diagnostics

```python
from vpfm import VirtualProbe, BlobDetector

# Virtual Langmuir probe
probe = VirtualProbe(x_pos=Lx/2, y_range=(0, Ly))

# During simulation
probe.measure(sim.time, sim.n_grid, sim.grid.vx, sim.grid.x, sim.grid.y)

# Get statistics
stats = probe.compute_statistics()
print(f"Skewness: {stats.skewness:.2f}")  # Compare with MAST-U data!
```

### Running Examples

```bash
# Lamb-Oseen vortex (structure preservation)
python examples/run_lamb_oseen.py

# Vortex pair dynamics
python examples/run_leapfrog.py

# Decaying turbulence (conservation)
python examples/run_turbulence.py

# Full Hasegawa-Wakatani turbulence
python examples/run_hasegawa_wakatani.py
```

## Project Structure

```
driftmap/
├── vpfm/                      # Core VPFM implementation
│   ├── grid.py               # Eulerian grid
│   ├── particles.py          # Lagrangian vortex particles
│   ├── transfers.py          # P2G and G2P operations (linear)
│   ├── kernels.py            # B-spline interpolation kernels
│   ├── poisson.py            # FFT Poisson solver
│   ├── velocity.py           # E×B velocity computation
│   ├── integrator.py         # RK4 time integration
│   ├── flow_map.py           # Advanced flow map (RK4 Jacobian, Hessian)
│   ├── diagnostics.py        # Energy, enstrophy metrics
│   ├── simulation.py         # Hasegawa-Mima simulation (v1)
│   ├── simulation_v2.py      # Improved simulation (B-spline/RK4)
│   ├── hasegawa_wakatani.py  # Full HW model
│   ├── arakawa.py            # Enstrophy-conserving Jacobian
│   └── flux_diagnostics.py   # Virtual probes, blob detection
├── baseline/                  # Finite difference comparison
├── tests/                     # Unit and integration tests
├── examples/                  # Example scripts
└── requirements.txt
```

## Algorithm

The VPFM algorithm per timestep:

1. **P2G Transfer**: Interpolate vorticity from particles to grid
2. **Poisson Solve**: Solve ∇²φ = ζ for potential
3. **Velocity Computation**: v = ẑ × ∇φ (E×B drift)
4. **Source Terms**: Apply α(φ-n), curvature drive, dissipation
5. **Particle Advection**: RK4 integration of positions
6. **Jacobian Evolution**: dJ/dt = -J·∇v
7. **Reinitialization**: Reset flow map when ||J-I|| exceeds threshold

## Experimental Validation Targets

Compare simulation results with:

- **MAST-U**: Fast camera blob imaging
- **ASDEX-Upgrade**: Lithium beam emission
- **NSTX-U**: Gas puff imaging

Typical experimental values:
- Flux skewness: 0.5 - 2.0 (positive, bursty outward transport)
- Flux kurtosis: 1 - 10 (heavy tails)
- Blob size: 1-5 cm (several ρ_s)

## Key Features

| Feature | Status |
|---------|--------|
| Hasegawa-Mima equation | ✅ |
| Hasegawa-Wakatani equation | ✅ |
| Arakawa enstrophy conservation | ✅ |
| B-spline interpolation kernels | ✅ |
| RK4 Jacobian evolution | ✅ |
| Hessian tracking | ✅ |
| Adaptive reinitialization | ✅ |
| Sheath boundary damping | ✅ |
| Virtual probe diagnostics | ✅ |
| Blob detection | ✅ |
| Zonal flow analysis | ✅ |
| GPU acceleration | 🔜 |
| 3D extension | 🔜 |

## References

1. Wang, S., et al. (2025). "Fluid Simulation on Vortex Particle Flow Maps." [arXiv:2505.21946](https://arxiv.org/abs/2505.21946)
2. Hasegawa, A. & Mima, K. (1978). "Pseudo-three-dimensional turbulence in magnetized nonuniform plasma." Physics of Fluids 21, 87.
3. Hasegawa, A. & Wakatani, M. (1983). "Plasma edge turbulence." Physical Review Letters 50, 682.
4. Arakawa, A. (1966). "Computational design for long-term numerical integration of the equations of fluid motion." J. Comp. Phys. 1, 119-143.

## License

MIT
