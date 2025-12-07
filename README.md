# VPFM-Plasma POC

**Vortex Particle Flow Maps for Plasma Edge Turbulence Simulation**

A proof-of-concept implementation adapting [Vortex Particle Flow Maps](https://arxiv.org/abs/2505.21946) to simulate plasma turbulence using the Hasegawa-Mima equation.

## Overview

This POC demonstrates the VPFM method's advantages for plasma turbulence simulation:

- **Better structure preservation**: VPFM maintains vortex structures longer than finite-difference methods
- **Material conservation**: Exploits the fact that potential vorticity is materially conserved
- **Reduced numerical dissipation**: Lagrangian particles avoid grid-scale diffusion

### Hasegawa-Mima Equation

The simplest relevant plasma turbulence model:

```
∂q/∂t + {φ, q} = 0
```

where:
- q = ∇²φ - φ is the potential vorticity
- φ is the electrostatic potential
- {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x is the Poisson bracket

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

### Quick Start

```python
from vpfm import Simulation, lamb_oseen
import numpy as np

# Create simulation
sim = Simulation(nx=128, ny=128, Lx=2*np.pi, Ly=2*np.pi, dt=0.01)

# Set initial condition (Gaussian vortex)
def ic(x, y):
    return lamb_oseen(x, y, np.pi, np.pi, Gamma=2*np.pi, r0=0.5)

sim.set_initial_condition(ic)

# Run simulation
sim.run(n_steps=1000, diag_interval=10, verbose=True)

# Access fields
q = sim.grid.q      # Potential vorticity
phi = sim.grid.phi  # Electrostatic potential
```

### Running Examples

```bash
cd examples

# Lamb-Oseen vortex test (centroid stability, peak preservation)
python run_lamb_oseen.py

# Vortex pair leapfrog (rotation dynamics)
python run_leapfrog.py

# Decaying turbulence (energy/enstrophy conservation)
python run_turbulence.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
vpfm_plasma_poc/
├── vpfm/                   # Core VPFM implementation
│   ├── grid.py            # Eulerian grid
│   ├── particles.py       # Lagrangian particles
│   ├── transfers.py       # P2G and G2P operations
│   ├── poisson.py         # FFT Poisson solver
│   ├── velocity.py        # E×B velocity computation
│   ├── integrator.py      # RK4 time integration
│   ├── diagnostics.py     # Energy, enstrophy, etc.
│   └── simulation.py      # Main simulation class
├── baseline/              # Finite difference baseline
│   └── finite_diff.py    # Upwind FD for comparison
├── tests/                 # Unit and integration tests
├── examples/              # Example scripts
└── requirements.txt
```

## Algorithm

The VPFM algorithm per timestep:

1. **P2G Transfer**: Interpolate vorticity from particles to grid
2. **Poisson Solve**: Solve (∇² - 1)φ = -q for potential
3. **Velocity Computation**: v = ẑ × ∇φ (E×B drift)
4. **Particle Advection**: RK4 integration of particle positions
5. **Jacobian Evolution**: dJ/dt = -J·∇v
6. **Reinitialization**: Reset flow map when ||J-I|| exceeds threshold

## Test Cases

| Test Case | Purpose | Success Criteria |
|-----------|---------|------------------|
| Lamb-Oseen | Vortex stability | Centroid drift < 0.1Δx, peak decay < 5% |
| Vortex Pair | Nonlinear advection | Rotation period error < 5% |
| Turbulence | Conservation | Energy/enstrophy error < 1% |

## Key Parameters

```python
# Grid
nx, ny = 128, 128    # Resolution
Lx, Ly = 2π, 2π      # Domain size

# Time stepping
dt = 0.01            # Time step (CFL ~ 0.5)
n_steps = 1000       # Number of steps

# Flow map
reinit_interval = 20 # Steps between reinitializations
reinit_threshold = 0.5  # Max ||J-I|| before reinit
```

## Limitations (POC Scope)

- 2D periodic domain only
- Single-threaded CPU implementation
- Bilinear interpolation (could use higher-order)
- Simple Euler for Jacobian evolution

## Next Steps

If POC succeeds (10x structure preservation demonstrated):

1. Higher-order interpolation kernels (quadratic/cubic B-splines)
2. Hasegawa-Wakatani model (density coupling)
3. GPU acceleration
4. Sheath boundary conditions
5. 3D extension

## References

1. Wang, S., et al. (2025). "Fluid Simulation on Vortex Particle Flow Maps." [arXiv:2505.21946](https://arxiv.org/abs/2505.21946)
2. Hasegawa, A. & Mima, K. (1978). "Pseudo-three-dimensional turbulence in magnetized nonuniform plasma." Physics of Fluids 21, 87.

## License

MIT
