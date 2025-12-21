# VPFM-Plasma Proof of Concept
## Technical Specification

**Version:** 0.1  
**Date:** December 2025  
**Scope:** Minimal viable implementation to validate VPFM for plasma turbulence  

---

## 1. Objective

Build the simplest possible implementation that demonstrates VPFM's advantages over traditional methods for plasma edge turbulence. Success means showing **10x+ improvement in vortex structure preservation** compared to a standard finite-difference baseline on the same hardware.

**Non-goals for POC:**
- Production-ready code
- GPU acceleration
- 3D geometry
- Full physics models
- Parallel scaling

---

## 2. Scope Definition

### 2.1 What We're Building

| Component | POC Scope |
|-----------|-----------|
| Geometry | 2D periodic box |
| Physics | Hasegawa-Mima equation only |
| Particles | Fixed count, uniform seeding |
| Flow maps | Single-scale (no adaptive) |
| Poisson solver | FFT only |
| Parallelization | Single-threaded CPU |
| Language | Python + NumPy (prototyping speed) |

### 2.2 Test Cases

1. **Lamb-Oseen vortex**: Single Gaussian vortex, verify no spurious drift
2. **Vortex pair leapfrog**: Two co-rotating vortices, measure trajectory accuracy  
3. **Decaying turbulence**: Random initial condition, measure energy/enstrophy conservation

### 2.3 Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Vortex centroid drift | < 0.1 Δx over 1000 steps | Track peak location |
| Energy conservation | < 1% error over 1000 steps | ∫\|∇φ\|² dx |
| Enstrophy conservation | < 1% error over 1000 steps | ∫ζ² dx |
| Structure preservation | 10x longer than FD baseline | Visual + quantitative |

---

## 3. Mathematical Model

### 3.1 Hasegawa-Mima Equation

The simplest relevant plasma turbulence model:

```
∂q/∂t + {φ, q} = 0
```

Where:
- **q = ∇²φ - φ** is the potential vorticity
- **φ** is the electrostatic potential  
- **{f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x** is the Poisson bracket
- The **-φ** term represents polarization drift (makes this different from pure 2D Euler)

### 3.2 E×B Velocity Field

```
vx = -∂φ/∂y
vy = +∂φ/∂x
```

This is automatically incompressible: ∇·v = 0

### 3.3 Poisson Equation

Given vorticity ζ = ∇²φ, recover φ via:

```
(∇² - 1)φ = -q
```

Or in Fourier space:

```
φ̂(k) = q̂(k) / (kx² + ky² + 1)
```

### 3.4 Material Conservation

Key insight: q is materially conserved along particle trajectories:

```
Dq/Dt = ∂q/∂t + v·∇q = 0
```

Therefore: **q(x,t) = q(X(x,t), 0)** where X is the backward flow map.

---

## 4. Algorithm

### 4.1 Core VPFM Loop

```
INITIALIZE:
    Create N×N grid
    Seed one particle per cell at cell centers
    Set initial q on particles from analytic function
    Set Jacobian J = Identity for all particles
    
MAIN LOOP (for each timestep):
    
    1. P2G TRANSFER
       For each grid point (i,j):
           q_grid[i,j] = weighted_average(nearby_particle_q)
    
    2. POISSON SOLVE
       φ = FFT_solve(q_grid)
       
    3. VELOCITY COMPUTATION
       vx = -∂φ/∂y  (central difference or spectral)
       vy = +∂φ/∂x
       
    4. VELOCITY GRADIENT (for Jacobian evolution)
       ∇v = [[∂vx/∂x, ∂vx/∂y],
             [∂vy/∂x, ∂vy/∂y]]
    
    5. PARTICLE UPDATE (for each particle p)
       v_p = interpolate(vx, vy, particle_pos[p])
       ∇v_p = interpolate(∇v, particle_pos[p])
       
       # RK4 position update
       particle_pos[p] += RK4_step(v_p, dt)
       
       # Jacobian update: dJ/dt = -J·∇v
       particle_J[p] += RK4_step(-J @ ∇v_p, dt)
    
    6. FLOW MAP VORTICITY UPDATE
       # q is conserved, but we track it via flow map
       # For POC: just advect q directly on particles
       # (Full VPFM would pull back from initial condition)
       
    7. REINITIALIZATION (every n_reinit steps)
       If ||J - I|| > threshold:
           Transfer q from particles to grid (P2G)
           Transfer q from grid back to particles (G2P)
           Reset J = Identity
    
    8. DIAGNOSTICS
       Compute energy, enstrophy, max|q|
```

### 4.2 P2G Transfer (Particle to Grid)

Linear interpolation (simplest):

```python
def P2G(particles, grid, dx):
    q_grid = zeros_like(grid)
    weight_grid = zeros_like(grid)
    
    for p in particles:
        # Find containing cell
        i = floor(p.x / dx)
        j = floor(p.y / dx)
        
        # Bilinear weights
        fx = (p.x / dx) - i
        fy = (p.y / dx) - j
        
        # Distribute to 4 corners
        q_grid[i,   j]   += (1-fx)*(1-fy) * p.q
        q_grid[i+1, j]   += fx*(1-fy) * p.q
        q_grid[i,   j+1] += (1-fx)*fy * p.q
        q_grid[i+1, j+1] += fx*fy * p.q
        
        # Same for weights
        weight_grid[i,   j]   += (1-fx)*(1-fy)
        weight_grid[i+1, j]   += fx*(1-fy)
        weight_grid[i,   j+1] += (1-fx)*fy
        weight_grid[i+1, j+1] += fx*fy
    
    return q_grid / weight_grid
```

### 4.3 G2P Transfer (Grid to Particle)

```python
def G2P(grid, particles, dx):
    for p in particles:
        i = floor(p.x / dx)
        j = floor(p.y / dx)
        fx = (p.x / dx) - i
        fy = (p.y / dx) - j
        
        p.q = (1-fx)*(1-fy) * grid[i,j] + \
              fx*(1-fy) * grid[i+1,j] + \
              (1-fx)*fy * grid[i,j+1] + \
              fx*fy * grid[i+1,j+1]
```

### 4.4 Poisson Solver (FFT)

```python
def solve_poisson_HM(q_grid, Lx, Ly):
    """Solve (∇² - 1)φ = -q with periodic BCs"""
    nx, ny = q_grid.shape
    
    # Wave numbers
    kx = fftfreq(nx, Lx/nx) * 2 * pi
    ky = fftfreq(ny, Ly/ny) * 2 * pi
    KX, KY = meshgrid(kx, ky, indexing='ij')
    
    # Fourier transform
    q_hat = fft2(q_grid)
    
    # Solve in Fourier space
    K2 = KX**2 + KY**2 + 1  # +1 for Hasegawa-Mima
    K2[0, 0] = 1  # Avoid division by zero (gauge)
    phi_hat = q_hat / K2
    phi_hat[0, 0] = 0  # Zero mean
    
    return real(ifft2(phi_hat))
```

### 4.5 Velocity and Gradient Computation

```python
def compute_velocity(phi, dx, dy):
    """Compute E×B velocity from potential"""
    # Spectral derivatives (more accurate)
    phi_hat = fft2(phi)
    
    kx = fftfreq(phi.shape[0], dx) * 2 * pi
    ky = fftfreq(phi.shape[1], dy) * 2 * pi
    KX, KY = meshgrid(kx, ky, indexing='ij')
    
    dphi_dx = real(ifft2(1j * KX * phi_hat))
    dphi_dy = real(ifft2(1j * KY * phi_hat))
    
    vx = -dphi_dy  # E×B
    vy = +dphi_dx
    
    return vx, vy

def compute_velocity_gradient(vx, vy, dx, dy):
    """Compute ∇v for Jacobian evolution"""
    # Central differences (OK for POC)
    dvx_dx = (roll(vx, -1, axis=0) - roll(vx, 1, axis=0)) / (2*dx)
    dvx_dy = (roll(vx, -1, axis=1) - roll(vx, 1, axis=1)) / (2*dy)
    dvy_dx = (roll(vy, -1, axis=0) - roll(vy, 1, axis=0)) / (2*dx)
    dvy_dy = (roll(vy, -1, axis=1) - roll(vy, 1, axis=1)) / (2*dy)
    
    return dvx_dx, dvx_dy, dvy_dx, dvy_dy
```

### 4.6 Jacobian Evolution

```python
def evolve_jacobian(J, grad_v, dt):
    """Evolve Jacobian: dJ/dt = -J @ ∇v"""
    # grad_v is 2×2 matrix [[dvx/dx, dvx/dy], [dvy/dx, dvy/dy]]
    # J is 2×2 matrix
    
    dJ_dt = -J @ grad_v
    
    # Simple Euler for POC (use RK4 for production)
    return J + dt * dJ_dt
```

### 4.7 RK4 Integration

```python
def RK4_step(pos, velocity_func, dt):
    """Fourth-order Runge-Kutta position update"""
    k1 = velocity_func(pos)
    k2 = velocity_func(pos + 0.5*dt*k1)
    k3 = velocity_func(pos + 0.5*dt*k2)
    k4 = velocity_func(pos + dt*k3)
    
    return pos + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

---

## 5. Data Structures

### 5.1 Particle Structure

```python
@dataclass
class Particle:
    x: float          # Position x
    y: float          # Position y
    q: float          # Potential vorticity
    J: ndarray        # 2×2 Jacobian matrix
    
# Storage: arrays for vectorization
class ParticleSystem:
    def __init__(self, n_particles):
        self.x = zeros(n_particles)
        self.y = zeros(n_particles)
        self.q = zeros(n_particles)
        self.J = zeros((n_particles, 2, 2))
        
        # Initialize Jacobians to identity
        self.J[:, 0, 0] = 1.0
        self.J[:, 1, 1] = 1.0
```

### 5.2 Grid Structure

```python
class Grid:
    def __init__(self, nx, ny, Lx, Ly):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        
        # Field arrays
        self.q = zeros((nx, ny))      # Potential vorticity
        self.phi = zeros((nx, ny))    # Electrostatic potential
        self.vx = zeros((nx, ny))     # E×B velocity x
        self.vy = zeros((nx, ny))     # E×B velocity y
```

### 5.3 Simulation State

```python
class Simulation:
    def __init__(self, nx, ny, Lx, Ly, n_particles_per_cell=1):
        self.grid = Grid(nx, ny, Lx, Ly)
        self.particles = ParticleSystem(nx * ny * n_particles_per_cell)
        
        self.time = 0.0
        self.step = 0
        self.dt = 0.01
        
        # Flow map parameters
        self.reinit_interval = 20
        self.reinit_threshold = 0.5
        
        # Diagnostics
        self.energy_history = []
        self.enstrophy_history = []
```

---

## 6. Test Cases

### 6.1 Test Case 1: Lamb-Oseen Vortex

**Purpose**: Verify single vortex doesn't drift or decay spuriously

**Initial condition**:
```python
def lamb_oseen(x, y, x0, y0, Gamma, r0):
    """Gaussian vortex"""
    r2 = (x - x0)**2 + (y - y0)**2
    zeta = (Gamma / (pi * r0**2)) * exp(-r2 / r0**2)
    return zeta
```

**For Hasegawa-Mima**: q = ζ - φ, need to solve iteratively or use small-amplitude limit

**Success criteria**:
- Vortex centroid stays within 0.1 grid cells over 1000 timesteps
- Peak vorticity decays < 5% (physical diffusion only)
- Energy conserved to < 1%

### 6.2 Test Case 2: Vortex Pair Leapfrog

**Purpose**: Test accuracy of nonlinear advection

**Initial condition**:
```python
def vortex_pair(x, y, Lx, Ly, separation, Gamma, r0):
    """Two co-rotating vortices"""
    y_center = Ly / 2
    x1, x2 = Lx/2 - separation/2, Lx/2 + separation/2
    
    q1 = lamb_oseen(x, y, x1, y_center, +Gamma, r0)
    q2 = lamb_oseen(x, y, x2, y_center, +Gamma, r0)
    
    return q1 + q2
```

**Analytic solution**: Vortices rotate around common center at rate Ω = Γ/(2πd²)

**Success criteria**:
- Rotation period matches theory to < 5%
- Vortex separation preserved to < 5%
- Run 10+ rotation periods without degradation

### 6.3 Test Case 3: Decaying Turbulence

**Purpose**: Test energy/enstrophy conservation in turbulent regime

**Initial condition**:
```python
def random_turbulence(nx, ny, Lx, Ly, k_peak=5, amplitude=0.1):
    """Random vorticity field with specified spectrum"""
    kx = fftfreq(nx, Lx/nx) * 2 * pi
    ky = fftfreq(ny, Ly/ny) * 2 * pi
    KX, KY = meshgrid(kx, ky, indexing='ij')
    K = sqrt(KX**2 + KY**2)
    
    # Energy spectrum peaked at k_peak
    E_k = K**4 * exp(-(K / k_peak)**2)
    
    # Random phases
    phases = random.uniform(0, 2*pi, (nx, ny))
    
    # Construct vorticity in Fourier space
    zeta_hat = sqrt(E_k) * exp(1j * phases)
    zeta_hat[0, 0] = 0  # Zero mean
    
    return amplitude * real(ifft2(zeta_hat))
```

**Success criteria**:
- Energy conserved to < 1% over 1000 steps
- Enstrophy conserved to < 1% over 1000 steps  
- Inverse cascade visible (energy moves to large scales)

---

## 7. Baseline Comparison

### 7.1 Finite Difference Baseline

Implement standard semi-Lagrangian or upwind scheme for comparison:

```python
def finite_difference_step(q, phi, vx, vy, dx, dy, dt):
    """Standard upwind advection"""
    # Upwind derivatives
    dq_dx_minus = (q - roll(q, 1, axis=0)) / dx
    dq_dx_plus = (roll(q, -1, axis=0) - q) / dx
    dq_dy_minus = (q - roll(q, 1, axis=1)) / dy
    dq_dy_plus = (roll(q, -1, axis=1) - q) / dy
    
    # Upwind selection
    dq_dx = where(vx > 0, dq_dx_minus, dq_dx_plus)
    dq_dy = where(vy > 0, dq_dy_minus, dq_dy_plus)
    
    # Update
    dq_dt = -(vx * dq_dx + vy * dq_dy)
    
    return q + dt * dq_dt
```

### 7.2 Comparison Metrics

```python
def compute_diagnostics(grid):
    """Compute conservation metrics"""
    dx, dy = grid.dx, grid.dy
    
    # Energy: E = 0.5 * ∫|∇φ|² dx
    dphi_dx = (roll(grid.phi, -1, 0) - roll(grid.phi, 1, 0)) / (2*dx)
    dphi_dy = (roll(grid.phi, -1, 1) - roll(grid.phi, 1, 1)) / (2*dy)
    energy = 0.5 * sum(dphi_dx**2 + dphi_dy**2) * dx * dy
    
    # Enstrophy: Z = 0.5 * ∫ζ² dx
    zeta = grid.q + grid.phi  # ζ = q + φ for HM
    enstrophy = 0.5 * sum(zeta**2) * dx * dy
    
    # Potential enstrophy: Q = 0.5 * ∫q² dx
    pot_enstrophy = 0.5 * sum(grid.q**2) * dx * dy
    
    return {
        'energy': energy,
        'enstrophy': enstrophy,
        'pot_enstrophy': pot_enstrophy,
        'max_q': abs(grid.q).max(),
        'mean_q': grid.q.mean()
    }

def compare_structure_preservation(vpfm_q, fd_q, initial_q):
    """Compare how well structures are preserved"""
    
    # Correlation with initial condition
    vpfm_corr = corrcoef(vpfm_q.flatten(), initial_q.flatten())[0,1]
    fd_corr = corrcoef(fd_q.flatten(), initial_q.flatten())[0,1]
    
    # Peak preservation
    vpfm_peak = abs(vpfm_q).max() / abs(initial_q).max()
    fd_peak = abs(fd_q).max() / abs(initial_q).max()
    
    # Spectral content (high-k preservation)
    def high_k_energy(q):
        q_hat = fft2(q)
        kx = fftfreq(q.shape[0]) * 2 * pi
        ky = fftfreq(q.shape[1]) * 2 * pi
        KX, KY = meshgrid(kx, ky, indexing='ij')
        K = sqrt(KX**2 + KY**2)
        high_k_mask = K > K.max() / 2
        return sum(abs(q_hat[high_k_mask])**2)
    
    vpfm_highk = high_k_energy(vpfm_q) / high_k_energy(initial_q)
    fd_highk = high_k_energy(fd_q) / high_k_energy(initial_q)
    
    return {
        'vpfm_correlation': vpfm_corr,
        'fd_correlation': fd_corr,
        'vpfm_peak_ratio': vpfm_peak,
        'fd_peak_ratio': fd_peak,
        'vpfm_highk_ratio': vpfm_highk,
        'fd_highk_ratio': fd_highk
    }
```

---

## 8. Implementation Plan

### 8.1 File Structure

```
vpfm_plasma_poc/
├── vpfm/
│   ├── __init__.py
│   ├── particles.py      # Particle data structures
│   ├── grid.py           # Grid data structures
│   ├── transfers.py      # P2G and G2P operations
│   ├── poisson.py        # FFT Poisson solver
│   ├── integrator.py     # Time integration (RK4)
│   ├── flow_map.py       # Jacobian evolution
│   └── diagnostics.py    # Energy, enstrophy, etc.
├── baseline/
│   ├── __init__.py
│   └── finite_diff.py    # FD baseline for comparison
├── tests/
│   ├── test_lamb_oseen.py
│   ├── test_leapfrog.py
│   └── test_turbulence.py
├── examples/
│   ├── run_lamb_oseen.py
│   ├── run_leapfrog.py
│   └── run_turbulence.py
├── notebooks/
│   └── analysis.ipynb    # Visualization and comparison
├── requirements.txt
└── README.md
```

### 8.2 Dependencies

```
# requirements.txt
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
jupyter>=1.0
pytest>=6.0
```

### 8.3 Development Phases

#### Phase 1: Core Infrastructure (Days 1-3)

- [ ] Grid and Particle data structures
- [ ] FFT Poisson solver
- [ ] Velocity computation
- [ ] Basic interpolation (P2G, G2P)
- [ ] Unit tests for each component

#### Phase 2: VPFM Integration (Days 4-6)

- [ ] Jacobian evolution
- [ ] RK4 integrator
- [ ] Main simulation loop
- [ ] Flow map reinitialization
- [ ] Lamb-Oseen test passing

#### Phase 3: Baseline & Comparison (Days 7-8)

- [ ] Finite difference baseline
- [ ] Comparison metrics
- [ ] Leapfrog test passing
- [ ] Side-by-side visualization

#### Phase 4: Validation & Documentation (Days 9-10)

- [ ] Turbulence test
- [ ] Energy/enstrophy conservation verification
- [ ] Performance comparison (structure preservation)
- [ ] Results notebook
- [ ] README documentation

---

## 9. Validation Checklist

### 9.1 Unit Test Checklist

```
□ P2G transfer conserves total vorticity
□ G2P followed by P2G is approximately identity
□ Poisson solver matches analytic solutions
□ Velocity is divergence-free (∇·v ≈ 0)
□ Jacobian stays near identity for rigid rotation
□ RK4 has 4th-order convergence
```

### 9.2 Integration Test Checklist

```
□ Lamb-Oseen: centroid drift < 0.1 Δx @ 1000 steps
□ Lamb-Oseen: peak decay < 5% @ 1000 steps
□ Leapfrog: rotation period error < 5%
□ Leapfrog: separation preserved < 5% over 10 periods
□ Turbulence: energy error < 1% @ 1000 steps
□ Turbulence: enstrophy error < 1% @ 1000 steps
```

### 9.3 Comparison Checklist

```
□ VPFM preserves vortex structure longer than FD
□ VPFM has better energy conservation than FD
□ VPFM has better high-k spectral content than FD
□ Quantify improvement factor (target: 10x)
□ Document computational cost comparison
```

---

## 10. Expected Results

### 10.1 Qualitative

**Lamb-Oseen vortex at t=1000Δt:**

| Method | Expected Behavior |
|--------|-------------------|
| VPFM | Clean Gaussian, minimal drift |
| FD Upwind | Diffused, possibly drifted |
| FD Central | Oscillatory artifacts |

**Vortex leapfrog at t=10 periods:**

| Method | Expected Behavior |
|--------|-------------------|
| VPFM | Clean vortices, circular orbits |
| FD | Merged or distorted vortices |

### 10.2 Quantitative Targets

| Metric | VPFM Target | FD Expected |
|--------|-------------|-------------|
| Energy conservation | < 1% | 5-20% |
| Enstrophy conservation | < 1% | 10-50% |
| Peak vorticity preservation | > 95% | 50-80% |
| High-k spectral content | > 80% | 20-50% |
| Structure lifetime | 10x baseline | 1x (baseline) |

### 10.3 Computational Cost

| Operation | Estimated Cost (N = grid size) |
|-----------|-------------------------------|
| P2G | O(N) |
| Poisson (FFT) | O(N log N) |
| Velocity computation | O(N log N) |
| G2P | O(N) |
| Particle advection | O(N) |
| Jacobian evolution | O(N) |
| **Total per step** | **O(N log N)** |

Expected runtime for N=256², 1000 steps: ~10-30 seconds (single-threaded Python)

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| P2G/G2P oscillations | Medium | High | Use higher-order kernels, add small diffusion |
| Jacobian blowup | Medium | High | Frequent reinitialization, error monitoring |
| Poor performance vs FD | Low | Critical | Check implementation, simplify if needed |
| Python too slow | Medium | Low | NumPy vectorization, profile hotspots |
| HM model too simple | Low | Medium | Document as limitation, plan for HW extension |

---

## 12. Deliverables

### 12.1 Code

- Working Python implementation of VPFM for Hasegawa-Mima
- Finite difference baseline for comparison
- Three test cases with visualization
- Unit and integration tests

### 12.2 Documentation

- This specification document
- README with installation and usage
- Jupyter notebook with results analysis
- Comparison figures (VPFM vs FD)

### 12.3 Results

- Quantitative comparison table
- Conservation plots (energy, enstrophy vs time)
- Structure preservation visualization
- Performance benchmarks

---

## 13. Next Steps After POC

If POC succeeds (10x structure preservation demonstrated):

1. **Optimize Python** → NumPy vectorization, Numba JIT
2. **Add Hasegawa-Wakatani** → density equation coupling
3. **Implement sheath BCs** → non-periodic boundaries  
4. **Port to C++/CUDA** → production performance
5. **Validate against BOUT++** → same physics, same parameters
6. **Contact experimental groups** → MAST-U, NSTX-U blob data

---

## Appendix A: Quick Reference

### A.1 Hasegawa-Mima Summary

```
Equation:     ∂q/∂t + {φ, q} = 0
              where q = ∇²φ - φ

Poisson:      (∇² - 1)φ = -q

Velocity:     vx = -∂φ/∂y,  vy = +∂φ/∂x

Conservation: dq/dt = 0 along particle paths
              E = ½∫|∇φ|² conserved
              Z = ½∫q² conserved
```

### A.2 VPFM Summary

```
Particle carries: position (x,y), vorticity (q), Jacobian (J)

Evolution:
    dx/dt = v(x)           (advection)
    dJ/dt = -J·∇v          (Jacobian)
    dq/dt = 0              (material conservation)

Reinitialization (every n steps):
    q_grid = P2G(particles)
    q_particle = G2P(q_grid)
    J = Identity
```

### A.3 Key Parameters

```
Grid:       nx = ny = 128 or 256
            Lx = Ly = 2π or 20π (depending on test)
            
Time:       dt = 0.01 (CFL ~ 0.5)
            n_steps = 1000-10000
            
Flow map:   n_reinit = 20-50 steps
            threshold = 0.5 (||J-I||)
            
Test cases: Lamb-Oseen:  r0 = 1.0, Γ = 2π
            Leapfrog:    separation = 3.0, Γ = 2π
            Turbulence:  k_peak = 5, amplitude = 0.1
```

---

**End of POC Specification**
