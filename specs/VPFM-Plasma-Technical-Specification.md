# Technical Specification: Vortex Particle Flow Maps for Plasma Edge Turbulence Simulation

**Document Version:** 1.0  
**Date:** December 2025  
**Status:** Draft  
**Classification:** Technical Specification  

---

## Executive Summary

This document specifies the design and implementation of **VPFM-Plasma**, a novel simulation framework that adapts Vortex Particle Flow Map (VPFM) methods from computational fluid dynamics to simulate turbulent plasma transport in the scrape-off layer (SOL) of magnetic confinement fusion devices. The approach exploits the mathematical equivalence between vorticity dynamics in neutral fluids and drift-wave turbulence in magnetized plasmas to achieve 10-50x improvements in simulation accuracy and computational efficiency over existing methods.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [System Architecture](#3-system-architecture)
4. [Core Algorithms](#4-core-algorithms)
5. [Boundary Conditions](#5-boundary-conditions)
6. [Data Structures](#6-data-structures)
7. [Computational Pipeline](#7-computational-pipeline)
8. [Validation Strategy](#8-validation-strategy)
9. [Performance Targets](#9-performance-targets)
10. [Integration Interfaces](#10-integration-interfaces)
11. [Risk Analysis](#11-risk-analysis)
12. [Development Roadmap](#12-development-roadmap)
13. [References](#13-references)

---

## 1. Introduction and Motivation

### 1.1 Problem Statement

The scrape-off layer (SOL) of tokamak fusion reactors exhibits turbulent transport dominated by coherent structures called "blobs" or "filaments." These structures:

- Transport significant heat and particle fluxes to plasma-facing components
- Determine divertor heat loads (critical for reactor survival)
- Evolve on microsecond timescales over meter-scale distances
- Exhibit vortex-like dynamics governed by E×B drifts

Current simulation approaches (BOUT++, GRILLIX, TOKAM3X, STORM) use Eulerian finite-difference or finite-volume methods that suffer from:

- **Numerical dissipation**: Artificial damping of fine-scale structures
- **Grid-dependent reconnection**: Unphysical blob merging at grid scale
- **Poor long-time accuracy**: Accumulated errors limit simulation duration
- **Computational cost**: Reactor-scale simulations remain prohibitive

### 1.2 Proposed Solution

VPFM-Plasma adapts the Vortex Particle Flow Map methodology to plasma turbulence by:

1. Representing the drift-vorticity field on Lagrangian particles
2. Evolving flow map quantities (Jacobian, Hessian) along particle trajectories
3. Reconstructing velocity/potential fields on an Eulerian grid
4. Exploiting the material conservation of potential vorticity

This approach preserves coherent structures over significantly longer spatiotemporal domains than existing methods.

### 1.3 Key Innovation

The central insight is that **potential vorticity in drift-wave turbulence obeys the same material conservation law as vorticity in incompressible fluids**. Specifically:

- In 2D incompressible flow: Dω/Dt = 0 (inviscid limit)
- In drift-wave turbulence: D(∇²φ - φ)/Dt ≈ 0 (adiabatic electron limit)

Both are advected by an incompressible velocity field (physical velocity or E×B drift), making VPFM directly applicable.

---

## 2. Mathematical Foundations

### 2.1 Governing Equations

#### 2.1.1 Hasegawa-Wakatani Model (Primary Target)

The Hasegawa-Wakatani equations describe resistive drift-wave turbulence:

```
∂ζ/∂t + {φ, ζ} = α(φ - n) + μ∇⁴ζ
∂n/∂t + {φ, n} = α(φ - n) - κ ∂φ/∂y + D∇²n
```

Where:
- ζ = ∇²φ is the vorticity (normalized)
- φ is the electrostatic potential
- n is the density perturbation
- {f,g} = ∂f/∂x ∂g/∂y - ∂f/∂y ∂g/∂x is the Poisson bracket
- α is the adiabaticity parameter
- κ is the background density gradient
- μ, D are dissipation coefficients

#### 2.1.2 Hasegawa-Mima Limit (Simplified Case)

In the adiabatic electron limit (α → ∞):

```
∂/∂t(∇²φ - φ) + {φ, ∇²φ - φ} + κ ∂φ/∂y = 0
```

The quantity q = ∇²φ - φ (potential vorticity) is materially conserved, making this the ideal test case for VPFM-Plasma.

#### 2.1.3 Extended Models (Future Targets)

For enhanced physics fidelity, the framework should accommodate:

- **Electromagnetic effects**: Including magnetic flutter (∂A∥/∂t terms)
- **Finite ion temperature**: Adding ion pressure gradient terms
- **3D parallel dynamics**: Coupling to parallel sound waves
- **Neutral interactions**: Charge exchange and ionization sources

### 2.2 Vorticity-Flow Map Correspondence

#### 2.2.1 Flow Map Definition

The forward flow map T^[a,b] maps particle positions from time a to time b:

```
x(t_b) = T^[a,b](x(t_a))
```

The backward flow map F^[a,b] is the inverse:

```
x(t_a) = F^[a,b](x(t_b))
```

#### 2.2.2 Vorticity Evolution on Flow Maps

In the inviscid limit, vorticity evolves as:

```
ω(x, t_b) = (∇F^[a,b])^T · ω(F^[a,b](x), t_a)
```

For 2D flows, this simplifies to:

```
ω(x, t_b) = ω(F^[a,b](x), t_a) / det(∇F^[a,b])
```

For incompressible flow, det(∇F) = 1, giving material conservation:

```
ω(x, t_b) = ω(F^[a,b](x), t_a)
```

#### 2.2.3 Plasma Vorticity Mapping

For drift-wave turbulence, define:
- **Drift vorticity**: ζ = ∇²φ
- **Potential vorticity**: q = ∇²φ - φ (Hasegawa-Mima)
- **Generalized vorticity**: Ω = ζ - n (Hasegawa-Wakatani)

The E×B velocity field:
```
v_E = ẑ × ∇φ = (-∂φ/∂y, ∂φ/∂x)
```

is automatically incompressible (∇·v_E = 0), ensuring flow map volume preservation.

### 2.3 Jacobian and Hessian Evolution

#### 2.3.1 Flow Map Jacobian

The Jacobian J = ∇F evolves along particle trajectories:

```
dJ/dt = -J · ∇v
```

where v is the E×B velocity at the particle position.

#### 2.3.2 Flow Map Hessian

The Hessian H = ∇²F (tensor of second derivatives) evolves as:

```
dH_ijk/dt = -H_ljk (∂v_i/∂x_l) - J_lj (∂²v_i/∂x_l∂x_k) - J_lk (∂²v_i/∂x_l∂x_j)
```

Accurate Hessian evolution is critical for:
- Computing vorticity gradients (∇ω)
- Enabling longer stable flow map lengths
- Reducing numerical dissipation at fine scales

### 2.4 Poisson Equation for Velocity Reconstruction

Given vorticity ζ on the grid, the potential φ is obtained from:

```
∇²φ = ζ
```

With appropriate boundary conditions (see Section 5). The E×B velocity is then:

```
v_E = ẑ × ∇φ
```

For Hasegawa-Mima, the modified Poisson equation is:

```
(∇² - 1)φ = -q
```

---

## 3. System Architecture

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VPFM-Plasma Framework                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Particle   │    │     Grid     │    │   Physics    │          │
│  │   Manager    │◄──►│    Manager   │◄──►│   Modules    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Flow Map    │    │   Poisson    │    │  Dissipation │          │
│  │  Integrator  │    │    Solver    │    │   & Sources  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             ▼                                       │
│                    ┌──────────────┐                                 │
│                    │   Output &   │                                 │
│                    │  Diagnostics │                                 │
│                    └──────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Descriptions

#### 3.2.1 Particle Manager

Responsibilities:
- Store and manage vortex particle data (position, vorticity, flow map quantities)
- Handle particle seeding, resampling, and deletion
- Implement particle-grid (P2G) and grid-particle (G2P) transfers
- Manage adaptive particle density based on local vorticity gradients

#### 3.2.2 Grid Manager

Responsibilities:
- Maintain Eulerian grid for velocity/potential reconstruction
- Support uniform Cartesian grids (Phase 1) and curvilinear grids (Phase 2+)
- Provide interpolation operators (linear, cubic, B-spline)
- Handle ghost cells and boundary exchanges for parallelization

#### 3.2.3 Flow Map Integrator

Responsibilities:
- Advance particle positions using E×B velocity
- Evolve Jacobian and Hessian tensors along trajectories
- Implement adaptive time stepping based on CFL and flow map error
- Handle flow map reinitialization when error thresholds exceeded

#### 3.2.4 Poisson Solver

Responsibilities:
- Solve ∇²φ = ζ (standard vorticity-streamfunction)
- Solve (∇² - 1)φ = -q (Hasegawa-Mima)
- Support FFT-based methods (periodic), multigrid (general), and GPU acceleration
- Enforce boundary conditions (Dirichlet, Neumann, mixed)

#### 3.2.5 Physics Modules

Responsibilities:
- Implement Hasegawa-Mima, Hasegawa-Wakatani, and extended models
- Handle coupling between vorticity and density fields
- Compute source terms (curvature drive, diamagnetic terms)
- Apply dissipation operators (hyperviscosity, collisional damping)

#### 3.2.6 Output & Diagnostics

Responsibilities:
- Compute physical diagnostics (energy, enstrophy, particle flux, heat flux)
- Output field snapshots and particle data
- Interface with visualization tools (VisIt, ParaView)
- Log performance metrics and error estimates

### 3.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Main Time Loop                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. P2G Transfer                                                     │
│    - Interpolate particle vorticity (ω_p) to grid (ω_g)            │
│    - Include gradient correction: ω_g = Σ w_ip (ω_p + ∇ω_p·Δx)     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Velocity Reconstruction                                          │
│    - Solve Poisson equation: ∇²φ = ζ                               │
│    - Compute E×B velocity: v = ẑ × ∇φ                               │
│    - Compute velocity gradients: ∇v, ∇²v                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Particle Advection                                               │
│    - Interpolate velocity to particle positions                     │
│    - Advance positions: x_p^{n+1} = x_p^n + Δt·v(x_p)              │
│    - Use RK4 or higher-order integrator                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Flow Map Evolution                                               │
│    - Evolve Jacobian: dJ/dt = -J·∇v                                 │
│    - Evolve Hessian: dH/dt = f(H, J, ∇v, ∇²v)                       │
│    - Update cumulative flow map quantities                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Vorticity Update (via Flow Map)                                  │
│    - Pull back vorticity: ω(x,t) = J^{-T}·ω(F(x),t_0)              │
│    - For 2D incompressible: ω(x,t) = ω(F(x),t_0)                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Physics Source Terms                                             │
│    - Add curvature drive: κ·∂φ/∂y                                   │
│    - Apply adiabatic coupling: α(φ-n)                               │
│    - Apply dissipation: μ∇⁴ζ, D∇²n                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. Flow Map Reinitialization (if needed)                            │
│    - Check flow map error: ε = ||J - I|| or Hessian norm            │
│    - If ε > threshold: G2P transfer, reset J=I, H=0                 │
│    - Adaptive: short maps for ∇ω, long maps for ω                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 8. Diagnostics & Output                                             │
│    - Compute energy: E = ½∫|∇φ|² dx                                 │
│    - Compute enstrophy: Z = ½∫ζ² dx                                 │
│    - Compute radial flux: Γ = <ñṽ_r>                                │
│    - Output snapshots at specified intervals                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         [Next timestep]
```

---

## 4. Core Algorithms

### 4.1 Particle-to-Grid (P2G) Transfer

#### 4.1.1 Standard P2G

Transfer vorticity from particles to grid using weighted interpolation:

```
ω_i^g = (Σ_p s_ip · ω_p) / (Σ_p s_ip)
```

where s_ip is the interpolation weight between grid node i and particle p.

#### 4.1.2 Gradient-Enhanced P2G

Include local gradient information for improved accuracy:

```
ω_i^g = (Σ_p s_ip · [ω_p + ∇ω_p · (x_i - x_p)]) / (Σ_p s_ip)
```

where ∇ω_p is the vorticity gradient stored on particle p.

#### 4.1.3 Interpolation Kernels

Supported kernels (in order of increasing accuracy/cost):
1. **Linear (tent)**: C⁰ continuous, 4 neighbors (2D)
2. **Quadratic B-spline**: C¹ continuous, 9 neighbors (2D)
3. **Cubic B-spline**: C² continuous, 16 neighbors (2D)

Default: Quadratic B-spline (balance of accuracy and cost)

### 4.2 Grid-to-Particle (G2P) Transfer

#### 4.2.1 Vorticity Transfer

During flow map reinitialization:

```
ω_p = Σ_i s_ip · ω_i^g
∇ω_p = Σ_i ∇s_ip · ω_i^g
```

#### 4.2.2 Velocity Transfer

For particle advection:

```
v_p = Σ_i s_ip · v_i^g
∇v_p = Σ_i s_ip · (∇v)_i^g   [or computed from ∇s_ip · v_i^g]
```

### 4.3 Flow Map Integration

#### 4.3.1 Jacobian Evolution

The Jacobian J = ∇F evolves according to:

```
dJ/dt = -J · ∇v
```

Discretized with RK4:
```
k1 = -J^n · ∇v(x^n, t^n)
k2 = -(J^n + Δt/2·k1) · ∇v(x^n + Δt/2·v^n, t^n + Δt/2)
k3 = -(J^n + Δt/2·k2) · ∇v(x^n + Δt/2·v^{n+1/2}, t^n + Δt/2)
k4 = -(J^n + Δt·k3) · ∇v(x^{n+1}, t^{n+1})
J^{n+1} = J^n + Δt/6·(k1 + 2k2 + 2k3 + k4)
```

#### 4.3.2 Hessian Evolution

The Hessian H_ijk = ∂²F_i/∂x_j∂x_k evolves as:

```
dH_ijk/dt = -H_ljk·(∂v_i/∂x_l) - J_lj·(∂²v_i/∂x_l∂x_k) - J_lk·(∂²v_i/∂x_l∂x_j)
```

This requires computing second derivatives of velocity (∇²v) at particle positions.

#### 4.3.3 Adaptive Flow Map Length

Two timescales are maintained:
- **n_L** (long): Flow map length for vorticity (3-12× standard)
- **n_S** (short): Flow map length for vorticity gradient (standard)

Jacobian composition at intermediate reinitializations:
```
F^[a,c] = F^[b,c] ∘ F^[a,b]
J^[a,c] = J^[b,c] · J^[a,b]
```

### 4.4 Poisson Solver

#### 4.4.1 FFT-Based Solver (Periodic Domains)

For periodic boundary conditions:
```
φ̂(k) = -ζ̂(k) / |k|²    (k ≠ 0)
φ̂(0) = 0               (gauge choice)
```

For Hasegawa-Mima:
```
φ̂(k) = q̂(k) / (|k|² + 1)
```

Implementation: FFTW (CPU) or cuFFT (GPU)

#### 4.4.2 Multigrid Solver (General Domains)

For non-periodic or complex boundary conditions:
- Geometric multigrid with V-cycles
- Red-black Gauss-Seidel smoother
- Direct solve on coarsest level

Target: O(N) complexity for N grid points

#### 4.4.3 GPU Acceleration

- cuFFT for spectral solves
- AmgX or custom multigrid for general domains
- Batched solves for ensemble simulations

### 4.5 Time Integration

#### 4.5.1 Operator Splitting

Split the evolution into:
1. **Advection** (Lagrangian, via flow maps)
2. **Poisson solve** (Eulerian, implicit)
3. **Source/dissipation** (Eulerian, explicit or implicit)

#### 4.5.2 Adaptive Time Stepping

CFL condition for advection:
```
Δt_adv < C_CFL · Δx / max|v|
```

Flow map error condition:
```
Δt_fm < C_fm · (ε_tol / ||dJ/dt||)
```

Effective timestep:
```
Δt = min(Δt_adv, Δt_fm, Δt_physics)
```

---

## 5. Boundary Conditions

### 5.1 Periodic Boundaries

Default for initial validation. Applied via:
- Wraparound in particle position updates
- FFT-natural periodicity in Poisson solve

### 5.2 Plasma-Wall Interaction (Sheath Boundary)

At material surfaces (limiter, divertor):

#### 5.2.1 Potential (Sheath Condition)

```
φ_wall = Λ·T_e    (floating potential)
```
where Λ ≈ 3 for hydrogen plasma.

Or specified: φ_wall = V_bias (biased surface)

#### 5.2.2 Density (Bohm Condition)

Outflow velocity at sheath entrance:
```
v_∥ = c_s = √(T_e/m_i)
```

Implemented as Neumann condition on density:
```
∂n/∂n̂ = -n/λ_SOL
```

#### 5.2.3 Vorticity (No-Slip / Free-Slip)

**No-slip** (viscous):
```
v_tangential = 0 → ω_wall determined by velocity gradient
```

**Free-slip** (inviscid):
```
ω_wall = 0
```

Default: Free-slip (consistent with inviscid flow map framework)

### 5.3 Solid Boundary Treatment in VPFM

#### 5.3.1 Cut-Cell Method

For complex geometries:
- Identify grid cells cut by boundaries
- Modify Poisson stencil using SPSD (Symmetric Positive Semi-Definite) formulation
- Compute boundary flux contributions

#### 5.3.2 Penalization Method (Brinkmann)

Add penalization term to enforce no-through condition:
```
∂v/∂t = ... - (v - v_solid)/τ · χ_solid
```
where χ_solid is the solid indicator function and τ << Δt.

For no-slip, add vorticity source:
```
∂ω/∂t = ... + (ω_target - ω)/τ · χ_boundary
```

### 5.4 Magnetic Geometry (Future)

For tokamak geometry:
- Radial: Sheath BC at wall, symmetry/periodicity at core
- Poloidal: Periodic (closed field lines) or sheath (open)
- Toroidal: Periodic (axisymmetric) or resolved (3D)

---

## 6. Data Structures

### 6.1 Particle Data

```
struct VortexParticle {
    // Position (2D or 3D)
    vec2/vec3 position;
    
    // Vorticity and gradient
    float vorticity;           // ω (scalar in 2D)
    vec2/vec3 vorticity_grad;  // ∇ω
    
    // Density (for Hasegawa-Wakatani)
    float density;             // n
    vec2/vec3 density_grad;    // ∇n
    
    // Flow map quantities
    mat2/mat3 jacobian;        // J = ∇F (2×2 or 3×3)
    tensor3 hessian;           // H = ∇²F (2×2×2 or 3×3×3)
    
    // Bookkeeping
    int id;                    // Unique identifier
    int cell_id;               // Containing grid cell
    float weight;              // Particle weight (for adaptive)
    int reinit_step;           // Last reinitialization step
};
```

Memory per particle (2D): ~200 bytes
Memory per particle (3D): ~500 bytes

### 6.2 Grid Data

```
struct Grid {
    // Dimensions
    int nx, ny, nz;            // Grid points per direction
    float dx, dy, dz;          // Grid spacing
    float x0, y0, z0;          // Origin
    
    // Field arrays (cell-centered)
    float* vorticity;          // ζ
    float* potential;          // φ
    float* density;            // n (for HW)
    
    // Velocity (staggered or cell-centered)
    float* vx, *vy, *vz;       // v_E components
    
    // Velocity gradients
    float* dvx_dx, *dvx_dy;    // ∇v components
    float* dvy_dx, *dvy_dy;
    
    // Second derivatives (for Hessian evolution)
    float* d2vx_dxdx, *d2vx_dxdy, *d2vx_dydy;
    float* d2vy_dxdx, *d2vy_dxdy, *d2vy_dydy;
    
    // Boundary info
    int* boundary_mask;        // 0=interior, 1=boundary, 2=solid
    float* boundary_normal_x, *boundary_normal_y;
};
```

Memory per grid point (2D): ~100 bytes (full), ~20 bytes (minimal)

### 6.3 Particle-Cell Indexing

For efficient P2G/G2P operations:

```
struct CellList {
    int* cell_start;           // Index of first particle in each cell
    int* cell_count;           // Number of particles in each cell
    int* particle_cell;        // Cell index for each particle
    int* particle_order;       // Sorted particle indices
};
```

Rebuild every N_rebuild timesteps or when particle count changes significantly.

### 6.4 Flow Map Storage

For adaptive dual-scale flow maps:

```
struct FlowMapState {
    // Long flow map (for vorticity)
    mat2* jacobian_long;       // J^[t_0, t]
    int step_long_start;       // t_0 for long map
    
    // Short flow map (for gradients)  
    mat2* jacobian_short;      // J^[t_1, t]
    tensor3* hessian_short;    // H^[t_1, t]
    int step_short_start;      // t_1 for short map
    
    // Reinitialization thresholds
    float error_threshold_long;
    float error_threshold_short;
};
```

---

## 7. Computational Pipeline

### 7.1 Initialization

```
1. Parse input parameters (grid size, physics model, BCs)
2. Initialize grid structures
3. Set initial condition:
   a. Analytic (test cases): e.g., Gaussian blob, vortex pair
   b. From file: restart or external initialization
   c. Random (turbulence): spectrum-matched noise
4. Seed particles:
   a. Uniform: one particle per cell
   b. Adaptive: higher density in high-vorticity regions
5. Initialize flow map quantities: J = I, H = 0
6. Compute initial diagnostics (E, Z, etc.)
```

### 7.2 Main Loop

```
for step = 1 to N_steps:
    
    // Adaptive timestepping
    dt = compute_timestep(CFL, flow_map_error)
    
    // P2G: particles → grid
    transfer_vorticity_P2G(particles, grid)
    if (HW_model): transfer_density_P2G(particles, grid)
    
    // Poisson solve
    solve_poisson(grid.vorticity, grid.potential, BCs)
    compute_velocity(grid.potential, grid.vx, grid.vy)
    compute_velocity_gradients(grid)
    
    // Particle advection and flow map update
    for each particle p:
        v_p = interpolate_velocity(grid, p.position)
        grad_v_p = interpolate_velocity_gradient(grid, p.position)
        hess_v_p = interpolate_velocity_hessian(grid, p.position)
        
        // RK4 integration
        advance_position(p, v_p, dt)
        advance_jacobian(p, grad_v_p, dt)
        advance_hessian(p, hess_v_p, p.jacobian, dt)
    
    // Physics updates (on grid)
    apply_curvature_drive(grid, kappa)
    if (HW_model): apply_adiabatic_coupling(grid, alpha)
    apply_dissipation(grid, mu, D)
    
    // G2P: grid → particles (for source terms)
    transfer_sources_G2P(grid, particles)
    
    // Flow map reinitialization (if needed)
    if (check_reinit_condition(particles)):
        reinitialize_flow_map(particles, grid)
    
    // Diagnostics
    if (step % diag_interval == 0):
        compute_diagnostics(grid, particles)
        output_data(grid, particles, step)
    
    // Particle management
    if (step % reseed_interval == 0):
        reseed_particles(particles, grid)
```

### 7.3 Parallelization Strategy

#### 7.3.1 Shared Memory (OpenMP/GPU)

- Grid operations: parallelize over grid points
- Particle operations: parallelize over particles
- P2G/G2P: atomic operations or cell-based reduction

#### 7.3.2 Distributed Memory (MPI)

Domain decomposition:
- 2D: Cartesian decomposition of grid
- Particles assigned to owning process based on position
- Ghost cell exchange for grid quantities
- Particle migration at domain boundaries

Communication pattern:
```
1. Exchange ghost cells (grid)
2. P2G (local + ghost contributions)
3. Reduce ghost contributions
4. Poisson solve (global or domain-decomposed)
5. Exchange ghost cells (velocity)
6. Particle advection (local)
7. Migrate particles crossing domain boundaries
```

#### 7.3.3 GPU Implementation

- Particle data: Structure of Arrays (SoA) for coalesced access
- Grid data: Texture memory for interpolation
- P2G: Atomic add or sorting-based gather
- Poisson: cuFFT (periodic) or AmgX (general)
- Advection: One thread per particle

---

## 8. Validation Strategy

### 8.1 Unit Tests

#### 8.1.1 Interpolation Accuracy

- Test P2G/G2P with known smooth functions
- Verify order of accuracy (linear: 2nd, cubic: 4th)
- Check conservation: Σ_i ω_i = Σ_p ω_p

#### 8.1.2 Flow Map Evolution

- Solid body rotation: J should remain identity
- Shear flow: J should match analytic solution
- Verify det(J) = 1 for incompressible flow

#### 8.1.3 Poisson Solver

- Compare against analytic solutions
- Verify convergence rate (FFT: machine precision, multigrid: specified tolerance)

### 8.2 Benchmark Cases

#### 8.2.1 Lamb-Oseen Vortex

Single diffusing vortex:
```
ω(r,t) = (Γ/4πνt) exp(-r²/4νt)
```
- Verify diffusion rate matches analytic
- Check vortex centroid stability

#### 8.2.2 Vortex Pair / Leapfrog

Two co-rotating or counter-rotating vortices:
- Measure trajectory accuracy vs analytic/reference
- Quantify structure preservation over long times
- Compare against standard methods (finite difference, spectral)

#### 8.2.3 Kelvin-Helmholtz Instability

Shear layer rollup:
- Verify linear growth rate
- Check nonlinear saturation
- Compare vortex sheet evolution against literature

#### 8.2.4 Hasegawa-Mima Turbulence

Decaying 2D turbulence:
- Verify energy conservation (inviscid limit)
- Check inverse cascade (energy to large scales)
- Compare spectra against published results

#### 8.2.5 Hasegawa-Wakatani Turbulence

Driven turbulence:
- Verify zonal flow generation
- Check particle flux scaling with adiabaticity
- Compare against BOUT++/GRILLIX for same parameters

### 8.3 Experimental Validation

#### 8.3.1 Blob Propagation

Compare against experimental blob tracking data:
- MAST-U: Fast camera imaging
- NSTX-U: Gas puff imaging
- ASDEX-Upgrade: Lithium beam emission

Metrics:
- Blob velocity (radial, poloidal)
- Blob size distribution
- Blob amplitude decay

#### 8.3.2 SOL Profiles

Compare time-averaged profiles:
- Density decay length (λ_n)
- Temperature decay length (λ_T)
- Fluctuation levels (δn/n, δT/T)

#### 8.3.3 Divertor Heat Flux

Compare heat flux profiles at divertor:
- Peak heat flux
- Wetted area
- Fluctuation-driven broadening

### 8.4 Convergence Studies

#### 8.4.1 Grid Convergence

- Run at Δx, Δx/2, Δx/4
- Verify expected order of accuracy
- Identify grid-independent regime

#### 8.4.2 Particle Convergence

- Run at N_p, 2N_p, 4N_p particles
- Verify diagnostics converge
- Identify minimum particle density

#### 8.4.3 Flow Map Length Convergence

- Vary reinitialization interval: n_L = 10, 20, 50, 100
- Measure accuracy vs computational cost tradeoff
- Identify optimal operating point

---

## 9. Performance Targets

### 9.1 Accuracy Metrics

| Metric | Target | Current SOTA |
|--------|--------|--------------|
| Energy conservation (inviscid) | < 0.1% drift per 1000 t_0 | ~1-5% |
| Enstrophy conservation (inviscid) | < 0.1% drift per 1000 t_0 | ~5-10% |
| Blob structure preservation | 30× longer than FD methods | 1× (baseline) |
| Flow map length | 50-100 timesteps | 5-10 (impulse methods) |

### 9.2 Computational Efficiency

| Metric | Target | Comparison |
|--------|--------|------------|
| Time per timestep (2D, 512²) | < 10 ms (GPU) | ~50 ms (BOUT++) |
| Memory per grid point | < 100 bytes | ~200 bytes (BOUT++) |
| Strong scaling efficiency | > 80% to 1024 cores | ~70% (typical) |
| GPU speedup vs CPU | > 50× | N/A (new capability) |

### 9.3 Simulation Capability

| Scenario | Target | Current Limit |
|----------|--------|---------------|
| 2D blob simulation (512²) | 10⁶ timesteps in 1 hour | 10⁵ |
| 3D SOL turbulence (256³) | 10⁵ timesteps in 24 hours | Not feasible |
| Reactor-scale (ITER params) | Demonstrate feasibility | Not achieved |

---

## 10. Integration Interfaces

### 10.1 Input File Format

YAML-based configuration:

```yaml
# VPFM-Plasma input file
simulation:
  name: "blob_propagation_test"
  dimensions: 2
  model: "hasegawa_wakatani"  # or "hasegawa_mima"

grid:
  nx: 512
  ny: 512
  Lx: 100.0  # normalized to rho_s
  Ly: 100.0
  
particles:
  initial_density: 4  # particles per cell
  adaptive: true
  min_density: 1
  max_density: 16

physics:
  adiabaticity: 1.0      # alpha
  curvature_drive: 0.05  # kappa
  viscosity: 1.0e-4      # mu (hyperviscosity)
  diffusivity: 1.0e-4    # D

flow_map:
  reinit_long: 50        # n_L
  reinit_short: 10       # n_S
  error_threshold: 0.1

time:
  dt: 0.01               # initial timestep
  adaptive: true
  cfl: 0.5
  t_end: 10000.0

boundary:
  x: "periodic"
  y: "sheath"            # or "periodic", "wall"
  sheath_potential: 3.0  # Lambda * T_e

initial_condition:
  type: "blob"           # or "turbulence", "file"
  blob_amplitude: 1.0
  blob_width: 5.0
  blob_position: [25.0, 50.0]

output:
  directory: "./output"
  field_interval: 100
  particle_interval: 1000
  diagnostic_interval: 10
  format: "hdf5"         # or "vtk", "numpy"
```

### 10.2 Output Data Format

#### 10.2.1 Field Data (HDF5)

```
/fields/
  /step_NNNNNN/
    /vorticity      [ny, nx]  float32
    /potential      [ny, nx]  float32
    /density        [ny, nx]  float32
    /vx             [ny, nx]  float32
    /vy             [ny, nx]  float32
    /attributes/
      time          float64
      step          int64
```

#### 10.2.2 Particle Data (HDF5)

```
/particles/
  /step_NNNNNN/
    /position       [n_particles, 2]  float32
    /vorticity      [n_particles]     float32
    /jacobian       [n_particles, 2, 2]  float32
    /attributes/
      n_particles   int64
      time          float64
```

#### 10.2.3 Diagnostics (CSV/HDF5)

```
step, time, energy, enstrophy, particle_flux, heat_flux, max_vorticity, ...
```

### 10.3 Coupling Interface

For integration with other codes (e.g., core transport, neutral models):

```cpp
class VPFMPlasmaInterface {
public:
    // Initialization
    void initialize(const Config& config);
    
    // Advance by one timestep
    void advance(double dt);
    
    // Field access (for coupling)
    const Field2D& get_density() const;
    const Field2D& get_potential() const;
    const Field2D& get_vorticity() const;
    
    // Source injection (from external physics)
    void add_density_source(const Field2D& source);
    void add_vorticity_source(const Field2D& source);
    
    // Flux output (for transport codes)
    double get_radial_particle_flux(double y) const;
    double get_radial_heat_flux(double y) const;
    
    // State management
    void save_state(const std::string& filename);
    void load_state(const std::string& filename);
};
```

### 10.4 Visualization Interface

Support for standard visualization tools:

- **VTK output**: For ParaView/VisIt
- **XDMF+HDF5**: For temporal datasets
- **NumPy export**: For Python analysis
- **In-situ**: ADIOS2 for live visualization

---

## 11. Risk Analysis

### 11.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Flow map instability in strong gradients | Medium | High | Adaptive reinitialization, Hessian tracking |
| P2G/G2P errors dominate | Low | High | Higher-order kernels, gradient correction |
| Poisson solver bottleneck | Medium | Medium | GPU acceleration, multigrid |
| Boundary condition artifacts | Medium | Medium | Cut-cell methods, extensive validation |
| 3D extension complexity | High | Medium | Phased approach, start with 2D |

### 11.2 Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Physics missing from model | Medium | High | Extensible design, modular physics |
| Experimental validation fails | Low | High | Multiple validation targets, parameter studies |
| Not competitive with neural surrogates | Low | Medium | Hybrid approach, VPFM-informed surrogates |

### 11.3 Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Development takes longer than expected | High | Medium | Phased milestones, minimum viable product |
| GPU porting complexity | Medium | Medium | Use portable frameworks (Kokkos, CUDA/HIP) |
| Lack of fusion community adoption | Medium | Low | Open source, extensive documentation, workshops |

---

## 12. Development Roadmap

### Phase 1: Proof of Concept (Months 1-6)

**Objective**: Demonstrate VPFM advantages for 2D plasma turbulence

**Deliverables**:
- 2D Hasegawa-Mima implementation
- Periodic boundary conditions only
- CPU implementation (OpenMP)
- Validation against spectral code
- Technical report with benchmarks

**Milestones**:
- M1.1 (Month 2): Basic infrastructure, Lamb-Oseen vortex test
- M1.2 (Month 4): Hasegawa-Mima model, leapfrog benchmark
- M1.3 (Month 6): Turbulence statistics, comparison with BOUT++

### Phase 2: Physics Extension (Months 7-12)

**Objective**: Add essential physics for SOL simulation

**Deliverables**:
- Hasegawa-Wakatani model
- Sheath boundary conditions
- Curvature drive terms
- Blob propagation validation

**Milestones**:
- M2.1 (Month 8): Hasegawa-Wakatani, zonal flow generation
- M2.2 (Month 10): Sheath BCs, blob velocity validation
- M2.3 (Month 12): Comparison with experimental blob data

### Phase 3: Performance Optimization (Months 13-18)

**Objective**: Achieve competitive performance for production use

**Deliverables**:
- GPU implementation (CUDA/HIP)
- MPI parallelization
- Adaptive particle management
- Performance benchmarks at scale

**Milestones**:
- M3.1 (Month 14): GPU port, 50× speedup demonstrated
- M3.2 (Month 16): MPI scaling to 256+ processes
- M3.3 (Month 18): Production-ready 2D code release

### Phase 4: 3D Extension (Months 19-30)

**Objective**: Enable 3D SOL turbulence simulation

**Deliverables**:
- 3D flow map formulation
- Magnetic field-aligned coordinates
- Parallel dynamics coupling
- 3D validation cases

**Milestones**:
- M4.1 (Month 22): 3D infrastructure, simple geometry tests
- M4.2 (Month 26): Field-aligned implementation
- M4.3 (Month 30): 3D blob dynamics validation

### Phase 5: Production & Dissemination (Months 31-36)

**Objective**: Enable community adoption and application

**Deliverables**:
- Open-source release
- User documentation and tutorials
- Integration with IMAS/OMAS data standards
- Partnership with experimental team

**Milestones**:
- M5.1 (Month 32): Public code release, documentation
- M5.2 (Month 34): First external user results
- M5.3 (Month 36): Publication of reactor-relevant simulation

---

## 13. References

### 13.1 VPFM Method

1. Wang, S., et al. (2025). "Fluid Simulation on Vortex Particle Flow Maps." ACM Transactions on Graphics 44(4).

2. Zhou, J., et al. (2024). "Eulerian-Lagrangian Fluid Simulation on Particle Flow Maps." ACM Transactions on Graphics 43(4).

3. Deng, Y., et al. (2023). "Fluid Simulation on Neural Flow Maps." ACM Transactions on Graphics 42(6).

### 13.2 Plasma Turbulence Models

4. Hasegawa, A. & Mima, K. (1978). "Pseudo-three-dimensional turbulence in magnetized nonuniform plasma." Physics of Fluids 21, 87.

5. Hasegawa, A. & Wakatani, M. (1983). "Plasma edge turbulence." Physical Review Letters 50, 682.

6. Scott, B.D. (2005). "Drift wave versus interchange turbulence in tokamak geometry." Physics of Plasmas 12, 062314.

### 13.3 SOL and Edge Physics

7. Krasheninnikov, S.I., et al. (2008). "Recent theoretical progress in understanding coherent structures in edge and SOL turbulence." Journal of Plasma Physics 74, 679.

8. D'Ippolito, D.A., et al. (2011). "Convective transport by intermittent blob-filaments." Physics of Plasmas 18, 060501.

### 13.4 Existing Codes

9. Dudson, B.D., et al. (2009). "BOUT++: A framework for parallel plasma fluid simulations." Computer Physics Communications 180, 1467.

10. Stegmeir, A., et al. (2018). "GRILLIX: A 3D turbulence code based on the flux-coordinate independent approach." Plasma Physics and Controlled Fusion 60, 035005.

11. Tamain, P., et al. (2016). "The TOKAM3X code for edge turbulence fluid simulations of tokamak plasmas in versatile magnetic geometries." Journal of Computational Physics 321, 606.

### 13.5 Experimental Validation Data

12. Kirk, A., et al. (2016). "L-mode filament characteristics on MAST." Plasma Physics and Controlled Fusion 58, 085008.

13. Zweben, S.J., et al. (2017). "Blob structure and motion in the edge and SOL of NSTX." Physics of Plasmas 24, 102509.

---

## Appendices

### Appendix A: Notation Summary

| Symbol | Description | Units |
|--------|-------------|-------|
| φ | Electrostatic potential | T_e/e |
| ζ | Vorticity (∇²φ) | ρ_s⁻² |
| n | Density perturbation | n_0 |
| q | Potential vorticity (∇²φ - φ) | ρ_s⁻² |
| v_E | E×B velocity | c_s |
| ρ_s | Ion sound gyroradius | m |
| c_s | Ion sound speed | m/s |
| t | Time | L_⊥/c_s |
| α | Adiabaticity parameter | - |
| κ | Curvature drive | ρ_s/R |

### Appendix B: Coordinate Systems

#### B.1 Slab Geometry (Phase 1-2)

- x: Radial direction (outward from plasma)
- y: Poloidal/binormal direction
- z: Parallel to magnetic field (if included)

#### B.2 Field-Aligned Geometry (Phase 4+)

- ψ: Radial (flux surface label)
- θ: Poloidal angle
- ζ: Toroidal angle

With metric coefficients g^{ij} from equilibrium.

### Appendix C: Normalization

All quantities normalized to drift-wave scales:

- Length: ρ_s = c_s/Ω_i (ion sound gyroradius)
- Time: L_⊥/c_s (perpendicular transit time)
- Potential: T_e/e (electron temperature)
- Density: n_0 (background density)

Typical values for tokamak edge:
- ρ_s ≈ 0.5 mm
- c_s ≈ 30 km/s
- L_⊥ ≈ 1 cm

---

**Document End**

*This specification is a living document and will be updated as the project progresses.*
