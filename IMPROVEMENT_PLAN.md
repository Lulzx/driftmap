# VPFM-Plasma Improvement Plan

## Current Status Summary

The project has completed **Phase 1** (Proof of Concept) and **partial Phase 2-3** according to the technical specification:

### Completed Features
- Hasegawa-Mima and Hasegawa-Wakatani physics models
- B-spline interpolation kernels (linear, quadratic, cubic)
- RK4 Jacobian evolution with Hessian tracking
- Adaptive flow map reinitialization
- FFT-based Poisson solver (periodic boundaries)
- Numba JIT acceleration (CPU)
- GPU acceleration (MLX for Apple Silicon, CuPy for NVIDIA)
- 3D simulation framework
- Virtual probes and blob detection diagnostics
- Basic validation tests (Lamb-Oseen, vortex pair, turbulence)

### Benchmark Results
- 100% peak preservation vs 49% for FD upwind
- 13x better energy conservation
- 12x better enstrophy conservation
- Up to 27x GPU speedup on large grids

---

## Improvement Roadmap

### Priority 1: Core Algorithm Enhancements

#### 1.1 Gradient-Enhanced P2G Transfer
**Spec Reference:** Section 4.1.2

Current P2G uses standard weighted interpolation. Adding gradient correction improves accuracy:

```
ω_i^g = (Σ_p s_ip · [ω_p + ∇ω_p · (x_i - x_p)]) / (Σ_p s_ip)
```

**Tasks:**
- [ ] Store vorticity gradient on particles (`particles.grad_q`)
- [ ] Update P2G_bspline to include gradient term
- [ ] Update G2P to compute gradient during reinitialization
- [ ] Add unit test for gradient-enhanced accuracy

**Files to modify:** `vpfm/kernels.py`, `vpfm/particles.py`, `vpfm/transfers.py`

---

#### 1.2 Dual-Scale Flow Maps (n_L / n_S)
**Spec Reference:** Section 4.3.3

The spec calls for two timescales:
- **n_L** (long): Flow map for vorticity (3-12× standard)
- **n_S** (short): Flow map for vorticity gradient (standard)

This allows longer stable flow maps while maintaining gradient accuracy.

**Tasks:**
- [ ] Add `FlowMapState.jacobian_long` for long-scale tracking
- [ ] Implement Jacobian composition: J^[a,c] = J^[b,c] · J^[a,b]
- [ ] Add separate reinitialization thresholds for long/short maps
- [ ] Update `FlowMapIntegrator.step()` to handle dual scales
- [ ] Benchmark improvement in flow map stability

**Files to modify:** `vpfm/flow_map.py`, `vpfm/simulation.py`

---

#### 1.3 Adaptive Time Stepping
**Spec Reference:** Section 4.5.2

Current implementation uses fixed dt. Add adaptive stepping based on:
- CFL condition: Δt_adv < C_CFL · Δx / max|v|
- Flow map error: Δt_fm < C_fm · (ε_tol / ||dJ/dt||)

**Tasks:**
- [ ] Add `compute_cfl_timestep()` function
- [ ] Add `compute_flow_map_timestep()` function
- [ ] Add `adaptive_dt` parameter to Simulation
- [ ] Update `advance()` and `step_hw()` to use adaptive dt
- [ ] Log dt history in diagnostics

**Files to modify:** `vpfm/simulation.py`, `vpfm/integrator.py`

---

### Priority 2: Boundary Conditions

#### 2.1 Sheath Boundary Conditions
**Spec Reference:** Section 5.2

For realistic SOL simulation, need sheath BCs at material surfaces:
- Potential: φ_wall = Λ·T_e (floating potential)
- Vorticity: Free-slip (ω_wall = 0) or no-slip
- Density: Bohm outflow condition

**Tasks:**
- [ ] Add boundary type enum: `periodic`, `sheath`, `wall`
- [ ] Implement sheath potential BC in Poisson solver
- [ ] Add particle reflection/absorption at boundaries
- [ ] Implement density outflow BC
- [ ] Add test case for blob hitting wall

**Files to modify:** `vpfm/poisson.py`, `vpfm/simulation.py`, new `vpfm/boundaries.py`

---

### Priority 3: Validation & Testing

#### 3.1 Kelvin-Helmholtz Instability Test
**Spec Reference:** Section 8.2.3

Add shear layer rollup test:
- Verify linear growth rate
- Check nonlinear saturation
- Compare against literature

**Tasks:**
- [ ] Add `kelvin_helmholtz_ic()` initial condition function
- [ ] Create `examples/run_kelvin_helmholtz.py`
- [ ] Compare growth rate with theoretical prediction
- [ ] Add to test suite

---

#### 3.2 Hasegawa-Wakatani Validation
**Spec Reference:** Section 8.2.5

Current HW tests are minimal. Need:
- Zonal flow generation verification
- Particle flux scaling with adiabaticity α
- Comparison with published results

**Tasks:**
- [ ] Add `test_hw_zonal_flow_generation()` test
- [ ] Add `test_hw_flux_scaling()` test
- [ ] Create `examples/run_hw_validation.py` with parameter scan
- [ ] Document comparison with BOUT++/GRILLIX results

---

#### 3.3 Convergence Studies
**Spec Reference:** Section 8.4

Add systematic convergence tests:
- Grid convergence (Δx, Δx/2, Δx/4)
- Particle convergence (N_p, 2N_p, 4N_p)
- Flow map length convergence

**Tasks:**
- [ ] Create `examples/convergence_study.py`
- [ ] Add convergence order verification to tests
- [ ] Document minimum requirements for accurate simulation

---

### Priority 4: Infrastructure

#### 4.1 YAML Configuration Files
**Spec Reference:** Section 10.1

Add YAML-based input files for easier simulation setup.

**Tasks:**
- [ ] Add `pyyaml` to requirements.txt
- [ ] Create `vpfm/config.py` for config parsing
- [ ] Add `Simulation.from_config()` class method
- [ ] Create example config files in `configs/`

---

#### 4.2 HDF5 Output
**Spec Reference:** Section 10.2

Add structured output format for analysis and restart.

**Tasks:**
- [ ] Add `h5py` to requirements.txt
- [ ] Create `vpfm/io.py` with save/load functions
- [ ] Add checkpoint/restart capability
- [ ] Document output format

---

#### 4.3 Full GPU Simulation Pipeline
**Spec Reference:** Section 7.3.3

Current GPU support is for individual kernels. Integrate into full simulation.

**Tasks:**
- [ ] Add GPU-aware Simulation class or mode
- [ ] Minimize CPU-GPU data transfers
- [ ] Benchmark full simulation GPU speedup
- [ ] Add MLX/CuPy backend selection in Simulation

---

### Priority 5: Future Extensions (Phase 4+)

#### 5.1 MPI Parallelization
**Spec Reference:** Section 7.3.2

For large-scale simulations, add distributed memory support.

**Tasks:**
- [ ] Design domain decomposition strategy
- [ ] Add mpi4py support
- [ ] Implement ghost cell exchange
- [ ] Implement particle migration
- [ ] Benchmark strong scaling

---

#### 5.2 Field-Aligned 3D Coordinates
**Spec Reference:** Appendix B.2

For tokamak geometry:
- ψ (radial), θ (poloidal), ζ (toroidal) coordinates
- Metric coefficients from equilibrium

---

#### 5.3 Electromagnetic Effects
**Spec Reference:** Section 2.1.3

Add magnetic flutter terms for enhanced physics fidelity.

---

## Implementation Order

Recommended implementation sequence:

1. **Gradient-Enhanced P2G** - Improves accuracy, relatively simple
2. **Adaptive Time Stepping** - Improves robustness
3. **Dual-Scale Flow Maps** - Key algorithm improvement
4. **KH Test + HW Validation** - Validates improvements
5. **Sheath BCs** - Enables realistic SOL simulations
6. **YAML Config + HDF5** - Improves usability
7. **Full GPU Pipeline** - Performance for production
8. **MPI** - Scale-out for large problems

---

## Success Metrics

From spec Section 9:

| Metric | Target | Current |
|--------|--------|---------|
| Energy conservation | < 0.1% drift/1000 t₀ | ~0.2% |
| Enstrophy conservation | < 0.1% drift/1000 t₀ | ~0.3% |
| Blob structure preservation | 30× longer than FD | ~10× |
| Flow map length | 50-100 timesteps | 50-200 |
| Time per timestep (512², GPU) | < 10 ms | ~50 ms* |

*Current GPU performance is for individual operations, not full timestep.

---

## Notes

- Focus on accuracy and physics validation before performance
- Keep backward compatibility with existing API
- Add tests for every new feature
- Document parameter choices and defaults
