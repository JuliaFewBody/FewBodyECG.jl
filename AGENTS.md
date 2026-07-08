# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

FewBodyECG.jl is a Julia package for quantum mechanical few-body systems using the Explicitly Correlated Gaussian (ECG) variational method. It computes ground state energies by expanding wavefunctions in Gaussian basis sets and solving generalized eigenvalue problems.

## Common Commands

```bash
# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. test/test_hamiltonian.jl

# Load package in REPL
julia --project=. -e 'using FewBodyECG'

# Build documentation
julia --project=docs docs/make.jl
```

## Architecture

### Core Flow
1. Define a `ParticleSystem` from particle masses → computes Jacobi transform matrices (J, U)
2. Build `Operator` list: `KineticOperator` (transformed kinetic energy) + `CoulombOperator` (charge interactions with weight vectors selecting particle pairs)
3. Generate `BasisSet` of `GaussianBase` functions via quasirandom or random sampling
4. Compute Hamiltonian (H) and overlap (S) matrices using analytic matrix element formulas
5. Solve generalized eigenvalue problem Hc = λSc → ground state energy and eigenvectors

### Source Files (src/)
- **types.jl** — Type hierarchy: `ParticleSystem`, `GaussianBase` (abstract) → `Rank0Gaussian`, `Rank1Gaussian`, `Rank2Gaussian`, `BasisSet`, `KineticOperator`, `CoulombOperator`, `ECG`, `SolverResults`
- **coordinates.jl** — Jacobi coordinate transforms, A-matrix generation from basis parameters, weight vector construction for particle pairs
- **matrix_elements.jl** — Analytic `⟨bra|op|ket⟩` formulas dispatched on Gaussian rank × operator type combinations
- **hamiltonian.jl** — Builds overlap/Hamiltonian matrices, solves generalized eigenproblem (`eigen(Symmetric(H), Symmetric(S))` → LAPACK dsygvd with ε·I regularisation if cond(S) > 1e12), contains `solve_ECG()` stochastic greedy solver
- **sampling.jl** — Generates Gaussian basis parameters via QuasiMonteCarlo (Sobol, Halton) or pseudorandom sampling
- **utils.jl** — `SolverResults`, wavefunction evaluation (`ψ₀`, `ψ`), `convergence`, `convergence_history`, `correlation_function`
- **variational.jl** — `solve_ECG_variational` (full cold-start LBFGS optimisation) and `solve_ECG_sequential` (SVM-style sequential: sample candidates, pick best, then jointly optimise all parameters with LBFGS)

### Key Design Patterns
- **Multiple dispatch on Gaussian rank**: Matrix element formulas are specialized per `(Rank0Gaussian, Rank0Gaussian, KineticOperator)` etc., making it straightforward to add higher-rank Gaussians
- **FewBodyHamiltonians dependency**: Provides the `Operator`, `KineticTerm`, `PotentialTerm` abstract types that `KineticOperator` and `CoulombOperator` extend
- **Jacobi coordinates**: Particle coordinates are transformed to relative (Jacobi) coordinates to factor out center-of-mass motion; both forward (J) and inverse (U) matrices are stored on `ParticleSystem`
- **Scale-aware defaults**: `ParticleSystem` accepts `:atomic`, `:molecular`, or `:nuclear` scale, which sets appropriate default Gaussian width parameters

### Three Solvers

All three return `SolverResults`:
- `solve_ECG` — stochastic greedy (`:quasirandom` / `:random`): samples candidates, keeps those that lower energy
- `solve_ECG_variational` — full cold-start variational (`:variational`): optimises all basis parameters jointly from scratch via LBFGS
- `solve_ECG_sequential` — SVM-style sequential (`:sequential`): at each step samples `n_candidates`, picks best, appends to basis, then jointly optimises all parameters

### Parameterisation for Variational Solvers

`Rank0Gaussian` is encoded as `[log-diag Cholesky of A | shift s]` (layout per Gaussian: n_chol + n_dim parameters). Positive-definiteness is guaranteed for any unconstrained parameter vector. Gradients use Hellmann-Feynman theorem via ForwardDiff with chunk size `min(5×n_per, len(θ))`.

## Testing

Tests use Julia's `Test` stdlib plus `Aqua.jl` for code quality checks. Test files mirror source structure; `test_variational.jl` covers `solve_ECG_variational` and `solve_ECG_sequential`.

## Formatting

Format a file with Runic.jl:
```bash
julia --project=. -e 'using Runic; Runic.format_file("src/file.jl")'
```
