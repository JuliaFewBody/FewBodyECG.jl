# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

FewBodyECG.jl solves quantum few-body bound-state problems with the explicitly correlated Gaussian (ECG) variational method. A wavefunction is expanded in a Gaussian basis, and energies come from the generalized eigenproblem Hc = εSc.

## Common Commands

```bash
# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Load package in REPL
julia --project=. -e 'using FewBodyECG'

# Build documentation
julia --project=docs docs/make.jl

# Run an example (examples/ has its own environment)
julia --project=examples examples/helium.jl
```

The scripts listed in `LITERATE_EXAMPLES` in `docs/make.jl` are rendered into `docs/src/generated/` with Literate.jl and run on every docs build; their `# ` comment lines are the page prose.

## Architecture

### Core Flow
1. Write a `Hamiltonian` as a sum of particle-indexed terms: `Kinetic(masses; units) + Coulomb(charges)`, `V₀ * Gaussian(i, j; exponent = γ)`, `c * Oscillator(i, j)`, `Custom(f, i, j)`; products of Gaussians give many-body Gaussians. `solve` transforms it to the Jacobi-frame operators.
2. Call `solve(ops, method)` with an `SVM`, `Refine`, `GVM`, `DynamicGVM`, or a `Pipeline` composed with `→` (e.g. `SVM(40) → Refine(2) → GVM()`).
3. Inspect the returned `Solution`: `energy`, `energy_history`, `convergence`, `wavefunction`, `radial_profile`, and the RecipesBase plot recipes.

A matrix-level layer (`build_overlap_matrix`, `build_operator_matrix`, `build_hamiltonian_matrix`, `solve_generalized_eigenproblem`) is `public` but not exported; it is used for hand-built bases, spin-dependent operators, and Rank-1/Rank-2 Gaussians. `build_operator_matrix` evaluates every entry, including non-Hermitian permutation cycles.

### Source Files (src/)

#### Core (`src/`)

- **types.jl** — `GaussianBase` → `Rank0Gaussian`, `Rank1Gaussian`, `Rank2Gaussian`, `SpinGaussian`; `BasisSet`; operator types (`KineticOperator`, `CoulombOperator`, `GaussianOperator`, `OscillatorOperator`, `ManyBodyGaussianOperator`, `NumericalPotential`, `GaussianTensorOperator`, `GaussianSpinOrbitOperator`); `SpinState`
- **FewBodyHamiltonians** — shared `UnitSystem`, particle-indexed terms and Hamiltonian algebra; imported and re-exported without local copies
- **coordinates.jl** — `jacobi_transform`, `Λ` (kinetic mass matrix)
- **matrix_elements.jl** — analytic ⟨bra|op|ket⟩ formulas dispatched on Gaussian rank × operator type, plus 1-D radial QuadGK integrals for `NumericalPotential`
- **validation.jl** — backend-specific checks for raw Jacobi-frame operators
- **jacobi_frame.jl** — `_to_jacobi(H)`: transform of a `Hamiltonian` to Jacobi-frame operators, masses and units
- **linalg.jl** — overlap/Hamiltonian assembly, `solve_generalized_eigenproblem`, `default_scale`
- **eigen.jl** — `SVMEigen`, the incremental whitened eigensolver that scores SVM candidates in O(k²)
- **sampling.jl** — quasi-random Gaussian width and shift sampling (QuasiMonteCarlo)
- **methods.jl** — `SolverMethod` hierarchy, method structs, `→` composition
- **state.jl** — `BasisState`, the cached S/H/eigensolver state shared by the stochastic methods
- **solve.jl** — `solve` entry points and the `SVM`/`Refine`/`Pipeline` drivers
- **gradient.jl** — `GVM`/`DynamicGVM`: log-Cholesky encoding and Hellmann–Feynman gradients via ForwardDiff

#### Physics (`src/physics/`)

- **polarized_elements.jl** — Rank1/Rank2 and cross-rank elements for any operator, as shift derivatives of the Rank0 element (nested ForwardDiff)
- **permutation_contact.jl** — spatial `PermutationOperator`, radial-operator × permutation products, and `ContactDensityOperator` for three-dimensional pair densities
- **symmetry.jl** — `PermutationSymmetry` and `ProjectedGaussian` for exchange-symmetry projection
- **spin_exchange.jl** — spin and isospin exchange operators
- **multipoles.jl** — dipole and quadrupole operators
- **relativistic.jl** — momentum to the fourth power, mass–velocity, and orbit–orbit operators

#### Results and utilities (`src/utils/`)

- **solution.jl** — `Solution`, `StageResult`, `ConvergenceReport`, and energy accessors
- **convergence.jl** — convergence analysis of energy histories
- **wavefunction.jl** — `Wavefunction` and `wavefunction`
- **observables.jl** — `radial_profile` and `expectation`
- **plotting.jl** — RecipesBase plot recipes
- **precompile.jl** — PrecompileTools workload

### Key Design Patterns
- **Multiple dispatch on Gaussian rank × operator type** for matrix elements.
- **FewBodyHamiltonians dependency** owns the physical construction types, units, algebra and validation; ECG only lowers shared Hamiltonians to its Jacobi-frame operators.
- **Shifts are N×3 supervectors** (row = Jacobi coordinate, column = Cartesian component); a length-N vector is placed in the z component.
- **Complex Hermitian assembly**: spin-dependent operators produce complex matrices; real-valued paths stay real.
- **Convergence reports are explicit**: stochastic saturation is reported as such, not as convergence to the exact eigenvalue.

### Parameterization for Gradient Methods

The gradient layout stores log-Cholesky widths, shifts when the layout permits them, and polarization vectors that are normalized on decode for Rank1. ForwardDiff differentiates the Hellmann–Feynman contraction one Gaussian at a time, using only its row and column of `H` and `S` and at most 12 dual partials per chunk. Hamiltonians with raw permutation terms or externally defined operator types use the full-matrix derivative.

## Precompilation

`src/utils/precompile.jl` runs a small PrecompileTools workload, which makes precompiling the package take about 20 s instead of 2 s. When editing the package without Revise, switch the workload off locally with
```julia
using PrecompileTools, FewBodyECG
PrecompileTools.Preferences.set_preferences!(FewBodyECG, "precompile_workload" => false; force = true)
```
(this writes `LocalPreferences.toml`; delete it to turn the workload back on).

## Testing

Tests use Julia's `Test` stdlib plus Aqua.jl and ExplicitImports.jl. Test files are organized by topic; reference energies come from Antique.jl and FewBodyDB.jl.

## Formatting

Format a file with Runic.jl:
```bash
julia --project=. -e 'using Runic; Runic.format_file("src/file.jl")'
```
