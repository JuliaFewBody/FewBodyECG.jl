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

A matrix-level layer (`build_overlap_matrix`, `build_hamiltonian_matrix`, `solve_generalized_eigenproblem`) is `public` but not exported; it is used for hand-built bases, spin-dependent operators, and Rank-1/Rank-2 Gaussians.

### Source Files (src/)
- **types.jl** — `GaussianBase` → `Rank0Gaussian`, `Rank1Gaussian`, `Rank2Gaussian`, `SpinGaussian`; `BasisSet`; operator types (`KineticOperator`, `CoulombOperator`, `GaussianOperator`, `OscillatorOperator`, `ManyBodyGaussianOperator`, `NumericalPotential`, `GaussianTensorOperator`, `GaussianSpinOrbitOperator`); `SpinState`
- **units.jl** — `UnitSystem` and the `:atomic`/`:nuclear` presets; labels only, nothing is converted
- **coordinates.jl** — `jacobi_transform`, `Λ` (kinetic mass matrix)
- **matrix_elements.jl** — analytic ⟨bra|op|ket⟩ formulas dispatched on Gaussian rank × operator type
- **numerical_matrix_elements.jl** — `NumericalPotential` matrix elements via a 1-D radial QuadGK integral
- **hamiltonian.jl** — physical terms (`Kinetic`, `Coulomb`, `Gaussian`, `ManyBodyGaussian`, `Oscillator`, `Custom`, `Tensor`, `SpinOrbit`), `Hamiltonian`, and their algebra; no Jacobi coordinates, so it can move to FewBodyHamiltonians unchanged
- **jacobi_frame.jl** — `_to_jacobi(H)`: transform of a `Hamiltonian` to Jacobi-frame operators, masses and units
- **linalg.jl** — overlap/Hamiltonian assembly, `solve_generalized_eigenproblem`, `default_scale`
- **eigen.jl** — `SVMEigen`, the incremental whitened eigensolver that scores SVM candidates in O(k²)
- **sampling.jl** — quasi-random Gaussian width and shift sampling (QuasiMonteCarlo)
- **methods.jl** — `SolverMethod` hierarchy, method structs, `→` composition
- **state.jl** — `BasisState`, the cached S/H/eigensolver state shared by the stochastic methods
- **solve.jl** — `solve` entry points and the `SVM`/`Refine`/`Pipeline` drivers
- **gradient.jl** — `GVM`/`DynamicGVM`: log-Cholesky encoding and Hellmann–Feynman gradients via ForwardDiff
- **solution.jl** — `Solution`, `StageResult`, `ConvergenceReport`
- **utils/** — `Wavefunction`, `convergence`, `radial_profile`, plot recipes, the PrecompileTools workload

### Key Design Patterns
- **Multiple dispatch on Gaussian rank × operator type** for matrix elements.
- **FewBodyHamiltonians dependency** supplies the `Operator`, `KineticTerm`, and `PotentialTerm` abstract types.
- **Shifts are N×3 supervectors** (row = Jacobi coordinate, column = Cartesian component); a length-N vector is placed in the z component.
- **Complex Hermitian assembly**: spin-dependent operators produce complex matrices; real-valued paths stay real.
- **Convergence reports are explicit**: stochastic saturation is reported as such, not as convergence to the exact eigenvalue.

### Parameterization for Gradient Methods

`Rank0Gaussian` is encoded as `[log-diag Cholesky of A | shift s]`, so positive-definiteness holds for any unconstrained parameter vector. Gradients use the Hellmann–Feynman theorem with ForwardDiff, at a fixed chunk size (`_GRADIENT_CHUNK = 12`) so every system reuses one compiled gradient.

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
