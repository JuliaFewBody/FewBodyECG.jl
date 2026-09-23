# Changelog

## Unreleased

## v2.0.0

This is a breaking API release.

### Migration

| v1 API | v2 API |
|---|---|
| `solve_ECG(ops, n; scale = s)` | `solve(ops, SVM(basis = n, candidates = 1, scale = s))` |
| `solve_ECG_competitive(ops, n; n_candidates = k, scale = s)` | `solve(ops, SVM(basis = n, candidates = k, scale = s))` |
| `solve_ECG_variational(ops, n; scale = s)` | `solve(ops, GVM(basis = n, scale = s))` |
| `solve_ECG_sequential(ops, n; scale = s)` | `solve(ops, DynamicGVM(basis = n, scale = s))` |
| `SolverResults` | `Solution` |
| `sr.ground_state` | `sol.E₀` |
| `sr.basis_functions` | `sol.basis.functions` |
| `sr.energies` | `energies(sol)` |
| `ψ₀(r, sr)` | `wavefunction(sol)(r)` |
| `convergence(sr)`, `convergence_history(sr)` | `convergence(sol)`, `energies(sol)`, `plot(sol)` |
| `correlation_function(sr)` | `plot(wavefunction(sol); coord = i)` |

### Removed public names

`solve_ECG`, `solve_ECG_competitive`, `solve_ECG_variational`,
`solve_ECG_sequential`, `SolverResults`, `ψ₀`, `ψ`,
`convergence_history`, `correlation_function`, `ECG`, `generate_bij`,
`_generate_A_matrix`, and `_jacobi_transform`.

### Added public names

`solve`, `SolverMethod`, `StochasticMethod`, `GradientMethod`, `SVM`, `Refine`,
`GVM`, `DynamicGVM`, `Pipeline`, `→`, `Solution`, `ConvergenceReport`,
`StageResult`, `converged`, `energies`, `convergence`, `wavefunction`,
`Wavefunction`, `radial_profile`, `jacobi_transform`, and `default_scale`.

### Public but no longer exported

`build_hamiltonian_matrix`, `build_overlap_matrix`,
`solve_generalized_eigenproblem`, `Λ`, `jacobi_transform`, `default_scale`,
`coulomb_weights`, `up`, and `down` are declared `public` instead of exported.
They remain documented and supported; call them as `FewBodyECG.name` or bring
them into scope with `using FewBodyECG: name`.

### `Operators` term addition

`ops + term` returns a new `Operators` and leaves `ops` unchanged;
`push!(ops, term)` adds a term in place. `ops += term` works as before.

`GVM()` infers its basis size from `init`; cold starts use
`GVM(basis = n)`. `DynamicGVM(basis = n)` treats `n` as the final basis size.
Both gradient methods accept an OptimKit `GradientDescent`,
`ConjugateGradient`, or `LBFGS` object through `optimizer`; iteration limits,
gradient tolerances, and OptimKit verbosity belong to that object.

All solvers accept `verbose = true` through `solve` for concise iteration-level
progress messages. FewBodyECG no longer installs or suppresses loggers around
solver calls.

### Correctness and support

- Require Julia 1.11 or newer.
- Reject requested eigenstates that do not fit in the final basis instead of
  silently targeting a lower state.
- Surface automatic-differentiation failures and non-finite gradients instead
  of reporting false stationarity.

### Dependencies

Added `RecipesBase` for plotting recipes without requiring Plots at package
load time. `Antique` is now used only by tests and documentation.
