# Changelog

## v2.0.0

This is a breaking API release.

### Migration

| v1 API | v2 API |
|---|---|
| `solve_ECG(ops, n; scale = s)` | `solve(ops, SVM(basis = n, candidates = 1, scale = s))` |
| `solve_ECG_competitive(ops, n; n_candidates = k, scale = s)` | `solve(ops, SVM(basis = n, candidates = k, scale = s))` |
| `solve_ECG_variational(ops, n; scale = s)` | `solve(ops, Variational(basis = n, scale = s))` |
| `solve_ECG_sequential(ops, n; scale = s)` | `solve(ops, GrowVariational(basis = n, scale = s))` |
| `SolverResults` | `Solution` |
| `sr.ground_state` | `sol.E₀` |
| `sr.basis_functions` | `sol.basis.functions` |
| `sr.energies` | `energies(sol)` |
| `ψ₀(r, sr)` | `wavefunction(sol)(r)` |
| `convergence(sr)`, `convergence_history(sr)` | `energies(sol)`, `plot(sol)` |
| `correlation_function(sr)` | `plot(wavefunction(sol); coord = i)` |

### Removed public names

`solve_ECG`, `solve_ECG_competitive`, `solve_ECG_variational`,
`solve_ECG_sequential`, `SolverResults`, `ψ₀`, `ψ`, `convergence`,
`convergence_history`, `correlation_function`, `ECG`, `generate_bij`,
`_generate_A_matrix`, and `_jacobi_transform`.

### Added public names

`solve`, `SVM`, `Refine`, `Variational`, `GrowVariational`, `Pipeline`, `→`,
`AutoDiff`, `Solution`, `ConvergenceReport`, `StageResult`, `converged`,
`energies`, `wavefunction`, `Wavefunction`, `jacobi_transform`, and
`default_scale`.

### Dependencies

Added `RecipesBase` for plotting recipes without requiring Plots at package
load time.
