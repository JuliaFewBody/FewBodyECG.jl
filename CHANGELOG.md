# Changelog

## Unreleased

### `scale = :adaptive` is the default

`SVM`, `Refine`, `DynamicGVM` and cold-start `GVM` default to
`scale = :adaptive` instead of `scale = :auto`. It sets the sampling length to
0.7 × the largest root-mean-square pair distance, measured from `init` on a
warm start and otherwise from short pilot runs with `widths = :wide`; the
convergence report notes the value chosen. Cold starts that did not pass
`scale` give different results. `scale = :auto` (`1/√m` of the lightest
particle) remains available. Like `:auto`, `:adaptive` requires `Operators`
built from masses.

### Wide width sampling

`SVM`, `Refine`, `DynamicGVM` and cold-start `GVM` accept `widths = :wide`,
which samples pair widths log-uniformly between `0.01 * scale` and
`100 * scale` instead of uniformly between `0.02 * scale` and `5 * scale` (the
default, `widths = :linear`, unchanged). The result depends much less on
`scale`, at the cost of slower convergence when `scale` is well chosen.

### Units

`Operators(...; units = :atomic | :nuclear | UnitSystem(energy, length))`
records the unit convention of the masses and couplings. It converts nothing;
`Solution` carries it (`sol.units`), and printed energies and the energy-plot
axis use its label instead of always printing `Ha`. `solve` on a raw operator
vector takes `units` as a keyword, and a warm start (`init`) from a `Solution`
in different units throws an `ArgumentError`.

`ops += "Coulomb"` throws an `ArgumentError` unless the units are `:atomic`,
because its couplings `q_i q_j` assume `e²/4πε₀ = 1`.

`Solution` has new trailing fields `units` and `draws`; code that calls the
`Solution` constructor directly must pass a `UnitSystem` and a draw count.

### Fixed

The incremental eigensolver behind `SVM` and `Refine` dropped arrowhead
couplings up to `10⁻⁶ × max|ε|`, a cutoff that grows with the widest part of
the whitened spectrum. For bases spanning many length scales this discarded
couplings that lower the ground state, so reported energies were too high
(0.67 mHa for a 200-function tdμ basis) and `Refine` could raise the energy
while choosing replacements from the biased scores. A coupling is now dropped
only when its eigenvalue shift is below the root-bracket margin or it is at
rounding level.

Warm starts (`init = sol`, and every stage of a `→` pipeline after the first)
restarted the quasi-random candidate sequence near its beginning, so `Refine`,
a warm `SVM` and a warm `DynamicGVM` redrew candidates that the previous stage
had already tried. `Solution` now records the points consumed in a new trailing
field `draws`, and warm starts continue the sequence after them. Cold starts
draw the same points as before.

`examples/charmonium.jl` assigned to the imported name `Λ` and failed to run.

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
| `sr.ground_state` | `sol.E₀` or `energy(sol)` |
| `sr.basis_functions` | `sol.basis.functions` |
| `sr.energies` | `energy_history(sol)` |
| `ψ₀(r, sr)` | `wavefunction(sol)(r)` |
| `convergence(sr)`, `convergence_history(sr)` | `convergence(sol)`, `energy_history(sol)`, `plot(sol)` |
| `correlation_function(sr)` | `plot(wavefunction(sol); coord = i)` |

### Removed public names

`solve_ECG`, `solve_ECG_competitive`, `solve_ECG_variational`,
`solve_ECG_sequential`, `SolverResults`, `ψ₀`, `ψ`,
`convergence_history`, `correlation_function`, `ECG`, `generate_bij`,
`_generate_A_matrix`, and `_jacobi_transform`.

### Added public names

`solve`, `SolverMethod`, `StochasticMethod`, `GradientMethod`, `SVM`, `Refine`,
`GVM`, `DynamicGVM`, `Pipeline`, `→`, `Solution`, `ConvergenceReport`,
`StageResult`, `converged`, `energy`, `energy_history`, `convergence`, `wavefunction`,
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

### Rank-1 and rank-2 Gaussian layout

`Rank1Gaussian` and `Rank2Gaussian` store their polarizations `a`, `b` and
shift `s` as `N × 3` matrices (row = Jacobi coordinate, column = Cartesian
component), like `Rank0Gaussian`. A length-`N` vector is still accepted and
placed in the `z` component; other shapes, including `N × 1` and `N × 2`
matrices, throw an `ArgumentError`. The constructors promote mixed element
types.

The overlap of shifted rank-1 and rank-2 Gaussians with polarizations not
parallel to the shift was wrong: the shift contributed to every Cartesian
component of the polarization. For example, an `x`-polarized `Rank1Gaussian`
with a `z` shift returned the `z`-polarized overlap. Rank-2 overlaps with
matrix polarizations and nonzero shifts, previously rejected, are now
supported.

### Wavefunction evaluation

`ψ(r)` for a `Wavefunction` accepts an `N × 3` matrix of Cartesian positions
(row = Jacobi coordinate); a length-`N` vector places every coordinate on the
`z` axis, as before. Rank-1 and rank-2 terms are evaluated as `(a·r)` and
`(a·r)(b·r)` with `a·r = tr(aᵀr)`; previously every polarization component was
applied to the `z`-axis position, so, for example, an `xy`-polarized function
was nonzero on the `z` axis. `radial_profile` and `plot(ψ)` take a
`direction` keyword (default `(0, 0, 1)`) for the Cartesian direction along
which the profile is sampled.

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
