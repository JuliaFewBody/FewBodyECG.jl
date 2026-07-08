# FewBodyECG.jl v2.0 — API Redesign Specification

**Date:** 2026-06-29
**Status:** Draft for review
**Method references:** Suzuki & Varga, *Stochastic Variational Approach to Quantum-Mechanical Few-Body Problems* (LNP m54, 1998), Chs. 3–4; Fedorov et al., *Explicitly Correlated Gaussians with Tensor Pre-factors* (Few-Body Syst 65:75, 2024); Fedorov, *Analytic Matrix Elements and Gradients with Shifted Correlated Gaussians* (Few-Body Syst 58:21, 2017).

## 1. Goal

Turn FewBodyECG.jl into a state-of-the-art, user-friendly package for building
correlated-Gaussian descriptions of quantum few-body systems: one entry point
for all solvers, an honest convergence statement on every result, physics-close
syntax with unicode, and a completely rewritten documentation and example
gallery — while staying faithful to the stochastic variational method and
compatible with the JuliaFewBody framework (FewBodyHamiltonians.jl).

Design inspiration: OptimKit.jl (`optimize(fg, x₀, LBFGS())` — algorithm structs
+ multiple dispatch) and ITensors.jl (small dispatched functions, ecosystem
plotting via recipes, unicode-friendly naming).

## 2. Decisions (locked with the author)

| Decision | Choice |
|---|---|
| Compatibility | **v2.0 hard break.** Old API deleted, no deprecation shims. |
| Solver interface | **`solve(ops, Method())`** — dispatch on algorithm structs. |
| Convergence | **Honest `ConvergenceReport`** on every `Solution`; pretty-printed. |
| System definition | **`Operators` builder preserved as-is** (FewBodyHamiltonians-compatible; masses/charges auto-build; `+=` appends). |
| Internals | **Hybrid strangler:** stochastic family unified on the whitened eigensolver; gradient family keeps proven OptimKit/ForwardDiff engines behind the same interface. |
| v2.0 scope | Core rewrite + **pipelines** (`→`) + **`Refine`** (Suzuki–Varga Sect. 4.2.6) + full docs/examples rewrite. |
| Out of scope (hooks only) | Symmetrization, scattering, analytic gradients, Rank1/2 stochastic sampling. |
| Observables | Energies, basis, convergence, **wavefunction with convenient plotting** (RecipesBase). Marginal densities removed from scope. |
| Style | Multiple dispatch as the extension mechanism everywhere. Unicode in names and accessors; no operator-overloading DSL for Hamiltonians. |
| Process | No git commits by the assistant; the author drives all git operations. |

## 3. Public API

### 3.1 System definition (unchanged)

```julia
mₚ  = 1836.15267343
ops = Operators([mₚ, mₚ, 1.0], [+1, +1, -1])
ops += "Kinetic"
ops += "Coulomb"                       # all pairs, coefficients qᵢqⱼ
ops += ("Gaussian", 1, 2, V₀, γ)       # optional extra terms
```

`Operators` remains FewBodyECG's builder producing
`FewBodyHamiltonians.Operator` terms (`KineticOperator`, `CoulombOperator`,
`GaussianOperator`). No changes to its behaviour or matrix elements.

### 3.2 Methods (algorithm structs)

```julia
abstract type Method end

SVM(; basis = 50, candidates = 25, scale = :auto,
      sampler = HaltonSample(), indep_tol = 1e-4)      <: Method
Refine(; sweeps = 1, candidates = 25, scale = :auto,
         sampler = HaltonSample(), indep_tol = 1e-4)   <: Method
Variational(; basis = 30, scale = :auto, maxiter = 500,
              gtol = 1e-6, gradient = AutoDiff())      <: Method
GrowVariational(; basis = 15, candidates = 10, scale = :auto,
                  maxiter_step = 100, gtol = 1e-6)     <: Method
```

* `SVM` is Suzuki–Varga stochastic selection (Sect. 4.2.5). `candidates = 1`
  implements the accept-first strategy of the old `solve_ECG` (admissibility
  via `indep_tol`, monotone energy — same strategy, not bit-identical
  results); `candidates = K` is competitive selection
  (`solve_ECG_competitive`). One method, one loop.
* `Refine` is Suzuki–Varga cyclic refinement (Sect. 4.2.6, steps r1–r4):
  revisit each basis function, draw `candidates` replacements, keep the best
  of {current, candidates}. New in v2.0.
* `Variational` is the cold-start joint LBFGS optimisation
  (old `solve_ECG_variational`).
* `GrowVariational` is per-step selection + joint LBFGS
  (old `solve_ECG_sequential`).
* `scale = :auto` resolves from the system via `default_scale(masses)`
  (masses are known to `Operators`); an explicit `Real` overrides.
* `gradient = AutoDiff()` is the only gradient backend in v2.0. The field
  exists so Fedorov-2017 analytic gradients can be added later as a new
  backend type without refactoring.

### 3.3 Pipelines

`→` (`\to<tab>`) composes methods left to right with warm starts:

```julia
→(a::Method, b::Method)   = Pipeline((a, b))
→(p::Pipeline, b::Method) = Pipeline((p.stages..., b))

sol = solve(ops, SVM(basis = 120) → Refine(sweeps = 2) → Variational())
```

Each stage receives the previous stage's result as its initial state.
Stochastic → gradient hands over the basis (parameter encoding);
gradient → stochastic rebuilds the incremental eigensolver state by
committing the basis (one O(k³) pass).

### 3.4 `solve`

```julia
solve(ops::Operators, alg::Method = SVM();
      state   = 1,          # target eigenstate (1 = ground state)
      tol     = 1e-4,       # saturation tolerance (Ha) for stochastic methods
      window  = 20,         # additions over which ΔE is measured
      init    = nothing,    # warm start from a previous Solution
      verbose = false) -> Solution

solve(ops::Operators, p::Pipeline; kw...) -> Solution
```

Problem-level options live on `solve`; algorithm-level options live on the
method structs. `solve(ops)` uses the default `SVM()`. For framework
compatibility, `solve` also accepts a raw `Vector{<:Operator}` (thin forward
to the same code path). `tol` is absolute, in Hartree; for deeply bound
systems (e.g. tdμ at ≈ −111 Ha) users set it accordingly.

### 3.5 `Solution`

```julia
struct StageResult
    method::Method
    energies::Vector{Float64}       # per-step target-state energy
    report::ConvergenceReport
end

struct Solution
    E::Vector{Float64}              # eigenvalues of the final basis (ascending)
    basis::BasisSet
    coefficients::Matrix{Float64}   # generalized eigenvectors, cᵀSc = I
    operators::Vector{Operator}
    state::Int
    stages::Vector{StageResult}     # length 1 unless a Pipeline ran
    convergence::ConvergenceReport  # final authoritative report
end
```

Accessors (all dispatched functions):

```julia
sol.E₀                    # E[state] via getproperty
converged(sol)            # Bool
energies(sol)             # concatenated per-step history (all stages)
energies(sol, i)          # history of stage i
wavefunction(sol; state = sol.state) -> Wavefunction   # callable ψ(r)
```

`Wavefunction` is a small struct (basis + coefficients) so plotting recipes
and future observables dispatch on it. `ψ(r)` evaluates in Jacobi
coordinates; its docstring states the mass-weighted coordinate convention
explicitly.

### 3.6 `ConvergenceReport`

```julia
struct ConvergenceReport
    converged::Bool
    criterion::Symbol         # :saturation | :stationarity | :max_steps | :early_stop
    ΔE::Float64               # tail energy change (Ha)
    tol::Float64
    window::Int               # saturation window (0 for gradient methods)
    gradnorm::Union{Nothing, Float64}
    cond_S::Float64           # final overlap condition number
    notes::Vector{String}     # caveats and early-stop messages
end
```

Semantics per family — each certifies only what it can:

* **Stochastic (`SVM`, `Refine`):** `converged = ΔE over the last `window`
  committed additions < tol` → criterion `:saturation`. The report always
  carries the caveat note: *"basis saturation under this sampler — not a
  certificate of the exact eigenvalue."*
* **Gradient (`Variational`, `GrowVariational`):** `converged = optimizer
  gradient tolerance met` → criterion `:stationarity`; `gradnorm` populated.
* **Early stops** (e.g. every candidate rejected — the singular-basis case)
  set `criterion = :early_stop`, `converged = false`, and a note explaining
  what happened and what to try (smaller `scale`, fewer functions).
* **Pipelines:** each `StageResult` has its own report;
  `Solution.convergence` is the final stage's report.

The variational upper-bound statement (`E₀ ≥ E_exact` never violated) is part
of the printed output for all methods.

### 3.7 Display

`show(io, ::Solution)` prints a physics-style block (structure fixed, values
illustrative):

```
FewBodyECG solution — 3 bodies, 4 operator terms
  method       SVM(120) → Refine(2) → Variational()
  E₀           −0.592568 Ha    (variational upper bound)
  basis        30 × Rank0Gaussian
  convergence  ✓ stationary  |∇E| = 8.1e-7  (gtol 1e-6)
  stages       SVM saturated ΔE=3.2e-5 → Refine −2.1 mHa → Variational −0.9 mHa
  conditioning cond(S) ≈ 3.9e14 — handled (whitened eigensolver)
  caveat       saturation ≠ exact eigenvalue; increase basis to test
```

`show(io, ::ConvergenceReport)` prints the report standalone.

### 3.8 Plotting (RecipesBase)

New dependency: RecipesBase.jl (tiny, no Plots dependency). Two recipes:

```julia
plot(sol::Solution)                 # E vs cumulative step, stage-coloured,
                                    # tol band, optional reference line via
                                    # plot(sol; reference = -0.597139)
plot(ψ::Wavefunction; coord = 1,    # r²|ψ|² along one Jacobi coordinate
     rmax = :auto, npoints = 400)
```

### 3.9 Exports (complete v2.0 list)

```
# system building (unchanged)
Operators, coulomb_weights, Operator,
KineticOperator, CoulombOperator, GaussianOperator,
GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet

# solving
solve, SVM, Refine, Variational, GrowVariational, Pipeline, →, AutoDiff

# results
Solution, ConvergenceReport, StageResult, converged, energies,
wavefunction, Wavefunction

# power-user linear algebra + coordinates
build_hamiltonian_matrix, build_overlap_matrix,
solve_generalized_eigenproblem, Λ, jacobi_transform, default_scale
```

Deleted from the public surface: `solve_ECG`, `solve_ECG_competitive`,
`solve_ECG_variational`, `solve_ECG_sequential`, `SolverResults`, `ψ₀`, `ψ`,
`convergence`, `convergence_history`, `correlation_function`, `ECG`,
`generate_bij`, `_generate_A_matrix`, `_jacobi_transform` (renamed
`jacobi_transform`, public and documented).

## 4. Internal architecture

### 4.1 Two families, one state

```julia
mutable struct BasisState
    basis::Vector{Rank0Gaussian}
    eig::SVMEigen             # whitened incremental eigensolver (R, H, W, ε)
    S::Matrix{Float64}        # cached overlap (grown column-by-column)
    H::Matrix{Float64}        # cached Hamiltonian
    E_hist::Vector{Float64}
    draw::Int                 # QMC stream position (reproducibility)
end
```

**Stochastic family** (`SVM`, `Refine`) shares one growth loop:
draw → matrix-element columns → `score_candidate` (O(k²)) → commit best →
record energy. This deletes the duplicated loops in `solve_ECG` (old
hamiltonian.jl) and `svm_solver.jl`.

`Refine` uses one new primitive: `rebuild_without(state, i)` — reconstruct
the (k−1)-function eigensolver state by re-committing cached S/H columns,
excluding function `i`. O(k³) with a small constant, no matrix-element
recomputation. Replacement candidates are then scored at O(k²) each; the best
of {current function, candidates} is committed. This is the book's r1–r4
procedure with honest costing (a sweep at k ≈ 150 is seconds).

**Gradient family** (`Variational`, `GrowVariational`) keeps the existing
OptimKit + ForwardDiff engines (Hellmann–Feynman gradients, Cholesky
log-diagonal encoding), adapted in exactly two ways:

1. accept `init` (a `Solution` or `BasisState`, encoded via the existing
   `_encode_basis`),
2. emit `Solution` + `ConvergenceReport` (from optimizer termination info
   and fg history).

The fg closure sits behind the `gradient::GradientBackend` field
(`AutoDiff` only implementation in v2.0).

### 4.2 Dispatch contract (the extension mechanism)

```julia
solve(ops::Operators, alg::Method; kw...)     # per-method dispatch
solve(ops::Operators, p::Pipeline; kw...)     # folds stages, threads init
step!(st::BasisState, alg::SVM; ...)          # one growth step
step!(st::BasisState, alg::Refine; ...)       # one replacement sweep
```

A future method (symmetrized solve, scattering, analytic-gradient backend)
is a new struct plus dispatched methods — no central code to edit.

### 4.3 File layout

| File | Content | Provenance |
|---|---|---|
| `FewBodyECG.jl` | module, exports | rewritten |
| `types.jl` | Gaussians, `BasisSet` | unchanged |
| `coordinates.jl` | Jacobi transform, `Λ`, `jacobi_transform` | rename only |
| `matrix_elements.jl` | analytic ⟨bra\|op\|ket⟩ | unchanged |
| `sampling.jl` | QMC candidate streams | unchanged |
| `operators.jl` | `Operators` builder | split from hamiltonian.jl, unchanged behaviour |
| `linalg.jl` | `build_*_matrix`, `solve_generalized_eigenproblem` | split from hamiltonian.jl |
| `eigen.jl` | whitened arrowhead eigensolver | rename of svm_eigen.jl |
| `state.jl` | `BasisState`, S/H caching, `rebuild_without` | new |
| `methods.jl` | method structs, `Pipeline`, `→`, `AutoDiff` | new |
| `solve.jl` | `solve` dispatch, stochastic growth loop | new (absorbs svm_solver.jl + greedy loop) |
| `gradient.jl` | LBFGS engines | from variational.jl, adapted |
| `solution.jl` | `Solution`, `ConvergenceReport`, `show`, accessors | new (absorbs parts of utils.jl) |
| `observables.jl` | `Wavefunction`, `wavefunction` | new (absorbs ψ evaluation from utils.jl) |
| `recipes.jl` | RecipesBase recipes | new |

Deleted: `hamiltonian.jl` (split), `svm_solver.jl` (absorbed), `utils.jl`
(absorbed), `variational.jl` (renamed/adapted).

### 4.4 Dependencies

Add RecipesBase. Keep Antique, FewBodyHamiltonians, ForwardDiff,
LinearAlgebra, OptimKit, QuasiMonteCarlo, SpecialFunctions. (Optim already
removed.)

## 5. Documentation — complete rewrite

Docs built with Documenter (`checkdocs = :exports` retained), examples
rendered as documentation pages via Literate.jl (docs-project dependency
only). Page plan:

1. **index.md** — what the package is, one compelling quickstart
   (hydrogen or H₂⁺ in ~10 lines: `Operators` → `solve` → printed
   convergence block → `plot`). README mirrors this page.
2. **systems.md** — building `Operators`; masses/charges/units (atomic
   units); Jacobi coordinates and the mass-weighted convention; `scale`
   and `default_scale`; adding Gaussian wells; manual basis construction
   with the power-user linalg layer (covers Rank1/Rank2 workflows).
3. **solvers.md** — the method family and *choosing a solver*: when
   stochastic sampling saturates (multiscale systems, the single-scale
   plateau), when gradient refinement pays, the recommended
   `SVM → Refine → Variational` workflow, cost scaling of each method.
4. **convergence.md** — exactly what each report certifies and does not:
   saturation vs. stationarity vs. the exact eigenvalue; the variational
   upper bound; conditioning and the whitened eigensolver; how to read
   `plot(sol)`.
5. **examples/** — the seven Literate examples (below).
6. **theory.md** — ECG method summary, matrix-element formulas with
   references, the incremental whitened arrowhead eigensolver, faithfulness
   notes to Suzuki–Varga and the Fedorov papers.
7. **API.md** — reference, grouped as in the export list.

## 6. Examples — uniform gallery

Seven Literate.jl examples, each ≤ ~40 lines with the same shape
(build `ops` → `solve` → display `sol` → `plot`), unicode physics names
(`mₚ`, `E₀`, `ψ`), each anchored to a known value:

| Example | System | Anchor |
|---|---|---|
| `hydrogen.jl` | H atom s/p/d via Rank0/1/2 | −1/2, −1/8, −1/18 Ha (Fedorov paper test) |
| `positronium.jl` | e⁺e⁻ | −0.25 Ha |
| `helium.jl` | He + H⁻ | −2.9037 Ha (Table 8.1), −0.5278 Ha |
| `tdmu.jl` | tdμ molecular ion | −111.36444 (Table 8.1) |
| `h2plus.jl` | H₂⁺ direct non-BO, method agreement | −0.597139 Ha reference, bound below −0.5 |
| `gaussian_well.jl` | nuclear-scale Gaussian wells | model system |
| `workflow.jl` | `SVM → Refine → Variational` pipeline showcase | stage-by-stage improvement plot |

Old examples directory is replaced wholesale.

## 7. Testing

**Ported numerical anchors (values must not change):** hydrogen s/p/d exact
energies; tdμ; H₂⁺ bound below −0.5 with variational-bound check; the 54
LAPACK cross-validation tests of the whitened eigensolver; all
matrix-element and coordinate tests (unchanged files, unchanged tests);
Operators builder tests.

**New tests:**
* `solve` dispatch for each method; default `solve(ops)`.
* `SVM(candidates = 1)` implements accept-first selection: monotone energy
  history, admissibility enforced via `indep_tol`.
* Pipeline monotonicity: each stage's final E₀ ≤ previous stage's (within
  1e-10 tolerance).
* Warm start: `solve(ops, Variational(); init = sol)` equals the manual
  encode path.
* `Refine` never raises the energy on a fixed seed; improves a deliberately
  under-converged basis.
* `rebuild_without` correctness vs. direct assembly of the (k−1) basis.
* `ConvergenceReport`: verdicts on constructed saturated / unsaturated /
  early-stop runs; criterion symbols; caveat notes present.
* `show(::Solution)` and `show(::ConvergenceReport)` smoke tests.
* RecipesBase: `RecipesBase.apply_recipe` smoke tests for both recipes.
* Aqua + docs build stay in CI.

## 8. Versioning & migration

* Version **2.0.0**; CHANGELOG with the old→new mapping:

| v1 | v2 |
|---|---|
| `solve_ECG(ops, n; ...)` | `solve(ops, SVM(basis = n, candidates = 1))` |
| `solve_ECG_competitive(ops, n; n_candidates = K)` | `solve(ops, SVM(basis = n, candidates = K))` |
| `solve_ECG_variational(ops, n)` | `solve(ops, Variational(basis = n))` |
| `solve_ECG_sequential(ops, n)` | `solve(ops, GrowVariational(basis = n))` |
| `SolverResults` | `Solution` |
| `sr.ground_state` | `sol.E₀` |
| `convergence(sr)`, `convergence_history(sr)` | `energies(sol)`, `plot(sol)` |
| `ψ₀(r, sr)` | `wavefunction(sol)(r)` |
| `correlation_function(sr)` | `plot(wavefunction(sol); coord = i)` |
| — (new) | `Refine`, pipelines `→`, `ConvergenceReport` |

* Implementation happens on a fresh `v2` branch off a clean `main`. The
  repository is currently mid-rebase (`rank2` onto `cf8a1d0`, branch
  `heliumplus`) with uncommitted docs fixes — the author resolves that
  state first; the assistant makes **no commits**.

## 9. Non-goals and designed hooks

* **Symmetrization** (v2.x): Fedorov 2024 Sect. 6 recipe —
  `P̂|(ab)A⟩ = |(Pᵀa Pᵀb)(PᵀAP)⟩`; same matrix-element formulas with
  transformed parameters. Slots in as a system-level projector at
  matrix-element assembly; no interface change required.
* **Scattering** (v2.x): new `Method` subtypes + a richer solution type;
  the dispatch contract accommodates it.
* **Analytic gradients** (Fedorov 2017): new `GradientBackend` subtype.
* **Rank1/Rank2 stochastic sampling:** stochastic candidates are
  Rank0-only in v2.0; Rank1/2 remain fully supported through manual basis
  construction and the public linalg layer (documented in systems.md).
