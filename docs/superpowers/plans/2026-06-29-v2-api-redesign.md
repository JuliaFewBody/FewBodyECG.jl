# FewBodyECG.jl v2.0 API Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four `solve_ECG_*` entry points with a dispatch-based `solve(ops, Method())` interface (SVM / Refine / Variational / GrowVariational / pipelines via `→`), honest `ConvergenceReport`s on every `Solution`, RecipesBase plotting, and a complete docs + examples rewrite — per the approved spec `docs/superpowers/specs/2026-06-29-v2-api-redesign-design.md`.

**Architecture:** Strangler: new layer is built additively (Tasks 1–8) while the old API keeps the test suite green; old tests are ported (Task 9); the old API is deleted and files are split/renamed in one cutover (Task 10); docs/examples/README follow (Tasks 11–13). Stochastic methods share one growth loop over the whitened incremental eigensolver via `BasisState`; gradient methods keep the proven OptimKit/ForwardDiff engines behind adapters.

**Tech Stack:** Julia ≥ 1.8, FewBodyHamiltonians.jl, OptimKit.jl, ForwardDiff.jl, QuasiMonteCarlo.jl, SpecialFunctions.jl, RecipesBase.jl (new), Documenter.jl + Literate.jl (docs only), Aqua.jl (tests).

## Global Constraints

- **NO GIT COMMANDS BY EXECUTORS.** The author makes all commits. Every "Checkpoint" step = stop, report, suggest a commit message; never run `git`.
- **Precondition (author, before Task 1):** finish the in-flight rebase (`rank2` onto `cf8a1d0`), land the pending `docs/make.jl` + `docs/src/API.md` fixes, cut a fresh `v2` branch from clean `main`.
- Numerical anchors must not regress: hydrogen s/p/d = −1/2, −1/8, −1/18 Ha; tdμ = −111.36444; H₂⁺ bound below −0.5 Ha; the 54 whitened-eigensolver LAPACK tests.
- `src/matrix_elements.jl`, `src/types.jl`, `src/sampling.jl` are **not modified** (except exports elsewhere).
- Format every new/edited file with Runic: `julia --project=. -e 'using Runic; Runic.format_file("FILE"; inplace=true)'`.
- Run a file's tests via `julia --project=. -e 'using FewBodyECG; include("test/FILE.jl")'`; full suite via `julia --project=. -e 'using Pkg; Pkg.test()'`.
- Public names exactly as in spec §3.9. `tol` is absolute (Hartree).

---

### Task 1: Method structs, `Pipeline`, `→`, `AutoDiff` (`src/methods.jl`)

**Files:**
- Create: `src/methods.jl`, `test/test_methods.jl`
- Modify: `Project.toml` (add RecipesBase dep now, used in Task 8), `src/FewBodyECG.jl` (include + exports), `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `HaltonSample` (in scope from `sampling.jl`'s `using QuasiMonteCarlo`).
- Produces: `abstract type Method`, `SVM`, `Refine`, `Variational`, `GrowVariational`, `Pipeline`, `→`, `AutoDiff`, `GradientBackend`, and `_resolve_scale(scale, masses)`. Field names used by later tasks: `basis`, `candidates`, `scale`, `sampler`, `indep_tol`, `sweeps`, `maxiter`, `gtol`, `gradient`, `maxiter_step`, `stages`.

- [ ] **Step 1: Add RecipesBase to Project.toml**

In `[deps]` add `RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"`; in `[compat]` add `RecipesBase = "1.3"`. Run `julia --project=. -e 'using Pkg; Pkg.resolve()'`.

- [ ] **Step 2: Write the failing tests**

Create `test/test_methods.jl`:

```julia
using Test
using FewBodyECG

@testset "Method structs and pipelines" begin
    @test SVM() isa FewBodyECG.Method
    @test SVM().basis == 50 && SVM().candidates == 25
    @test SVM(120).basis == 120                    # positional convenience
    @test SVM(120; candidates = 40).candidates == 40
    @test Refine(3).sweeps == 3
    @test Variational(30).basis == 30
    @test Variational().gradient isa AutoDiff
    @test GrowVariational().basis == 15

    p = SVM(120) → Refine(2) → Variational()
    @test p isa Pipeline
    @test length(p.stages) == 3
    @test p.stages[1] isa SVM && p.stages[3] isa Variational
    @test (SVM() → (Refine() → Variational())).stages |> length == 3

    @test sprint(show, SVM(120)) == "SVM(120)"
    @test occursin("→", sprint(show, p))

    @test FewBodyECG._resolve_scale(2.0, [1.0, 1.0]) == 2.0
    @test FewBodyECG._resolve_scale(:auto, [1.0e15, 1.0]) ≈ 1.0
    @test_throws ArgumentError FewBodyECG._resolve_scale(:auto, nothing)
end
```

- [ ] **Step 3: Run to verify failure**

Run: `julia --project=. -e 'using FewBodyECG; include("test/test_methods.jl")'`
Expected: FAIL — `UndefVarError: SVM not defined`.

- [ ] **Step 4: Implement `src/methods.jl`**

```julia
"""
    Method

Abstract supertype of all solver algorithms.  A method is a small struct of
algorithm-level options; problem-level options (`state`, `tol`, `window`,
`init`, `verbose`) live on [`solve`](@ref).  Adding a new method = defining a
new subtype plus `solve`/`step!` methods — pure multiple dispatch.
"""
abstract type Method end

"""
    GradientBackend

How gradients are obtained in the gradient-based methods.  `AutoDiff` (the
default and only v2.0 backend) uses ForwardDiff with Hellmann–Feynman
gradients.  Analytic gradients (Fedorov, Few-Body Syst 58:21, 2017) can be
added later as another subtype without interface changes.
"""
abstract type GradientBackend end

"""
    AutoDiff()

ForwardDiff-based gradient backend (Hellmann–Feynman theorem).
"""
struct AutoDiff <: GradientBackend end

"""
    SVM(basis; candidates = 25, scale = :auto, sampler = HaltonSample(), indep_tol = 1e-4)

Suzuki–Varga stochastic selection (Sect. 4.2.5).  At each of `basis` steps,
`candidates` quasi-random Gaussians are drawn and scored in O(k²) by the
incremental whitened eigensolver; the best admissible one is committed.
`candidates = 1` is the accept-first strategy.  `scale = :auto` resolves via
[`default_scale`](@ref) from the system's masses.
"""
Base.@kwdef struct SVM <: Method
    basis::Int = 50
    candidates::Int = 25
    scale::Union{Float64, Symbol} = :auto
    sampler::Any = HaltonSample()
    indep_tol::Float64 = 1.0e-4
end
SVM(basis::Int; kw...) = SVM(; basis, kw...)

"""
    Refine(sweeps; candidates = 25, scale = :auto, sampler = HaltonSample(), indep_tol = 1e-4)

Suzuki–Varga cyclic refinement (Sect. 4.2.6, steps r1–r4): revisit each basis
function in turn, draw `candidates` replacements, keep the best of
{current, candidates}.  Requires an existing basis (`init =` or a pipeline).
"""
Base.@kwdef struct Refine <: Method
    sweeps::Int = 1
    candidates::Int = 25
    scale::Union{Float64, Symbol} = :auto
    sampler::Any = HaltonSample()
    indep_tol::Float64 = 1.0e-4
end
Refine(sweeps::Int; kw...) = Refine(; sweeps, kw...)

"""
    Variational(basis; scale = :auto, maxiter = 500, gtol = 1e-6, gradient = AutoDiff())

Joint LBFGS optimisation of all Gaussian parameters (widths via log-Cholesky
encoding, plus shifts).  Cold-starts from a quasi-random basis unless
`solve(...; init = sol)` provides one.
"""
Base.@kwdef struct Variational <: Method
    basis::Int = 30
    scale::Union{Float64, Symbol} = :auto
    maxiter::Int = 500
    gtol::Float64 = 1.0e-6
    gradient::GradientBackend = AutoDiff()
end
Variational(basis::Int; kw...) = Variational(; basis, kw...)

"""
    GrowVariational(basis; candidates = 10, scale = :auto, maxiter_step = 100, gtol = 1e-6)

Per-step selection followed by joint LBFGS of the whole current basis
(SVM-style sequential growth).
"""
Base.@kwdef struct GrowVariational <: Method
    basis::Int = 15
    candidates::Int = 10
    scale::Union{Float64, Symbol} = :auto
    maxiter_step::Int = 100
    gtol::Float64 = 1.0e-6
end
GrowVariational(basis::Int; kw...) = GrowVariational(; basis, kw...)

"""
    Pipeline(stages)
    alg₁ → alg₂ → alg₃

Composition of methods run left to right; each stage warm-starts from the
previous stage's result.  Built with the `→` operator (`\\to<tab>`).
"""
struct Pipeline <: Method
    stages::Tuple{Vararg{Method}}
end

→(a::Method, b::Method) = Pipeline((a, b))
→(p::Pipeline, b::Method) = Pipeline((p.stages..., b))
→(a::Method, p::Pipeline) = Pipeline((a, p.stages...))
→(p::Pipeline, q::Pipeline) = Pipeline((p.stages..., q.stages...))

Base.show(io::IO, m::SVM) = print(io, "SVM(", m.basis, ")")
Base.show(io::IO, m::Refine) = print(io, "Refine(", m.sweeps, ")")
Base.show(io::IO, m::Variational) = print(io, "Variational(", m.basis, ")")
Base.show(io::IO, m::GrowVariational) = print(io, "GrowVariational(", m.basis, ")")
Base.show(io::IO, p::Pipeline) = join(io, p.stages, " → ")

# Forward declaration: `solve` methods live in solve.jl (Task 4).  Defining
# the empty generic function here makes the Task-1 export well-defined.
function solve end

# Resolve `scale = :auto` against the system's masses (`nothing` when the
# operators were built without masses — then an explicit scale is required).
_resolve_scale(scale::Real, _) = float(scale)
function _resolve_scale(scale::Symbol, masses)
    scale === :auto || throw(ArgumentError("unknown scale $scale; use :auto or a number"))
    masses === nothing && throw(
        ArgumentError(
            "scale = :auto requires Operators(masses[, charges]); pass an explicit scale"
        )
    )
    return default_scale(collect(Float64, masses))
end
```

- [ ] **Step 5: Wire into the module**

In `src/FewBodyECG.jl`: add `include("methods.jl")` **after** `include("sampling.jl")` (needs `HaltonSample` in scope), and add:

```julia
export solve, SVM, Refine, Variational, GrowVariational, Pipeline, →, AutoDiff
```

(`solve` is the empty generic function declared in methods.jl; its methods arrive in Task 4.) Add `include("test_methods.jl")` to `test/runtests.jl` after `test_sampling.jl`.

- [ ] **Step 6: Run tests to verify pass**

Run: `julia --project=. -e 'using FewBodyECG; include("test/test_methods.jl")'` → all pass. Format `src/methods.jl` and `test/test_methods.jl` with Runic.

- [ ] **Step 7: Checkpoint — USER COMMIT** (suggested: `feat: v2 method types, pipelines via →, gradient backend seam`)

---

### Task 2: `Solution`, `ConvergenceReport`, accessors, `show` (`src/solution.jl`)

**Files:**
- Create: `src/solution.jl`, `test/test_solution.jl`
- Modify: `src/FewBodyECG.jl` (include after methods.jl; exports), `test/runtests.jl`

**Interfaces:**
- Consumes: `Method` subtypes (Task 1), `BasisSet` (types.jl), `FewBodyHamiltonians.Operator`.
- Produces (exact, relied on by Tasks 4–8):
  - `ConvergenceReport(converged, criterion, ΔE, tol, window, gradnorm, cond_S, notes)`
  - `StageResult(method, energies, report)`
  - `Solution(E, basis, coefficients, operators, state, stages, convergence)`
  - `sol.E₀`, `converged(sol)`, `converged(report)`, `energies(sol)`, `energies(sol, i)`
  - `const SATURATION_CAVEAT` (the standard stochastic caveat string)

- [ ] **Step 1: Write the failing tests**

Create `test/test_solution.jl`:

```julia
using Test
using FewBodyECG
using FewBodyECG: StageResult, SATURATION_CAVEAT

function _dummy_solution(; converged = true)
    g = Rank0Gaussian([1.0;;], [0.0])
    rep = ConvergenceReport(
        converged, :saturation, 3.2e-5, 1.0e-4, 20, nothing, 1.0e3,
        [SATURATION_CAVEAT]
    )
    st = StageResult(SVM(2), [-0.3, -0.42], rep)
    return Solution(
        [-0.42, 1.7], BasisSet([g, g]), [1.0 0.0; 0.0 1.0],
        FewBodyECG.Operator[], 1, [st, st], rep
    )
end

@testset "Solution and ConvergenceReport" begin
    sol = _dummy_solution()
    @test sol.E₀ ≈ -0.42
    @test sol.E == [-0.42, 1.7]
    @test converged(sol)
    @test !converged(_dummy_solution(converged = false))
    @test energies(sol) == [-0.3, -0.42, -0.3, -0.42]
    @test energies(sol, 2) == [-0.3, -0.42]
    @test :E₀ in propertynames(sol)

    out = sprint(show, MIME"text/plain"(), sol)
    @test occursin("E₀", out)
    @test occursin("-0.42", out) || occursin("−0.42", out)
    @test occursin("variational upper bound", out)
    @test occursin("saturation", out)
    @test occursin("SVM(2)", out)

    rout = sprint(show, MIME"text/plain"(), sol.convergence)
    @test occursin("saturated", rout) && occursin("1.0e-4", rout)
end
```

- [ ] **Step 2: Run to verify failure**

Run: `julia --project=. -e 'using FewBodyECG; include("test/test_solution.jl")'`
Expected: FAIL — `UndefVarError: ConvergenceReport`.

- [ ] **Step 3: Implement `src/solution.jl`**

```julia
const SATURATION_CAVEAT =
    "basis saturation under this sampler — not a certificate of the exact eigenvalue"

"""
    ConvergenceReport

What a solver run can honestly certify.

- `converged::Bool`
- `criterion::Symbol` — `:saturation` (stochastic: ΔE over the last `window`
  additions below `tol`), `:stationarity` (gradient tolerance met),
  `:max_steps`, or `:early_stop`
- `ΔE::Float64` — tail energy change (Ha)
- `tol::Float64`, `window::Int` (0 for gradient methods)
- `gradnorm` — final gradient norm (`nothing` for stochastic methods)
- `cond_S::Float64` — final overlap condition number
- `notes::Vector{String}` — caveats and early-stop explanations
"""
struct ConvergenceReport
    converged::Bool
    criterion::Symbol
    ΔE::Float64
    tol::Float64
    window::Int
    gradnorm::Union{Nothing, Float64}
    cond_S::Float64
    notes::Vector{String}
end

"""
    StageResult(method, energies, report)

One pipeline stage: the method that ran, its per-step target-state energies,
and its convergence report.
"""
struct StageResult
    method::Method
    energies::Vector{Float64}
    report::ConvergenceReport
end

"""
    Solution

Result of [`solve`](@ref).  Fields: `E` (eigenvalues of the final basis,
ascending), `basis::BasisSet`, `coefficients` (generalized eigenvectors,
`cᵀSc = I`), `operators`, `state` (target eigenstate), `stages`
(length 1 unless a `Pipeline` ran), `convergence` (final report).
`sol.E₀` is the target-state energy `E[state]`.
"""
struct Solution
    E::Vector{Float64}
    basis::BasisSet
    coefficients::Matrix{Float64}
    operators::Vector{FewBodyHamiltonians.Operator}
    state::Int
    stages::Vector{StageResult}
    convergence::ConvergenceReport
end

function Base.getproperty(sol::Solution, s::Symbol)
    s === :E₀ && return getfield(sol, :E)[getfield(sol, :state)]
    return getfield(sol, s)
end
Base.propertynames(::Solution) = (fieldnames(Solution)..., :E₀)

"""
    converged(sol::Solution) -> Bool
    converged(report::ConvergenceReport) -> Bool
"""
converged(r::ConvergenceReport) = r.converged
converged(sol::Solution) = converged(getfield(sol, :convergence))

"""
    energies(sol::Solution)            -> Vector{Float64}
    energies(sol::Solution, i::Integer)

Per-step target-state energy history — concatenated across stages, or of
stage `i`.  Ready for plotting (see also `plot(sol)`).
"""
energies(sol::Solution) = reduce(vcat, (s.energies for s in getfield(sol, :stages)))
energies(sol::Solution, i::Integer) = getfield(sol, :stages)[i].energies

_fmtE(x) = string(round(x, sigdigits = 8))

function Base.show(io::IO, ::MIME"text/plain", r::ConvergenceReport)
    verdict = r.converged ? "✓" : "✗"
    desc = r.criterion === :saturation ?
        "$(verdict) saturated  ΔE = $(_fmtE(r.ΔE)) Ha over last $(r.window) additions (tol $(r.tol))" :
        r.criterion === :stationarity ?
        "$(verdict) stationary  |∇E| = $(_fmtE(something(r.gradnorm, NaN))) (gtol $(r.tol))" :
        r.criterion === :early_stop ? "✗ stopped early" : "✗ max steps reached"
    print(io, "ConvergenceReport: ", desc)
    for n in r.notes
        print(io, "\n  note: ", n)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sol::Solution)
    n = length(getfield(sol, :basis).functions)
    G = isempty(getfield(sol, :basis).functions) ? "Gaussian" :
        string(nameof(typeof(first(getfield(sol, :basis).functions))))
    r = getfield(sol, :convergence)
    println(io, "FewBodyECG solution — ", n, " × ", G, ", ",
        length(getfield(sol, :operators)), " operator terms")
    println(io, "  method       ", join((s.method for s in getfield(sol, :stages)), " → "))
    println(io, "  E₀           ", _fmtE(sol.E₀), " Ha    (variational upper bound)")
    print(io, "  convergence  ")
    show(io, MIME"text/plain"(), r)
    println(io)
    if length(getfield(sol, :stages)) > 1
        chain = join(
            ("$(s.method): E→$(_fmtE(last(s.energies)))" for s in getfield(sol, :stages)),
            "  →  "
        )
        println(io, "  stages       ", chain)
    end
    print(io, "  conditioning cond(S) ≈ ", round(r.cond_S, sigdigits = 2),
        " — handled (whitened eigensolver)")
    return nothing
end
```

- [ ] **Step 4: Wire in and run**

`src/FewBodyECG.jl`: `include("solution.jl")` after `include("methods.jl")`; add `export Solution, ConvergenceReport, StageResult, converged, energies`. Add the test include to `runtests.jl`. Run the test file → PASS. Runic-format both files.

- [ ] **Step 5: Checkpoint — USER COMMIT** (suggested: `feat: Solution + honest ConvergenceReport with physics-style display`)

---

### Task 3: `BasisState` and `rebuild_without` (`src/state.jl`)

**Files:**
- Create: `src/state.jl`, `test/test_state.jl`
- Modify: `src/FewBodyECG.jl` (include after `svm_eigen.jl`), `test/runtests.jl`

**Interfaces:**
- Consumes: `SVMEigen`, `commit_candidate!`, `score_candidate`, `coefficients` (svm_eigen.jl); `_compute_matrix_element` (matrix_elements.jl); `generate_bij`, `_generate_A_matrix`, `generate_shift` (sampling.jl).
- Produces (used by Tasks 4, 5, 6):
  - `mutable struct BasisState` with fields `basis::Vector{Rank0Gaussian}`, `eig::SVMEigen`, `S::Matrix{Float64}`, `H::Matrix{Float64}`, `E_hist::Vector{Float64}`, `draw::Int`
  - `BasisState()` — empty; `BasisState(basis, operators)` — rebuild from functions (O(k³))
  - `nfuns(st)::Int`
  - `_candidate_columns(cand, basis, operators) -> (s_col, h_col, s_diag, h_diag) | nothing`
  - `_draw_candidate!(st, scale, sampler, w_list, d) -> Rank0Gaussian` (advances `st.draw`)
  - `commit!(st, cand, cols) -> Vector{Float64}` (new ε; grows S/H caches)
  - `rebuild_without(st, i) -> BasisState`
  - `_solution_basis_state(sol::Solution, operators) -> BasisState` (warm start)

- [ ] **Step 1: Write the failing tests**

Create `test/test_state.jl`:

```julia
using Test
using LinearAlgebra
using FewBodyECG
using FewBodyECG: BasisState, nfuns, commit!, rebuild_without,
    _candidate_columns, _draw_candidate!, _solution_basis_state

# hydrogen-like fixture
ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"
terms = ops.terms
w_list = [op.w for op in terms if op isa CoulombOperator]
d = length(w_list[1])

@testset "BasisState growth and caches" begin
    st = BasisState()
    @test nfuns(st) == 0
    for _ in 1:6
        cand = _draw_candidate!(st, 1.0, FewBodyECG.HaltonSample(), w_list, d)
        cols = _candidate_columns(cand, st.basis, terms)
        cols === nothing && continue
        commit!(st, cand, cols)
    end
    k = nfuns(st)
    @test k ≥ 4
    # caches match direct assembly
    bs = BasisSet(st.basis)
    @test st.S ≈ build_overlap_matrix(bs) atol = 1e-12
    @test st.H ≈ build_hamiltonian_matrix(bs, terms) atol = 1e-12
    # eigensolver state consistent with caches
    λ = eigen(Symmetric(st.H), Symmetric(st.S)).values
    @test minimum(st.eig.ε) ≈ minimum(λ) rtol = 1e-8

    st2 = BasisState(copy(st.basis), terms)     # rebuild from functions
    @test st2.S ≈ st.S atol = 1e-12
    @test minimum(st2.eig.ε) ≈ minimum(st.eig.ε) rtol = 1e-10

    r = rebuild_without(st, 2)
    @test nfuns(r) == k - 1
    idx = setdiff(1:k, 2)
    @test r.S ≈ st.S[idx, idx] atol = 1e-12
    λr = eigen(Symmetric(st.H[idx, idx]), Symmetric(st.S[idx, idx])).values
    @test minimum(r.eig.ε) ≈ minimum(λr) rtol = 1e-8
    @test r.draw == st.draw                      # QMC stream carried over
end
```

- [ ] **Step 2: Run to verify failure** — `UndefVarError: BasisState`.

- [ ] **Step 3: Implement `src/state.jl`**

```julia
# Shared incremental state of the stochastic solver family.  Caching S and H
# (a few k² floats) makes Refine's rebuilds and warm starts cheap: no matrix
# element is ever recomputed.
mutable struct BasisState
    basis::Vector{Rank0Gaussian}
    eig::SVMEigen
    S::Matrix{Float64}
    H::Matrix{Float64}
    E_hist::Vector{Float64}
    draw::Int
end

BasisState() = BasisState(
    Rank0Gaussian[], SVMEigen(),
    Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
    Float64[], 0
)

nfuns(st::BasisState) = length(st.basis)

# Overlap/Hamiltonian columns of `cand` against `basis`; `nothing` if any
# element is non-finite.  (Moved from svm_solver.jl; deleted there in Task 10.)
function _candidate_columns(cand, basis, operators)
    k = length(basis)
    s_col = Vector{Float64}(undef, k)
    h_col = Vector{Float64}(undef, k)
    for j in 1:k
        s_col[j] = _compute_matrix_element(cand, basis[j])
        h_col[j] = sum(_compute_matrix_element(cand, basis[j], op) for op in operators)
    end
    s_diag = _compute_matrix_element(cand, cand)
    h_diag = sum(_compute_matrix_element(cand, cand, op) for op in operators)
    ok = isfinite(s_diag) && isfinite(h_diag) &&
        (k == 0 || (all(isfinite, s_col) && all(isfinite, h_col)))
    return ok ? (s_col, h_col, s_diag, h_diag) : nothing
end

# Draw the next quasi-random Rank0 candidate; advances the stream counter.
function _draw_candidate!(st::BasisState, scale::Float64, sampler, w_list, d)
    st.draw += 1
    bij = generate_bij(:quasirandom, st.draw, length(w_list), scale; qmc_sampler = sampler)
    A = _generate_A_matrix(bij, w_list)
    s = generate_shift(:quasirandom, st.draw, d, scale; qmc_sampler = sampler)
    return Rank0Gaussian(A, s)
end

# Append `cand` (whose columns are `cols`): update eigensolver + S/H caches.
function commit!(st::BasisState, cand::Rank0Gaussian, cols)
    s_col, h_col, s_diag, h_diag = cols
    ε = commit_candidate!(st.eig, s_col, h_col, s_diag, h_diag)
    ε === nothing && return nothing
    k = nfuns(st)
    S = Matrix{Float64}(undef, k + 1, k + 1)
    H = Matrix{Float64}(undef, k + 1, k + 1)
    S[1:k, 1:k] = st.S;  H[1:k, 1:k] = st.H
    S[1:k, k + 1] = s_col;  S[k + 1, 1:k] = s_col;  S[k + 1, k + 1] = s_diag
    H[1:k, k + 1] = h_col;  H[k + 1, 1:k] = h_col;  H[k + 1, k + 1] = h_diag
    st.S = S;  st.H = H
    push!(st.basis, cand)
    return ε
end

# Rebuild the eigensolver state from an explicit basis (O(k³) total).
function BasisState(basis::Vector{<:Rank0Gaussian}, operators)
    st = BasisState()
    for g in basis
        cols = _candidate_columns(g, st.basis, operators)
        cols === nothing && error("non-finite matrix element while rebuilding basis state")
        commit!(st, g, cols) === nothing &&
            error("linearly dependent basis while rebuilding state")
    end
    st.draw = length(basis)
    return st
end

# The (k−1)-function state with function `i` removed, re-committed from the
# cached S/H columns — no matrix-element recomputation.  O(k³), small constant.
function rebuild_without(st::BasisState, i::Integer)
    k = nfuns(st)
    idx = setdiff(1:k, i)
    r = BasisState()
    for (m, j) in enumerate(idx)
        prev = idx[1:(m - 1)]
        cols = (st.S[prev, j], st.H[prev, j], st.S[j, j], st.H[j, j])
        commit!(r, st.basis[j], cols) === nothing &&
            error("linear dependence while rebuilding without function $i")
    end
    r.draw = st.draw
    return r
end

# Warm start: rebuild a BasisState from a Solution's basis.
function _solution_basis_state(sol::Solution, operators)
    fns = getfield(sol, :basis).functions
    all(g -> g isa Rank0Gaussian, fns) || throw(
        ArgumentError("warm starts into stochastic methods require a Rank0Gaussian basis")
    )
    return BasisState(Rank0Gaussian[g for g in fns], operators)
end
```

- [ ] **Step 4: Wire in and run**

`src/FewBodyECG.jl`: `include("state.jl")` **after** `include("solution.jl")` (uses `Solution`) — final include order: `... svm_eigen.jl, sampling.jl, methods.jl, utils.jl, solution.jl, state.jl, svm_solver.jl, variational.jl`. Add test include. Run → PASS. Runic-format.

- [ ] **Step 5: Checkpoint — USER COMMIT** (suggested: `feat: BasisState with S/H caches and rebuild_without primitive`)

---

### Task 4: `solve` + the SVM growth loop (`src/solve.jl`)

**Files:**
- Create: `src/solve.jl`, `test/test_solve.jl`
- Modify: `src/FewBodyECG.jl` (include last, after state.jl), `test/runtests.jl`

**Interfaces:**
- Consumes: everything from Tasks 1–3; `score_candidate` (svm_eigen.jl); `coefficients(eig)`.
- Produces (relied on by Tasks 5–7):
  - `solve(ops::Operators, alg::Method = SVM(); state=1, tol=1e-4, window=20, init=nothing, verbose=false) -> Solution`
  - `solve(terms::Vector{<:FewBodyHamiltonians.Operator}, alg; kw...)`
  - internal `_SolveCtx` (fields `terms, masses, state, tol, window, verbose, w_list, d`) and `_ctx(terms, masses; kw...)`
  - `step!(st::BasisState, alg::SVM, ctx) -> Bool` (false = no admissible candidate)
  - `_stochastic_report(st, tol, window; extra_notes=String[]) -> ConvergenceReport`
  - `_solution(st::BasisState, ctx, stages) -> Solution`

- [ ] **Step 1: Write the failing tests**

Create `test/test_solve.jl`:

```julia
using Test
using LinearAlgebra
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "solve dispatch + SVM" begin
    sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0))
    @test sol isa Solution
    @test sol.E₀ ≈ -0.5 atol = 5.0e-2              # hydrogen anchor
    @test sol.E₀ > -0.5 - 1.0e-6                    # variational bound
    @test length(sol.stages) == 1
    @test sol.stages[1].method isa SVM
    @test all(diff(energies(sol)) .<= 1.0e-9)       # monotone selection
    @test sol.convergence.criterion in (:saturation, :max_steps)
    @test FewBodyECG.SATURATION_CAVEAT in sol.convergence.notes
    @test size(sol.coefficients, 2) == length(sol.E)
    # coefficients are S-orthonormal
    S = build_overlap_matrix(sol.basis)
    @test sol.coefficients' * S * sol.coefficients ≈ I atol = 1.0e-6

    # accept-first strategy
    sol1 = solve(ops, SVM(basis = 15, candidates = 1, scale = 1.0))
    @test sol1.E₀ < -0.4

    # default method + raw-terms entry + :auto scale
    @test solve(ops).E₀ < -0.4
    @test solve(ops.terms, SVM(basis = 10, candidates = 5, scale = 1.0)) isa Solution
    @test_throws ArgumentError solve(ops.terms, SVM(basis = 5))   # :auto needs masses

    # deterministic (Halton)
    @test solve(ops, SVM(basis = 15, candidates = 10, scale = 1.0)).E₀ ==
        solve(ops, SVM(basis = 15, candidates = 10, scale = 1.0)).E₀

    # excited state targeting
    sol2 = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0); state = 2)
    @test sol2.state == 2 && sol2.E₀ == sol2.E[2] && sol2.E₀ > sol2.E[1]

    # warm start grows an existing basis
    small = solve(ops, SVM(basis = 5, candidates = 10, scale = 1.0))
    bigger = solve(ops, SVM(basis = 10, candidates = 10, scale = 1.0); init = small)
    @test length(bigger.basis.functions) == 15
    @test bigger.E₀ <= small.E₀ + 1.0e-12

    # early stop: an impossible independence floor rejects every candidate,
    # leaving the warm-start basis intact with an honest :early_stop report
    stuck = solve(ops, SVM(basis = 5, candidates = 5, scale = 1.0, indep_tol = 1.0);
        init = small)
    @test stuck.convergence.criterion == :early_stop
    @test !converged(stuck)
    @test length(stuck.basis.functions) == length(small.basis.functions)
    @test any(occursin("no admissible candidate", n) for n in stuck.convergence.notes)
end
```

- [ ] **Step 2: Run to verify failure** — `MethodError: no method matching solve(...)`.

- [ ] **Step 3: Implement `src/solve.jl`**

```julia
# Problem-level context threaded through step!/report/solution assembly.
struct _SolveCtx
    terms::Vector{FewBodyHamiltonians.Operator}
    masses::Union{Nothing, Vector{Float64}}
    state::Int
    tol::Float64
    window::Int
    verbose::Bool
    w_list::Vector{Vector{Float64}}
    d::Int
end

function _ctx(terms, masses; state, tol, window, verbose)
    state ≥ 1 || throw(ArgumentError("state must be ≥ 1, got $state"))
    w_list = Vector{Float64}[op.w for op in terms
                             if op isa Union{CoulombOperator, GaussianOperator}]
    isempty(w_list) && throw(
        ArgumentError(
            "need at least one pairwise potential term (Coulomb/Gaussian) " *
            "to define the candidate geometry"
        )
    )
    return _SolveCtx(
        collect(FewBodyHamiltonians.Operator, terms), masses,
        state, float(tol), window, verbose, w_list, length(w_list[1])
    )
end

"""
    solve(ops, alg::Method = SVM();
          state = 1, tol = 1e-4, window = 20, init = nothing, verbose = false)

Solve the few-body eigenproblem defined by `ops` (an [`Operators`](@ref)
builder or a raw `Vector{<:Operator}`) with algorithm `alg` — one of
[`SVM`](@ref), [`Refine`](@ref), [`Variational`](@ref),
[`GrowVariational`](@ref), or a [`Pipeline`](@ref) composed with `→`.

Problem-level keywords: `state` targets the `state`-th eigenvalue, `tol`
(absolute, Hartree) and `window` define the stochastic saturation criterion,
`init` warm-starts from a previous [`Solution`](@ref).

Returns a [`Solution`](@ref) carrying energies, the basis, S-orthonormal
coefficients, and an honest [`ConvergenceReport`](@ref).
"""
solve(ops::Operators, alg::Method = SVM(); kw...) = _solve(ops.terms, ops.masses, alg; kw...)
solve(terms::Vector{<:FewBodyHamiltonians.Operator}, alg::Method = SVM(); kw...) =
    _solve(terms, nothing, alg; kw...)

# One SVM growth step: draw → score all candidates → commit the best.
# Returns false when no admissible candidate was found.
function step!(st::BasisState, alg::SVM, ctx::_SolveCtx)
    scale = _resolve_scale(alg.scale, ctx.masses)
    bestE = Inf
    best = nothing
    bestcols = nothing
    for _ in 1:alg.candidates
        cand = _draw_candidate!(st, scale, alg.sampler, ctx.w_list, ctx.d)
        cols = _candidate_columns(cand, st.basis, ctx.terms)
        cols === nothing && continue
        E = score_candidate(
            st.eig, cols...;
            state = ctx.state, min_resid_ratio = alg.indep_tol
        )
        E === nothing && continue
        if E < bestE
            bestE, best, bestcols = E, cand, cols
        end
    end
    best === nothing && return false
    commit!(st, best, bestcols) === nothing && return false
    push!(st.E_hist, st.eig.ε[min(ctx.state, length(st.eig.ε))])
    ctx.verbose && @info "step $(nfuns(st))" E = last(st.E_hist)
    return true
end

function _stochastic_report(st::BasisState, tol, window; extra_notes = String[])
    notes = vcat([SATURATION_CAVEAT], extra_notes)
    hist = st.E_hist
    condS = nfuns(st) == 0 ? NaN : cond(Symmetric(st.S))
    if length(hist) > window
        ΔE = hist[end - window] - hist[end]           # ≥ 0 by monotone selection
        sat = 0 ≤ ΔE < tol
        return ConvergenceReport(
            sat, sat ? :saturation : :max_steps, ΔE, tol, window,
            nothing, condS, notes
        )
    end
    push!(notes, "energy history shorter than window ($window); cannot assess saturation")
    return ConvergenceReport(false, :max_steps, NaN, tol, window, nothing, condS, notes)
end

function _solution(st::BasisState, ctx::_SolveCtx, stages::Vector{StageResult})
    nfuns(st) > 0 || error("solver produced no basis functions")
    return Solution(
        copy(st.eig.ε), BasisSet(copy(st.basis)), coefficients(st.eig),
        ctx.terms, min(ctx.state, length(st.eig.ε)),
        stages, last(stages).report
    )
end

function _solve(terms, masses, alg::SVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false)
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    st = init === nothing ? BasisState() : _solution_basis_state(init, ctx.terms)
    n₀ = length(st.E_hist)
    notes = String[]
    for _ in 1:alg.basis
        if !step!(st, alg, ctx)
            push!(notes,
                "stopped at $(nfuns(st)) functions: no admissible candidate " *
                "among $(alg.candidates) draws (try a different scale)")
            break
        end
    end
    stage_hist = st.E_hist[(n₀ + 1):end]
    rep = if isempty(notes)
        _stochastic_report(st, tol, window)
    else
        r = _stochastic_report(st, tol, window; extra_notes = notes)
        ConvergenceReport(false, :early_stop, r.ΔE, tol, window, nothing, r.cond_S, r.notes)
    end
    return _solution(st, ctx, [StageResult(alg, stage_hist, rep)])
end
```

- [ ] **Step 4: Wire in and run**

`src/FewBodyECG.jl`: `include("solve.jl")` after `include("state.jl")`. Add test include. Run `test/test_solve.jl` → PASS. Runic-format.

- [ ] **Step 5: Run the full suite** — `julia --project=. -e 'using Pkg; Pkg.test()'` → old + new all pass (old API untouched).

- [ ] **Step 6: Checkpoint — USER COMMIT** (suggested: `feat: solve(ops, SVM()) — unified stochastic growth loop with saturation reports`)

---

### Task 5: `Refine` (`src/solve.jl` additions)

**Files:**
- Modify: `src/solve.jl` (append), `test/runtests.jl`
- Create: `test/test_refine.jl`

**Interfaces:**
- Consumes: `rebuild_without`, `commit!`, `score_candidate`, `_candidate_columns`, `_draw_candidate!`, `_stochastic_report`, `_solution`, `_solution_basis_state`.
- Produces: `step!(st, ::Refine, ctx) -> (st′, improved::Bool)` (one full sweep) and `_solve(terms, masses, ::Refine; ...)`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_refine.jl`:

```julia
using Test
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "Refine" begin
    # Deliberately poor starting basis (wrong scale), then refine at scale 1.
    poor = solve(ops, SVM(basis = 10, candidates = 5, scale = 4.0))
    ref = solve(ops, Refine(sweeps = 2, candidates = 25, scale = 1.0); init = poor)
    @test ref isa Solution
    @test ref.E₀ <= poor.E₀ + 1.0e-12               # never raises the energy
    @test ref.E₀ < poor.E₀ - 1.0e-3                 # actually improves a bad basis
    @test length(ref.basis.functions) == length(poor.basis.functions)
    @test ref.stages[end].method isa Refine
    @test length(energies(ref, length(ref.stages))) == 2    # one entry per sweep

    # standalone Refine without a basis is a user error
    @test_throws ArgumentError solve(ops, Refine(1))
end
```

- [ ] **Step 2: Run to verify failure** — `MethodError: _solve(..., ::Refine ...)`.

- [ ] **Step 3: Append to `src/solve.jl`**

```julia
# One refinement sweep (Suzuki–Varga r1–r4): for each basis slot, rebuild the
# (k−1)-state from cached columns, then keep the best of {current, candidates}.
function step!(st::BasisState, alg::Refine, ctx::_SolveCtx)
    scale = _resolve_scale(alg.scale, ctx.masses)
    improved = false
    for i in 1:nfuns(st)
        k = nfuns(st)
        base = rebuild_without(st, i)
        # score the incumbent from cached columns
        idx = setdiff(1:k, i)
        cur_cols = (st.S[idx, i], st.H[idx, i], st.S[i, i], st.H[i, i])
        bestE = something(
            score_candidate(
                base.eig, cur_cols...;
                state = ctx.state, min_resid_ratio = 0.0
            ), Inf
        )
        best, bestcols = st.basis[i], cur_cols
        replaced = false
        for _ in 1:alg.candidates
            cand = _draw_candidate!(base, scale, alg.sampler, ctx.w_list, ctx.d)
            cols = _candidate_columns(cand, base.basis, ctx.terms)
            cols === nothing && continue
            E = score_candidate(
                base.eig, cols...;
                state = ctx.state, min_resid_ratio = alg.indep_tol
            )
            E === nothing && continue
            if E < bestE - 1.0e-12
                bestE, best, bestcols, replaced = E, cand, cols, true
            end
        end
        commit!(base, best, bestcols)
        base.E_hist = copy(st.E_hist)
        st = base
        improved |= replaced
    end
    push!(st.E_hist, st.eig.ε[min(ctx.state, length(st.eig.ε))])
    ctx.verbose && @info "refine sweep done" E = last(st.E_hist)
    return st, improved
end

function _solve(terms, masses, alg::Refine;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false)
    init === nothing && throw(
        ArgumentError("Refine requires an existing basis: pass init = sol or use a pipeline")
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    st = _solution_basis_state(init, ctx.terms)
    sweep_hist = Float64[]
    for _ in 1:alg.sweeps
        st, _ = step!(st, alg, ctx)
        push!(sweep_hist, last(st.E_hist))
    end
    ΔE = length(sweep_hist) ≥ 2 ? sweep_hist[end - 1] - sweep_hist[end] :
        (isempty(init.stages) ? NaN : last(energies(init)) - sweep_hist[end])
    sat = isfinite(ΔE) && 0 ≤ ΔE < tol
    rep = ConvergenceReport(
        sat, sat ? :saturation : :max_steps, ΔE, tol, 1, nothing,
        cond(Symmetric(st.S)), [SATURATION_CAVEAT,
            "refinement: ΔE measured per sweep (window = 1 sweep)"]
    )
    return _solution(st, ctx, [StageResult(alg, sweep_hist, rep)])
end
```

Note: `step!` for `Refine` returns the **new** state (rebuild creates a fresh object) — callers must rebind, as `_solve` does.

- [ ] **Step 4: Wire in test include, run** `test/test_refine.jl` → PASS. Runic-format.

- [ ] **Step 5: Checkpoint — USER COMMIT** (suggested: `feat: Refine — Suzuki–Varga 4.2.6 cyclic replacement via rebuild_without`)

---

### Task 6: Gradient-family adapters (`src/solve.jl` + `src/variational.jl`)

**Files:**
- Modify: `src/variational.jl` (return raw engine data; accept init θ), `src/solve.jl` (append adapters), `test/runtests.jl`
- Create: `test/test_gradient.jl`

**Interfaces:**
- Consumes (existing, in `variational.jl`): `_encode_basis(::BasisSet)`, `_decode_basis(θ, n, n_dim)`, the LBFGS fg machinery in `solve_ECG_variational` / `solve_ECG_sequential`.
- Produces:
  - In `variational.jl`: `_variational_engine(terms, n, θ0, scale, maxiter, gtol, verbose) -> (basis::BasisSet, fg_hist::Vector{Float64}, gradnorm::Float64)` — the core of `solve_ECG_variational` extracted, `θ0 === nothing` ⇒ QMC init at the given scale.
  - `_sequential_engine(terms, n, θ0, scale, candidates, maxiter_step, gtol, verbose) -> (basis, step_hist::Vector{Float64}, fg_hist, gradnorm)` — core of `solve_ECG_sequential`, `θ0` seeds `θ_running`.
  - In `solve.jl`: `_solve(terms, masses, ::Variational; ...)`, `_solve(terms, masses, ::GrowVariational; ...)`, `_gradient_report(gradnorm, gtol, ΔE, cond_S) -> ConvergenceReport`, `_solution_from_basis(basis, ctx, stages)` (dense eigensolve of the final basis via `build_*_matrix` + `solve_generalized_eigenproblem`).

- [ ] **Step 1: Write the failing tests**

Create `test/test_gradient.jl`:

```julia
using Test
using LinearAlgebra
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "Variational and GrowVariational" begin
    sol = solve(ops, Variational(basis = 8, scale = 1.0, maxiter = 300))
    @test sol.E₀ ≈ -0.5 atol = 1.0e-2
    @test sol.E₀ > -0.5 - 1.0e-6
    @test sol.convergence.criterion in (:stationarity, :max_steps)
    @test sol.convergence.gradnorm isa Float64
    @test sol.convergence.window == 0
    @test !isempty(energies(sol))

    # warm start from a stochastic run must not be worse than the start
    svm = solve(ops, SVM(basis = 8, candidates = 10, scale = 1.0))
    ref = solve(ops, Variational(basis = 8, maxiter = 200); init = svm)
    @test ref.E₀ <= svm.E₀ + 1.0e-10
    @test length(ref.basis.functions) == 8

    # init size mismatch is a clear user error
    @test_throws ArgumentError solve(ops, Variational(basis = 5); init = svm)

    g = solve(ops, GrowVariational(basis = 5, candidates = 5, scale = 1.0))
    @test g.E₀ < -0.45
    @test length(energies(g)) == length(g.basis.functions)
end
```

- [ ] **Step 2: Run to verify failure** — `MethodError: _solve(..., ::Variational ...)`.

- [ ] **Step 3: Extract engines in `src/variational.jl`**

Refactor `solve_ECG_variational` minimally: move its body from "build initial basis" through the `optimize` call into

```julia
# Core LBFGS engine.  θ0 === nothing ⇒ fresh QMC basis of n functions at
# `scale`.  Returns the optimised basis, the cumulative-min fg history, and
# the final gradient norm from OptimKit's normgradhistory.
function _variational_engine(terms, n::Int, θ0, scale::Float64,
        maxiter::Int, gtol::Float64, verbose::Bool)
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    if θ0 === nothing
        w_list = [op.w for op in terms if op isa CoulombOperator]
        fns = Rank0Gaussian[]
        for i in 1:n
            bij = generate_bij(:quasirandom, i, length(w_list), scale)
            A = _generate_A_matrix(bij, w_list)
            s = generate_shift(:quasirandom, i, n_dim, scale)
            push!(fns, Rank0Gaussian(A, s))
        end
        θ0 = _encode_basis(BasisSet(fns))
    end
    # ... existing fg closure and LBFGS call of solve_ECG_variational,
    # verbatim, operating on θ0 ...
    x, _, _, _, normgradhistory = Base.CoreLogging.with_logger(
        Base.CoreLogging.ConsoleLogger(Base.stderr, Base.CoreLogging.Error)
    ) do
        optimize(fg, θ0, LBFGS(; maxiter, gradtol = gtol, verbosity = 0))
    end
    basis = _decode_basis(x, n, n_dim)
    fg_hist = accumulate(min, energy_log)
    return basis, fg_hist, float(last(normgradhistory))
end
```

`solve_ECG_variational` becomes a thin call to the engine that re-wraps into the old `SolverResults` (it is deleted in Task 10; keeping it alive keeps old tests green until Task 9's port). Apply the same extraction to `solve_ECG_sequential` → `_sequential_engine(terms, n, θ0, scale, candidates, maxiter_step, gtol, verbose)` returning `(basis, step_hist, fg_hist, gradnorm)`, where `θ0` (if given) seeds `θ_running` and growth continues from `length(θ0) ÷ n_per` functions up to `n`.

- [ ] **Step 4: Append adapters to `src/solve.jl`**

```julia
function _gradient_report(gradnorm, gtol, ΔE, cond_S)
    conv = gradnorm < gtol
    return ConvergenceReport(
        conv, conv ? :stationarity : :max_steps, ΔE, gtol, 0,
        gradnorm, cond_S,
        ["stationary point of the parameter optimisation; " *
         "the variational upper bound still applies"]
    )
end

# Assemble a Solution by one dense eigensolve of the final basis.
function _solution_from_basis(basis::BasisSet, ctx::_SolveCtx, stages)
    H = build_hamiltonian_matrix(basis, ctx.terms)
    S = build_overlap_matrix(basis)
    evals, evecs = solve_generalized_eigenproblem(H, S)
    return Solution(
        evals, basis, evecs, ctx.terms,
        min(ctx.state, length(evals)), stages, last(stages).report
    )
end

function _init_θ(init::Solution, n::Int)
    length(init.basis.functions) == n || throw(
        ArgumentError(
            "init has $(length(init.basis.functions)) functions but the method " *
            "expects basis = $n; set basis = $(length(init.basis.functions))"
        )
    )
    return _encode_basis(BasisSet(Rank0Gaussian[g for g in init.basis.functions]))
end

function _solve(terms, masses, alg::Variational;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false)
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    scale = _resolve_scale(alg.scale, ctx.masses)
    θ0 = init === nothing ? nothing : _init_θ(init, alg.basis)
    basis, fg_hist, gradnorm =
        _variational_engine(ctx.terms, alg.basis, θ0, scale, alg.maxiter, alg.gtol, verbose)
    ΔE = length(fg_hist) ≥ 2 ? abs(fg_hist[end - 1] - fg_hist[end]) : NaN
    S = build_overlap_matrix(basis)
    rep = _gradient_report(gradnorm, alg.gtol, ΔE, cond(Symmetric(S)))
    return _solution_from_basis(basis, ctx, [StageResult(alg, fg_hist, rep)])
end

function _solve(terms, masses, alg::GrowVariational;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false)
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    scale = _resolve_scale(alg.scale, ctx.masses)
    θ0 = init === nothing ? nothing :
        _encode_basis(BasisSet(Rank0Gaussian[g for g in init.basis.functions]))
    basis, step_hist, _, gradnorm = _sequential_engine(
        ctx.terms, alg.basis, θ0, scale, alg.candidates, alg.maxiter_step,
        alg.gtol, verbose
    )
    ΔE = length(step_hist) ≥ 2 ? step_hist[end - 1] - step_hist[end] : NaN
    S = build_overlap_matrix(basis)
    rep = _gradient_report(gradnorm, alg.gtol, ΔE, cond(Symmetric(S)))
    return _solution_from_basis(basis, ctx, [StageResult(alg, step_hist, rep)])
end
```

- [ ] **Step 5: Run** `test/test_gradient.jl` → PASS; then the **full suite** (old variational tests still pass through the thin wrappers). Runic-format changed files.

- [ ] **Step 6: Checkpoint — USER COMMIT** (suggested: `feat: Variational/GrowVariational behind solve() with stationarity reports`)

---

### Task 7: Pipelines

**Files:**
- Modify: `src/solve.jl` (append), `test/runtests.jl`
- Create: `test/test_pipeline.jl`

**Interfaces:**
- Consumes: all `_solve` methods; `StageResult`.
- Produces: `_solve(terms, masses, p::Pipeline; ...)` threading `init` and concatenating stages.

- [ ] **Step 1: Write the failing tests**

Create `test/test_pipeline.jl`:

```julia
using Test
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "Pipelines" begin
    p = SVM(basis = 12, candidates = 10, scale = 1.0) →
        Refine(sweeps = 1, candidates = 15, scale = 1.0) →
        Variational(basis = 12, maxiter = 200)
    sol = solve(ops, p)
    @test length(sol.stages) == 3
    @test sol.stages[1].method isa SVM
    @test sol.stages[3].method isa Variational
    # monotone: each stage's final energy ≤ the previous stage's
    finals = [last(s.energies) for s in sol.stages]
    @test all(diff(finals) .<= 1.0e-10)
    @test sol.convergence === sol.stages[end].report
    @test occursin("→", sprint(show, MIME"text/plain"(), sol))
    # pipeline respects an outer init
    pre = solve(ops, SVM(basis = 6, candidates = 10, scale = 1.0))
    sol2 = solve(ops, SVM(basis = 6, candidates = 10, scale = 1.0) →
        Variational(basis = 12, maxiter = 100); init = pre)
    @test length(sol2.basis.functions) == 12
end
```

- [ ] **Step 2: Run to verify failure** — `MethodError: _solve(..., ::Pipeline ...)`.

- [ ] **Step 3: Append to `src/solve.jl`**

```julia
function _solve(terms, masses, p::Pipeline;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false)
    isempty(p.stages) && throw(ArgumentError("empty pipeline"))
    stages = StageResult[]
    sol = init
    for alg in p.stages
        sol = _solve(terms, masses, alg; state, tol, window, init = sol, verbose)
        append!(stages, sol.stages)
    end
    return Solution(
        sol.E, sol.basis, sol.coefficients, sol.operators, sol.state,
        stages, last(stages).report
    )
end
```

- [ ] **Step 4: Run** `test/test_pipeline.jl` → PASS; wire include; Runic-format.

- [ ] **Step 5: Checkpoint — USER COMMIT** (suggested: `feat: solver pipelines — SVM → Refine → Variational with warm starts`)

---

### Task 8: `Wavefunction` + RecipesBase recipes

**Files:**
- Create: `src/observables.jl`, `src/recipes.jl`, `test/test_observables.jl`
- Modify: `src/FewBodyECG.jl` (includes + `export wavefunction, Wavefunction`), `test/runtests.jl`

**Interfaces:**
- Consumes: `Solution`, `_polar_projection` (types.jl), Gaussian types.
- Produces: `Wavefunction` (fields `basis::BasisSet`, `c::Vector{Float64}`), callable `(ψ::Wavefunction)(r::AbstractVector)`, `wavefunction(sol; state = sol.state)`; recipes for `Solution` (optionally `plot(sol, reference)`) and `Wavefunction`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_observables.jl`:

```julia
using Test
using RecipesBase
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"
sol = solve(ops, SVM(basis = 15, candidates = 15, scale = 1.0))

@testset "Wavefunction" begin
    ψ = wavefunction(sol)
    @test ψ isa Wavefunction
    @test isfinite(ψ([0.5]))
    # matches the explicit linear combination
    c = sol.coefficients[:, 1]
    fns = sol.basis.functions
    ref = sum(c[i] * exp(-([0.5]' * fns[i].A * [0.5]) + fns[i].s' * [0.5])
              for i in eachindex(fns))
    @test ψ([0.5]) ≈ ref rtol = 1.0e-12
    # Rank1 evaluation: (aᵀr)·exp(−rᵀAr)
    g1 = Rank1Gaussian([1.0;;], [1.0], [0.0])
    ψ1 = Wavefunction(BasisSet([g1]), [1.0])
    @test ψ1([0.7]) ≈ 0.7 * exp(-0.49) rtol = 1.0e-12
end

@testset "Recipes" begin
    # convergence recipe
    plots = RecipesBase.apply_recipe(Dict{Symbol, Any}(), sol)
    @test !isempty(plots)
    # with reference energy
    plots2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(), sol, -0.5)
    @test length(plots2) ≥ 2
    # wavefunction recipe
    ψ = wavefunction(sol)
    wplots = RecipesBase.apply_recipe(Dict{Symbol, Any}(), ψ)
    @test !isempty(wplots)
end
```

- [ ] **Step 2: Run to verify failure** — `UndefVarError: Wavefunction`.

- [ ] **Step 3: Implement `src/observables.jl`**

```julia
"""
    Wavefunction

Callable variational wavefunction `ψ(r) = Σᵢ cᵢ gᵢ(r)` in **Jacobi
coordinates** (mass-weighted: the package's Jacobi transform normalises each
relative coordinate by √μ — see `jacobi_transform`).  Obtained from
[`wavefunction`](@ref); plot with `plot(ψ; coord = i)`.
"""
struct Wavefunction
    basis::BasisSet
    c::Vector{Float64}
end

_gauss(g, r) = exp(-(r' * g.A * r) + g.s' * r)
_eval(g::Rank0Gaussian, r) = _gauss(g, r)
_eval(g::Rank1Gaussian, r) = sum(_polar_projection(g.a, r)) * _gauss(g, r)
_eval(g::Rank2Gaussian, r) =
    dot(_polar_projection(g.a, r), _polar_projection(g.b, r)) * _gauss(g, r)

(ψ::Wavefunction)(r::AbstractVector) =
    sum(ψ.c[i] * _eval(ψ.basis.functions[i], r) for i in eachindex(ψ.c))

"""
    wavefunction(sol::Solution; state = sol.state) -> Wavefunction
"""
wavefunction(sol::Solution; state::Int = sol.state) =
    Wavefunction(getfield(sol, :basis), getfield(sol, :coefficients)[:, state])
```

- [ ] **Step 4: Implement `src/recipes.jl`**

```julia
using RecipesBase

# plot(sol):        per-stage energy curves vs cumulative step
# plot(sol, E_ref): same, plus a reference-energy hline
@recipe function f(sol::Solution, reference::Union{Nothing, Real} = nothing)
    xguide --> "step"
    yguide --> "E (Ha)"
    legend --> :topright
    offset = 0
    for st in getfield(sol, :stages)
        xs = offset .+ (1:length(st.energies))
        offset += length(st.energies)
        @series begin
            label --> sprint(show, st.method)
            seriestype --> :path
            linewidth --> 2
            xs, st.energies
        end
    end
    if reference !== nothing
        @series begin
            label --> "reference"
            seriestype --> :hline
            linestyle --> :dash
            [float(reference)]
        end
    end
end

# plot(ψ; coord = 1, rmax = 10.0, npoints = 400): radial profile r²|ψ|²
# along one Jacobi coordinate (others fixed at 0).
@recipe function f(ψ::Wavefunction; coord = 1, rmax = 10.0, npoints = 400)
    d = length(first(ψ.basis.functions).s)
    1 ≤ coord ≤ d || throw(ArgumentError("coord must be in 1:$d"))
    rs = range(1.0e-3, rmax, length = npoints)
    ys = map(rs) do r
        v = zeros(d)
        v[coord] = r
        r^2 * abs2(ψ(v))
    end
    xguide --> "r (Jacobi coordinate $coord, mass-weighted)"
    yguide --> "r²|ψ(r)|²"
    label --> "|ψ|²"
    linewidth --> 2
    collect(rs), ys
end
```

Compatibility note: if RecipesBase mishandles the optional positional
`reference` argument (default-arg expansion inside `@recipe` varies across
versions), split it into two recipes — `@recipe f(sol::Solution)` with the
stage loop only, and `@recipe f(sol::Solution, reference::Real)` duplicating
the loop plus the hline series. The Task-8 tests exercise both call forms
and will catch this immediately.

- [ ] **Step 5: Wire in (`include("observables.jl")`, `include("recipes.jl")` after solve.jl; exports), run tests** → PASS. Runic-format. Full suite → green.

- [ ] **Step 6: Checkpoint — USER COMMIT** (suggested: `feat: Wavefunction + RecipesBase plotting for solutions and wavefunctions`)

---

### Task 9: Port the legacy test suite to the new API

**Files:**
- Modify: `test/test_hamiltonian.jl`, `test/test_variational.jl`, `test/test_utils.jl`, `test/test_svm_eigen.jl` (solver-facing testsets only), `test/test_operators.jl` (only if it calls `solve_ECG*`)

**Interfaces:** Consumes the full new API. Produces a test suite with **zero references to** `solve_ECG`, `solve_ECG_competitive`, `solve_ECG_variational`, `solve_ECG_sequential`, `SolverResults`, `ψ₀`, `convergence(`, `convergence_history(`, `correlation_function(` — so Task 10 can delete them without breaking tests.

- [ ] **Step 1: Inventory** — `grep -rn "solve_ECG\|SolverResults\|ψ₀\|correlation_function\|convergence(" test/` and list every hit.

- [ ] **Step 2: Port, file by file, using this exact mapping**

| Old call | New call |
|---|---|
| `solve_ECG(ops, n; scale = s, verbose = false)` | `solve(ops, SVM(basis = n, candidates = 1, scale = s))` |
| `solve_ECG_competitive(ops, n; n_candidates = K, scale = s, verbose = false)` | `solve(ops, SVM(basis = n, candidates = K, scale = s))` |
| `solve_ECG_variational(ops, n; scale = s, verbose = false)` | `solve(ops, Variational(basis = n, scale = s))` |
| `solve_ECG_sequential(ops, n; scale = s, verbose = false)` | `solve(ops, GrowVariational(basis = n, scale = s))` |
| `sr.ground_state` | `sol.E₀` |
| `sr.basis_functions` | `sol.basis.functions` |
| `sr.energies` | `energies(sol)` |
| `sr.eigenvectors[end][:, k]` | `sol.coefficients[:, k]` |
| `ψ₀(r, sr)` / `ψ₀(r, c, fns)` | `wavefunction(sol)(r)` / `Wavefunction(BasisSet(fns), c)(r)` |
| `convergence(sr)` / `convergence_history(sr)` | `(1:length(energies(sol)), energies(sol))` |
| `correlation_function(sr; ...)` | delete the testset (feature removed; recipe covers plotting) |

Keep every numerical tolerance and anchor **identical**. In `test_svm_eigen.jl`, only the "competitive solver is self-consistent with LAPACK" testset changes (`solve_ECG_competitive` → `solve(ops, SVM(...))`, `sr.ground_state` → `sol.E₀`, `sr.basis_functions` → `sol.basis.functions`, `ψ₀([0.5], sr)` → `wavefunction(sol)([0.5])`); the 49 eigensolver tests are untouched.

- [ ] **Step 3: Run the full suite** — all green, old API still present but now unreferenced by tests.

- [ ] **Step 4: Checkpoint — USER COMMIT** (suggested: `test: port suite to solve()/Solution API`)

---

### Task 10: The cutover — delete old API, split/rename files, scrub exports

**Files:**
- Delete: `src/svm_solver.jl`, `src/utils.jl`
- Create: `src/operators.jl`, `src/linalg.jl` (both extracted from `src/hamiltonian.jl`, then delete `src/hamiltonian.jl`)
- Rename: `src/svm_eigen.jl` → `src/eigen.jl`, `src/variational.jl` → `src/gradient.jl`
- Modify: `src/FewBodyECG.jl` (includes + final export list), `src/coordinates.jl` (rename `_jacobi_transform` → `jacobi_transform` with an internal `const _jacobi_transform = jacobi_transform` NOT kept — update all call sites), `src/gradient.jl` (delete `solve_ECG_variational`/`solve_ECG_sequential` wrappers, keep engines), `test/*` (update any `_jacobi_transform`/import references), `test/Aqua.jl` unchanged

**Interfaces:** Produces the final public surface of spec §3.9 — nothing else exported.

- [ ] **Step 1: Split `hamiltonian.jl`**

`src/operators.jl` ← the `Operators` struct, constructors, all `Base.:+` methods, `Base.length/iterate/getindex/eltype/show`, `coulomb_weights`, and the `solve(ops::Operators, ...)`-style forwards **except** the old solvers. `src/linalg.jl` ← `_compute_overlap_element`, `build_overlap_matrix`, `_build_operator_matrix`, `build_hamiltonian_matrix` (both methods), `solve_generalized_eigenproblem`, `normalized_overlap`, `is_linearly_independent`, `default_scale`. **Delete** `solve_ECG` (the greedy loop) and the old-API forwards (`solve_ECG(ops::Operators, ...)` etc.). Delete `src/hamiltonian.jl`.

- [ ] **Step 2: Delete `src/svm_solver.jl` and `src/utils.jl`**

`svm_solver.jl` is fully superseded by `state.jl` + `solve.jl`. From `utils.jl` nothing survives: `SolverResults`, `ψ₀`, `convergence`, `convergence_history`, `correlation_function`, `ψ` all go (spec §3.9). ⚠️ `SolverResults` is constructed in `gradient.jl`'s old wrappers — delete those wrappers in the same step.

- [ ] **Step 3: Renames**

`git mv`-style renames are the author's; executors instead create the new files with identical content and delete the old ones is NOT allowed (no git) — so: **rename via file write**: copy `svm_eigen.jl` content to `eigen.jl`, `variational.jl` (minus deleted wrappers) to `gradient.jl`, remove originals with `rm` (plain filesystem, not git). In `coordinates.jl` rename `_jacobi_transform` → `jacobi_transform` (docstring updated to state it is public); `grep -rn "_jacobi_transform" src/ test/ examples/ docs/` and update every site.

- [ ] **Step 4: Rewrite `src/FewBodyECG.jl`**

```julia
module FewBodyECG

using LinearAlgebra
import Antique
using FewBodyHamiltonians

const Operator = FewBodyHamiltonians.Operator

# system building
export Operators, coulomb_weights, Operator,
    KineticOperator, CoulombOperator, GaussianOperator,
    GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet
# solving
export solve, SVM, Refine, Variational, GrowVariational, Pipeline, →, AutoDiff
# results
export Solution, ConvergenceReport, StageResult, converged, energies,
    wavefunction, Wavefunction
# power-user layer
export build_hamiltonian_matrix, build_overlap_matrix,
    solve_generalized_eigenproblem, Λ, jacobi_transform, default_scale

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("operators.jl")
include("linalg.jl")
include("eigen.jl")
include("sampling.jl")
include("methods.jl")
include("solution.jl")
include("state.jl")
include("gradient.jl")
include("solve.jl")
include("observables.jl")
include("recipes.jl")

end
```

- [ ] **Step 5: Run the full suite + Aqua** — `julia --project=. -e 'using Pkg; Pkg.test()'`. Expect failures only from stale imports in tests (e.g. `FewBodyECG: _generate_A_matrix` still fine — internal but existing; `_jacobi_transform` renamed). Fix until green. Verify the export surface: `julia --project=. -e 'using FewBodyECG; println(sort(names(FewBodyECG)))'` matches spec §3.9 exactly.

- [ ] **Step 6: Set `version = "2.0.0"` in `Project.toml`.**

- [ ] **Step 7: Checkpoint — USER COMMIT** (suggested: `feat!: v2.0 cutover — delete solve_ECG* API, split hamiltonian.jl, scrub exports`)

---

### Task 11: Example gallery (7 uniform Literate examples)

**Files:**
- Delete: all files in `examples/`
- Create: `examples/hydrogen.jl`, `examples/positronium.jl`, `examples/helium.jl`, `examples/tdmu.jl`, `examples/h2plus.jl`, `examples/gaussian_well.jl`, `examples/workflow.jl`

Each file: Literate.jl conventions (`# ` markdown lines, `#src` for excluded lines), ≤ ~40 code lines, shape *build → solve → display → plot*, unicode names. Complete content for two representative files below; the remaining five follow the identical template with the systems/anchors from spec §6 (helium: `Operators([1e15,1,1],[+2,-1,-1])`, anchor −2.9037; positronium: `Operators([1,1],[+1,-1])`, anchor −0.25; gaussian_well: the current GaussianWell physics ported; tdmu: masses `[5496.918, 3670.481, 206.7686]`, charges `[+1,+1,-1]`, anchor −111.36444 with `tol` guidance; workflow: the pipeline of h2plus with a stage-coloured `plot(sol)`).

- [ ] **Step 1: Write `examples/hydrogen.jl`**

```julia
# # Hydrogen: s-, p- and d-waves
#
# The classic first test (Fedorov et al., Few-Body Syst 65:75): exact
# energies −1/2, −1/8 and −1/18 Ha for the lowest s, p and d states.

using FewBodyECG
using Plots

# ## Ground state (s-wave, rank-0 Gaussians)
ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0))
sol

# The convergence statement above is a *saturation* statement — the
# variational upper bound guarantees E₀ ≥ −1/2 exactly.
plot(sol, -0.5)

# ## p- and d-waves (rank-1 / rank-2 prefactor Gaussians, manual basis)
# Prefactor bases are built by hand and solved with the power-user layer:
αs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
basis₁ = BasisSet([Rank1Gaussian([α;;], [1.0], [0.0]) for α in αs])
H = build_hamiltonian_matrix(basis₁, ops)
S = build_overlap_matrix(basis₁)
E₁, _ = solve_generalized_eigenproblem(H, S)
println("2p energy: ", minimum(E₁), "  (exact −0.125)")

a = reshape([1.0, 0.0, 0.0], 1, 3)   # a ⊥ b ⇒ pure d-wave
b = reshape([0.0, 1.0, 0.0], 1, 3)
αd = exp10.(range(log10(0.002), log10(0.8), length = 24))
basis₂ = BasisSet([Rank2Gaussian([α;;], a, b, [0.0]) for α in αd])
E₂, _ = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis₂, ops), build_overlap_matrix(basis₂)
)
println("3d energy: ", minimum(E₂), "  (exact −1/18 ≈ −0.05556)")
```

- [ ] **Step 2: Write `examples/h2plus.jl`**

```julia
# # H₂⁺ without Born–Oppenheimer
#
# The dihydrogen cation as a *direct* three-body Coulomb problem — no
# adiabatic separation.  Reference non-BO ground state: −0.597139 Ha; the
# molecule is bound because E₀ < −0.5 Ha (the H + p⁺ threshold).

using FewBodyECG
using Plots

mₚ = 1836.15267343
ops = Operators([mₚ, mₚ, 1.0], [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

# The recommended workflow: cheap stochastic exploration, cyclic
# refinement, then gradient optimisation of every Gaussian.
sol = solve(ops, SVM(basis = 60, candidates = 30, scale = 1.0)
                 → Refine(sweeps = 2, scale = 1.0)
                 → Variational(basis = 60, maxiter = 300))
sol

# Stage-by-stage convergence toward the reference energy:
plot(sol, -0.597139)

# The wavefunction along the proton–proton Jacobi coordinate
# (mass-weighted — see the docs on coordinates):
ψ = wavefunction(sol)
plot(ψ; coord = 1, rmax = 80.0)
```

- [ ] **Step 3: Write the remaining five examples** with the same template and the anchors listed above; every example ends with a `plot`. Run each headless: `GKSwstype=100 julia --project=. examples/FILE.jl` → exits 0. (Examples use Plots; verify Plots is available in the shared environment as today, or run with `--project=examples` if an examples project exists — match current repo practice.)

- [ ] **Step 4: Checkpoint — USER COMMIT** (suggested: `docs: uniform v2 example gallery (7 Literate examples)`)

---

### Task 12: Documentation rewrite

**Files:**
- Modify: `docs/make.jl`, `docs/Project.toml` (+ Literate), `docs/src/index.md`, `docs/src/API.md`, `docs/src/theory.md`
- Create: `docs/src/systems.md`, `docs/src/solvers.md`, `docs/src/convergence.md`
- Delete: `docs/src/examples.md`, `docs/src/resources.md` (content folded into new pages)

- [ ] **Step 1: `docs/make.jl`** — add Literate preprocessing of `examples/*.jl` into `docs/src/examples/`, page list:

```julia
using Documenter, Literate, FewBodyECG

const EXDIR = joinpath(@__DIR__, "..", "examples")
const OUTDIR = joinpath(@__DIR__, "src", "examples")
for f in readdir(EXDIR; join = true)
    endswith(f, ".jl") && Literate.markdown(f, OUTDIR; documenter = true)
end

makedocs(
    build = "build",
    modules = [FewBodyECG],
    checkdocs = :exports,
    sitename = "FewBodyECG.jl",
    pages = [
        "Home" => "index.md",
        "Building systems" => "systems.md",
        "Choosing a solver" => "solvers.md",
        "Convergence" => "convergence.md",
        "Examples" => [
            "Hydrogen" => "examples/hydrogen.md",
            "Positronium" => "examples/positronium.md",
            "Helium & H⁻" => "examples/helium.md",
            "tdμ" => "examples/tdmu.md",
            "H₂⁺ (non-BO)" => "examples/h2plus.md",
            "Gaussian wells" => "examples/gaussian_well.md",
            "Workflow" => "examples/workflow.md",
        ],
        "Theory" => "theory.md",
        "API" => "API.md",
    ],
    format = Documenter.HTML()
)
deploydocs(repo = "github.com/JuliaFewBody/FewBodyECG.jl", target = "build",
    branch = "gh-pages", devbranch = "main")
```

Add `Literate` + `Plots` to `docs/Project.toml`.

- [ ] **Step 2: Write the six prose pages.** Required content per page (write full prose; each page 60–150 lines):
  - **index.md**: one-paragraph pitch; the 10-line hydrogen quickstart (`Operators` → `solve(ops, SVM(...))` → displayed solution block → `plot(sol, -0.5)`); install instructions; links to the other pages.
  - **systems.md**: `Operators(masses, charges)`; `+= "Kinetic"` / `+= "Coulomb"` / explicit pairs / `("Gaussian", i, j, V₀, γ)`; atomic units; Jacobi coordinates **including the mass-weighted convention and the √μ factor** with `jacobi_transform`/`Λ`; `scale` and `default_scale`; manual Rank1/Rank2 basis construction with `build_*_matrix` + `solve_generalized_eigenproblem` (this is the documented Rank1/2 path).
  - **solvers.md**: one subsection per method with its options table; the *choosing* guidance verbatim from spec §5.3: single-scale stochastic sampling saturates on multiscale systems; gradient methods move Gaussians where sampling can't reach; recommended default workflow `SVM → Refine → Variational`; cost table (SVM step O(k²) per candidate; Refine sweep O(k⁴) worst case; Variational O(iter · n_param · k³ engine cost)).
  - **convergence.md**: `:saturation` vs `:stationarity` vs the exact eigenvalue; the H₂⁺-style plateau example (a saturated report at the wrong energy) as a worked warning; the variational upper bound; `cond(S)` and the whitened eigensolver; reading `plot(sol)`.
  - **theory.md**: keep existing ECG summary; add subsections for the incremental whitened arrowhead eigensolver (Theorem 3.5 sketch, whitening rationale) and the exact citations (Suzuki–Varga 1998; Fedorov 2017; Fedorov et al. 2024).
  - **API.md**: `@docs` blocks grouped exactly as the export list in `src/FewBodyECG.jl` (Task 10 Step 4), adding the new symbols and removing all deleted ones.

- [ ] **Step 3: Build** — `julia --project=docs docs/make.jl` → exit 0, no `missing_docs`, no doctest failures. Fix until clean.

- [ ] **Step 4: Checkpoint — USER COMMIT** (suggested: `docs: complete v2 documentation rewrite with Literate example gallery`)

---

### Task 13: README, CHANGELOG, final verification

**Files:**
- Modify: `README.md`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Rewrite `README.md`** — badges kept; pitch paragraph; the same quickstart as index.md (copy, don't diverge); a "v2.0" note pointing to the CHANGELOG migration table; feature bullets (unified `solve`, honest convergence reports, pipelines, plotting recipes).

- [ ] **Step 2: Write `CHANGELOG.md`** — `## v2.0.0` section: breaking-change banner; the full old→new mapping table from Task 9 Step 2; removed names list (spec §3.9 "Deleted" list); added names list; dependency changes (+RecipesBase).

- [ ] **Step 3: Final verification** — run in order and report all outputs: `julia --project=. -e 'using Pkg; Pkg.test()'` (all green incl. Aqua), `julia --project=docs docs/make.jl` (exit 0), `GKSwstype=100 julia --project=. examples/hydrogen.jl` and `examples/h2plus.jl` (exit 0), export-surface check against spec §3.9.

- [ ] **Step 4: Checkpoint — USER COMMIT** (suggested: `docs: v2.0 README + CHANGELOG; release candidate`) — author tags/releases at their discretion.
