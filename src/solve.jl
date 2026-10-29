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
    units::UnitSystem
    hamiltonian::Union{Nothing, Hamiltonian}
end

_pairwise_weights(terms) = Vector{Float64}[
    Vector{Float64}(op.w) for op in terms
        if op isa Union{CoulombOperator, GaussianOperator, OscillatorOperator, NumericalPotential}
]

function _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian = nothing)
    state ≥ 1 || throw(ArgumentError("state must be ≥ 1, got $state"))
    units = _unitsystem(units)
    init === nothing || init.units == units || throw(
        ArgumentError("init was solved in $(init.units), but this problem uses $units")
    )
    # A basis is defined in the Jacobi coordinates of its masses, so a warm
    # start for other masses would silently describe a different state.
    if init !== nothing && init.hamiltonian !== nothing && hamiltonian !== nothing
        init_masses = _kinetic(init.hamiltonian).masses
        init_masses == masses || throw(
            ArgumentError("init was solved for masses $init_masses, but this problem has masses $masses")
        )
    end
    w_list = _pairwise_weights(terms)
    isempty(w_list) && throw(
        ArgumentError(
            "need at least one pairwise potential term " *
                "(Coulomb/Gaussian/Oscillator/NumericalPotential) " *
                "to define the candidate geometry"
        )
    )
    return _SolveCtx(
        collect(FewBodyHamiltonians.Operator, terms), masses,
        state, float(tol), window, verbose, w_list, length(w_list[1]), units, hamiltonian
    )
end

"""
    solve(H::Hamiltonian, alg::SolverMethod = SVM();
          state = 1, tol = 1e-4, window = 20, init = nothing, verbose = false)
    solve(terms::Vector{<:Operator}, alg::SolverMethod = SVM();
          units = :atomic, kw...)

Solve the few-body eigenproblem defined by a [`Hamiltonian`](@ref) (or a raw
vector of Jacobi-frame operators) with algorithm `alg` — one of [`SVM`](@ref), [`Refine`](@ref), [`GVM`](@ref),
[`DynamicGVM`](@ref), or a [`Pipeline`](@ref) composed with `→`.

Problem-level keywords: `state` targets the `state`-th eigenvalue, `tol`
(absolute, in the system's energy unit) and `window` define the stochastic
saturation criterion, `init` warm-starts from a previous [`Solution`](@ref)
solved in the same units and, for a Hamiltonian, for the same masses. A
Hamiltonian takes its [`UnitSystem`](@ref) from its [`Kinetic`](@ref) term; a
raw operator vector takes it from the `units` keyword. Spin-dependent terms
([`Tensor`](@ref), [`SpinOrbit`](@ref)) need a [`SpinGaussian`](@ref) basis
and are assembled with `FewBodyECG.build_hamiltonian_matrix` instead.

Returns a [`Solution`](@ref) carrying energies, the basis, S-orthonormal
coefficients, and an honest [`ConvergenceReport`](@ref).
"""
function solve(H::Hamiltonian, alg::SolverMethod = SVM(); kw...)
    haskey(kw, :units) && throw(
        ArgumentError("units are set by Kinetic(masses; units = ...), not by solve")
    )
    any(t -> t isa Union{Tensor, SpinOrbit}, H.terms) && throw(
        ArgumentError(
            "spin-dependent terms need a SpinGaussian basis; assemble them with " *
                "FewBodyECG.build_hamiltonian_matrix(basis, H)"
        )
    )
    ops, masses, units = _to_jacobi(H)
    return _solve(ops, masses, alg; units, hamiltonian = H, kw...)
end
solve(terms::Vector{<:FewBodyHamiltonians.Operator}, alg::SolverMethod = SVM(); units = :atomic, kw...) =
    _solve(terms, nothing, alg; units, kw...)

# `scale = :adaptive` sets the sampling length to this multiple of the largest
# rms pair distance.  Coulomb systems do best near 0.5 and smooth nuclear
# potentials near 1–2; 0.7 stays within a small factor of the best energy for
# both.
const _ADAPTIVE_FACTOR = 0.7
const _PILOT_BASIS = 30
const _PILOT_CANDIDATES = 20
const _PILOT_MAX_ROUNDS = 3

# Largest root-mean-square distance ⟨r_ij²⟩^½ over all particle pairs in the
# target state of `sol`, from exact r² matrix elements.
function _rms_pair_max(sol::Solution, masses)
    _, U = jacobi_transform(collect(Float64, masses))
    N = length(masses)
    c = getfield(sol, :coefficients)[:, getfield(sol, :state)]
    basis = getfield(sol, :basis)
    r2 = map(((i, j) for i in 1:N for j in (i + 1):N)) do (i, j)
        e = zeros(N)
        e[i], e[j] = 1.0, -1.0
        O = build_hamiltonian_matrix(basis, FewBodyHamiltonians.Operator[OscillatorOperator(1.0, U' * e)])
        real(c' * O * c)
    end
    return sqrt(maximum(r2))
end

# `scale = :adaptive`: measure the system's size from `init` when there is
# one, otherwise from pilot `SVM` runs with wide width sampling.  Each pilot is
# re-centered on the previous size estimate, until the estimate changes by
# less than 30%.  Short pilots are poorly converged and can overestimate the
# size through diffuse components, so the size comes from the pilot with the
# lowest energy, whose wavefunction is the most accurate.
function _adaptive_scale(ctx::_SolveCtx, init)
    ctx.masses === nothing && throw(
        ArgumentError(
            "scale = :adaptive needs the particle masses: build the system with Kinetic(masses), or pass a numeric scale"
        )
    )
    unit = ctx.units.length
    if init !== nothing
        r = _rms_pair_max(init, ctx.masses)
        s = _ADAPTIVE_FACTOR * r
        return s, "scale = :adaptive → $(round(s; sigdigits = 3)) ($(_ADAPTIVE_FACTOR) × rms pair " *
            "distance $(round(r; sigdigits = 3)) $unit of the initial basis)"
    end
    center = default_scale(collect(Float64, ctx.masses))
    best_E, best_r = Inf, NaN
    r_prev = NaN
    rounds = 0
    for round in 1:_PILOT_MAX_ROUNDS
        pilot = _solve(
            ctx.terms, ctx.masses,
            SVM(; basis = _PILOT_BASIS, candidates = _PILOT_CANDIDATES, scale = center, widths = :wide);
            state = ctx.state, tol = ctx.tol, window = ctx.window, init = nothing,
            verbose = false, units = ctx.units
        )
        r = _rms_pair_max(pilot, ctx.masses)
        isfinite(r) && r > 0 || throw(
            ArgumentError(
                "scale = :adaptive could not measure the system size (pilot rms pair " *
                    "distance $r); pass a numeric scale"
            )
        )
        ctx.verbose && @info "scale = :adaptive: pilot round $round" center energy = pilot.E₀ rms_pair_max = r
        pilot.E₀ < best_E && ((best_E, best_r) = (pilot.E₀, r))
        rounds = round
        isfinite(r_prev) && abs(r - r_prev) ≤ 0.3 * r_prev && break
        r_prev, center = r, r
    end
    s = _ADAPTIVE_FACTOR * best_r
    return s, "scale = :adaptive → $(round(s; sigdigits = 3)) ($(_ADAPTIVE_FACTOR) × rms pair distance " *
        "$(round(best_r; sigdigits = 3)) $unit, from the lowest-energy of $rounds pilot " *
        "runs of $(_PILOT_BASIS) functions with widths = :wide)"
end

# Resolve `scale` once per solve: `(value, note)`, where `note` explains an
# adaptive choice and is `nothing` otherwise.
_solve_scale(scale, ctx::_SolveCtx, init) =
    scale === :adaptive ? _adaptive_scale(ctx, init) : (_resolve_scale(scale, ctx.masses), nothing)

_add_note!(report::ConvergenceReport, note) = (note === nothing || push!(report.notes, note); report)

# One SVM growth step: draw → score all candidates → commit the best.
# Returns false when no admissible candidate was found.
function step!(st::BasisState, alg::SVM, ctx::_SolveCtx, scale::Float64)
    bestE = Inf
    best = nothing
    bestcols = nothing
    for _ in 1:alg.candidates
        cand = _draw_candidate!(st, scale, alg.sampler, alg.widths, ctx.w_list, ctx.d)
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

function _require_available_state(state::Int, nstates::Int)
    state ≤ nstates || throw(
        ArgumentError("requested state $state, but the solver produced only $nstates eigenstates")
    )
    return state
end

function _solution(st::BasisState, ctx::_SolveCtx, stages::Vector{StageResult})
    nfuns(st) > 0 || error("solver produced no basis functions")
    state = _require_available_state(ctx.state, length(st.eig.ε))
    return Solution(
        copy(st.eig.ε), BasisSet(copy(st.basis)), coefficients(st.eig),
        ctx.terms, state,
        stages, last(stages).report, ctx.units, st.draw, ctx.hamiltonian
    )
end

function _solve(
        terms, masses, alg::SVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian)
    st = init === nothing ? BasisState() : _solution_basis_state(init, ctx.terms)
    scale, scale_note = _solve_scale(alg.scale, ctx, init)
    n₀ = length(st.E_hist)
    n₀_funs = nfuns(st)
    failed = 0
    for iteration in 1:alg.basis
        accepted = step!(st, alg, ctx, scale)
        accepted || (failed += 1)
        energy = isempty(st.E_hist) ? NaN : last(st.E_hist)
        ctx.verbose && @info "SVM: iteration $iteration/$(alg.basis)" energy basis = nfuns(st) accepted
    end
    added = nfuns(st) - n₀_funs
    notes = String[]
    if failed > 0
        push!(
            notes,
            "$(failed) of $(alg.basis) growth steps found no admissible candidate " *
                "among $(alg.candidates) draws (indep_tol = $(alg.indep_tol)); " *
                "added $(added) new functions"
        )
    end
    stage_hist = st.E_hist[(n₀ + 1):end]
    rep = if added == 0
        r = _stochastic_report(st, tol, window; extra_notes = notes)
        ConvergenceReport(false, :early_stop, r.ΔE, tol, window, nothing, r.cond_S, r.notes)
    else
        _stochastic_report(st, tol, window; extra_notes = notes)
    end
    return _solution(st, ctx, [StageResult(alg, stage_hist, _add_note!(rep, scale_note))])
end

# One refinement sweep (Suzuki–Varga r1–r4): for each basis slot, rebuild the
# (k−1)-state from cached columns, then keep the best of {current, candidates}.
function step!(st::BasisState, alg::Refine, ctx::_SolveCtx, scale::Float64)
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
            cand = _draw_candidate!(base, scale, alg.sampler, alg.widths, ctx.w_list, ctx.d)
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
    return st, improved
end

function _solve(
        terms, masses, alg::Refine;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing
    )
    init === nothing && throw(
        ArgumentError("Refine requires an existing basis: pass init = sol or use a pipeline")
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian)
    st = _solution_basis_state(init, ctx.terms)
    scale, scale_note = _solve_scale(alg.scale, ctx, init)
    sweep_hist = Float64[]
    for iteration in 1:alg.sweeps
        st, _ = step!(st, alg, ctx, scale)
        push!(sweep_hist, last(st.E_hist))
        ctx.verbose && @info "Refine: iteration $iteration/$(alg.sweeps)" energy = last(st.E_hist)
    end
    ΔE = length(sweep_hist) ≥ 2 ? sweep_hist[end - 1] - sweep_hist[end] :
        (isempty(energy_history(init)) ? NaN : last(energy_history(init)) - sweep_hist[end])
    sat = isfinite(ΔE) && 0 ≤ ΔE < tol
    rep = ConvergenceReport(
        sat, sat ? :saturation : :max_steps, ΔE, tol, 1, nothing,
        cond(Symmetric(st.S)), [
            SATURATION_CAVEAT,
            "refinement: ΔE measured per sweep (window = 1 sweep)",
        ]
    )
    return _solution(st, ctx, [StageResult(alg, sweep_hist, _add_note!(rep, scale_note))])
end

# Fold a Pipeline left-to-right: each stage's own `_solve` runs unmodified,
# warm-started from the previous stage's Solution; every stage's StageResult
# is kept, and the final Solution carries the last stage's basis/coefficients.
function _solve(
        terms, masses, p::Pipeline;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing
    )
    isempty(p.stages) && throw(ArgumentError("empty pipeline"))
    stages = StageResult[]
    sol = init
    for alg in p.stages
        sol = _solve(terms, masses, alg; state, tol, window, init = sol, verbose, units, hamiltonian)
        append!(stages, sol.stages)
    end
    return Solution(
        sol.E, sol.basis, sol.coefficients, sol.operators, sol.state,
        stages, last(stages).report, sol.units, sol.draws, sol.hamiltonian
    )
end
