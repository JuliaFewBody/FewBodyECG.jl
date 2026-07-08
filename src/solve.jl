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
    w_list = Vector{Float64}[
        op.w for op in terms
            if op isa Union{CoulombOperator, GaussianOperator}
    ]
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

function _solve(
        terms, masses, alg::SVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    st = init === nothing ? BasisState() : _solution_basis_state(init, ctx.terms)
    n₀ = length(st.E_hist)
    n₀_funs = nfuns(st)
    failed = 0
    for _ in 1:alg.basis
        step!(st, alg, ctx) || (failed += 1)
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
    return _solution(st, ctx, [StageResult(alg, stage_hist, rep)])
end

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

function _solve(
        terms, masses, alg::Refine;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
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
        (isempty(energies(init)) ? NaN : last(energies(init)) - sweep_hist[end])
    sat = isfinite(ΔE) && 0 ≤ ΔE < tol
    rep = ConvergenceReport(
        sat, sat ? :saturation : :max_steps, ΔE, tol, 1, nothing,
        cond(Symmetric(st.S)), [
            SATURATION_CAVEAT,
            "refinement: ΔE measured per sweep (window = 1 sweep)",
        ]
    )
    return _solution(st, ctx, [StageResult(alg, sweep_hist, rep)])
end

function _gradient_report(gradnorm, gtol, ΔE, cond_S)
    conv = gradnorm < gtol
    return ConvergenceReport(
        conv, conv ? :stationarity : :max_steps, ΔE, gtol, 0,
        gradnorm, cond_S,
        [
            "stationary point of the parameter optimisation; " *
                "the variational upper bound still applies",
        ]
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

function _solve(
        terms, masses, alg::Variational;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
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

function _solve(
        terms, masses, alg::GrowVariational;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    scale = _resolve_scale(alg.scale, ctx.masses)
    θ0 = nothing
    if init !== nothing
        k0 = length(init.basis.functions)
        k0 < alg.basis || throw(
            ArgumentError(
                "init already has $k0 functions but GrowVariational grows to basis = $(alg.basis); " *
                    "set basis > $k0 or use Variational(basis = $k0) to re-optimise"
            )
        )
        θ0 = _encode_basis(BasisSet(Rank0Gaussian[g for g in init.basis.functions]))
    end
    basis, step_hist, _, gradnorm = _sequential_engine(
        ctx.terms, alg.basis, θ0, scale, alg.candidates, alg.maxiter_step,
        alg.gtol, verbose
    )
    ΔE = length(step_hist) ≥ 2 ? step_hist[end - 1] - step_hist[end] : NaN
    S = build_overlap_matrix(basis)
    rep = _gradient_report(gradnorm, alg.gtol, ΔE, cond(Symmetric(S)))
    return _solution_from_basis(basis, ctx, [StageResult(alg, step_hist, rep)])
end

# Fold a Pipeline left-to-right: each stage's own `_solve` runs unmodified,
# warm-started from the previous stage's Solution; every stage's StageResult
# is kept, and the final Solution carries the last stage's basis/coefficients.
function _solve(
        terms, masses, p::Pipeline;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
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
