# =============================================================================
# Competitive-selection stochastic variational solver.
#
# Suzuki & Varga, Sect. 4.2.5 ("Trial and error search", steps s1-s4): at each
# step draw `n_candidates` random Gaussians, score every one cheaply against the
# current basis, and commit only the best.  Scoring uses the O(k²) arrowhead
# update (`score_candidate`) instead of a full O(k³) eigensolve, so looking at
# many candidates per step is affordable — this is the difference between the
# slow random-basis curve (Fig 4.1) and the fast competitive curve (Fig 4.2).
# =============================================================================

# Draw a single Rank0Gaussian candidate from the (quasi-)random stream at index `idx`.
function _draw_rank0_candidate(method, idx, n_pairs, d, b₁, scale, w_list, sampler)
    bij = generate_bij(method, idx, n_pairs, b₁; qmc_sampler = sampler)
    A = _generate_A_matrix(bij, w_list)
    s = generate_shift(method, idx, d, scale; qmc_sampler = sampler)
    return Rank0Gaussian(A, s)
end

# Overlap/Hamiltonian columns of `candidate` against the current basis, plus its
# self-overlaps.  Returns `nothing` if any element is non-finite.
function _candidate_columns(candidate, basis_fns, operators)
    k = length(basis_fns)
    s_col = Vector{Float64}(undef, k)
    h_col = Vector{Float64}(undef, k)
    for j in 1:k
        s_col[j] = _compute_matrix_element(candidate, basis_fns[j])
        h_col[j] = sum(_compute_matrix_element(candidate, basis_fns[j], op) for op in operators)
    end
    s_diag = _compute_matrix_element(candidate, candidate)
    h_diag = sum(_compute_matrix_element(candidate, candidate, op) for op in operators)
    if !isfinite(s_diag) || !isfinite(h_diag) ||
            (k > 0 && (!all(isfinite, s_col) || !all(isfinite, h_col)))
        return nothing
    end
    return (s_col, h_col, s_diag, h_diag)
end

"""
    solve_ECG_competitive(operators, n=50; kwargs...) -> SolverResults

Build an ECG basis of `n` `Rank0Gaussian` functions by **competitive selection**
(Suzuki & Varga, Sect. 4.2.5).  At each of the `n` steps, `n_candidates`
quasi-random Gaussians are drawn, each scored in O(k²) by the incremental
arrowhead eigensolver, and the one giving the lowest target-state energy is
committed.  The energy after each committed function is recorded in
`SolverResults.energies`.

Because candidate scoring never performs a full eigensolve, this evaluates many
more candidates per basis function than [`solve_ECG`](@ref) at lower cost, and
converges in far fewer basis functions.

# Keyword arguments
| keyword            | default          | description |
|:-------------------|:-----------------|:------------|
| `n_candidates`     | `10`             | candidates drawn and scored per step |
| `sampler`          | `HaltonSample()` | QuasiMonteCarlo sampler |
| `method`           | `:quasirandom`   | `:quasirandom` or `:random` |
| `scale`            | `0.2`            | characteristic Gaussian width (a.u.) |
| `indep_tol`        | `1e-4`           | reject candidates whose Cholesky residual ρ² is below this fraction of their norm (keeps the overlap well-conditioned) |
| `verbose`          | `true`           | print per-step info |
| `state`            | `1`              | target eigenvalue index (1 = ground state) |

# Example
```julia
ops = Operators([1e15, 1.0], [+1, -1]); ops += "Kinetic"; ops += "Coulomb"
sr = solve_ECG_competitive(ops, 30; n_candidates=20, scale=1.0, verbose=false)
sr.ground_state    # ≈ -0.5 Ha
```
"""
function solve_ECG_competitive(
        operators::Vector{<:FewBodyHamiltonians.Operator},
        n::Int = 50;
        n_candidates::Int = 10,
        sampler = HaltonSample(),
        method::Symbol = :quasirandom,
        scale::Real = 0.2,
        indep_tol::Real = 1.0e-4,
        verbose::Bool = true,
        state::Int = 1
    )
    state >= 1 || throw(ArgumentError("state must be >= 1, got $state"))
    n_candidates >= 1 || throw(ArgumentError("n_candidates must be >= 1"))

    b₁ = float(scale)
    w_list = [op.w for op in operators if op isa Union{CoulombOperator, GaussianOperator}]
    n_pairs = length(w_list)
    d = length(w_list[1])

    eig = SVMEigen()
    basis_fns = Rank0Gaussian[]
    E_hist = Float64[]
    draw = 0

    for k in 1:n
        best_E = Inf
        best_cand = nothing
        best_cols = nothing

        for _ in 1:n_candidates
            draw += 1
            cand = _draw_rank0_candidate(method, draw, n_pairs, d, b₁, scale, w_list, sampler)
            cols = _candidate_columns(cand, basis_fns, operators)
            cols === nothing && continue
            s_col, h_col, s_diag, h_diag = cols
            E = score_candidate(eig, s_col, h_col, s_diag, h_diag;
                                state = state, min_resid_ratio = indep_tol)
            E === nothing && continue
            if E < best_E
                best_E = E
                best_cand = cand
                best_cols = cols
            end
        end

        if best_cand === nothing
            verbose && @warn "Step $k: no admissible candidate among $n_candidates draws"
            continue
        end

        commit_candidate!(eig, best_cols...)
        push!(basis_fns, best_cand)
        target = min(state, length(eig.ε))
        push!(E_hist, eig.ε[target])
        verbose && @info "Step $(length(basis_fns))" E = eig.ε[target] candidates = n_candidates
    end

    n_accepted = length(basis_fns)
    if n_accepted == 0
        error("Competitive selection produced no basis functions")
    end
    Emin = last(E_hist)
    @info "Competitive optimisation complete" E₀ = Emin n_basis = n_accepted state = state
    return SolverResults(
        basis_fns, n_accepted, operators, method, sampler, b₁,
        Emin, state, E_hist, [coefficients(eig)], E_hist
    )
end

solve_ECG_competitive(ops::Operators, n::Int = 50; kwargs...) =
    solve_ECG_competitive(ops.terms, n; kwargs...)
