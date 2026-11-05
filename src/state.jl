# Shared incremental state of the stochastic solver family.  Caching S and H
# (a few k² floats) makes Refine's rebuilds and warm starts cheap: no matrix
# element is ever recomputed.
mutable struct BasisState{G <: GaussianBase}
    basis::Vector{G}
    eig::SVMEigen
    S::Matrix{Float64}
    H::Matrix{Float64}
    E_hist::Vector{Float64}
    draw::Int
    rank::Int
    symmetry::Union{Nothing, PermutationSymmetry}
end

function BasisState(::Type{G}; rank = G <: Rank1Gaussian ? 1 : 0, symmetry = nothing) where {G <: GaussianBase}
    (G <: ProjectedGaussian) == (symmetry !== nothing) || throw(ArgumentError("projected basis states require their symmetry"))
    return BasisState{G}(
        G[], SVMEigen(), Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
        Float64[], 0, rank, symmetry
    )
end
BasisState() = BasisState(Rank0Gaussian)

nfuns(st::BasisState) = length(st.basis)

# Overlap/Hamiltonian columns of `cand` against `basis`; `nothing` if any
# element is non-finite.
function _candidate_columns(cand, basis, operators, invariant::Bool = false)
    s_diag = _compute_matrix_element(cand, cand)
    if cand isa ProjectedGaussian
        s_diag > 1.0e-12 * _compute_matrix_element(cand.orbital, cand.orbital) || return nothing
    end
    k = length(basis)
    s_col = Vector{Float64}(undef, k)
    h_col = Vector{Float64}(undef, k)
    for j in 1:k
        s_col[j] = _compute_matrix_element(cand, basis[j])
        h_col[j] = _hamiltonian_element(cand, basis[j], operators, invariant)
    end
    h_diag = _hamiltonian_element(cand, cand, operators, invariant)
    ok = isfinite(s_diag) && isfinite(h_diag) &&
        (k == 0 || (all(isfinite, s_col) && all(isfinite, h_col)))
    return ok ? (s_col, h_col, s_diag, h_diag) : nothing
end

# Draw the next quasi-random candidate; advances the stream counter.
# Candidates are unshifted to preserve L=0, or L=1 through the Rank1 prefactor.
function _draw_candidate!(st::BasisState{G}, scale::Float64, sampler, widths, w_list, d) where {G}
    st.draw += 1
    bij = generate_bij(:quasirandom, st.draw, length(w_list), scale; qmc_sampler = sampler, widths)
    A = _generate_A_matrix(bij, w_list)
    g = _candidate_orbital(st.rank == 0 ? Rank0Gaussian : Rank1Gaussian, A, st.draw, sampler, length(w_list), d)
    return st.symmetry === nothing ? g : ProjectedGaussian(g, st.symmetry)
end

_candidate_orbital(::Type{<:Rank0Gaussian}, A, _, _, _, d) = Rank0Gaussian(A, zeros(d, 3))

function _candidate_orbital(::Type{<:Rank1Gaussian}, A, draw, sampler, n_terms, d)
    # Extra QMC axes keep the global vector independent of the pair widths.
    a = d == 1 ? [1.0] : 2 .* _qmc_point(draw, n_terms + d; sampler)[(n_terms + 1):end] .- 1
    norm(a) > 0 || (a[1] = 1)
    return Rank1Gaussian(A, a / norm(a), zeros(d, 3))
end

# Append `cand` (whose columns are `cols`): update eigensolver + S/H caches.
function commit!(st::BasisState{G}, cand::G, cols) where {G}
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
function BasisState(basis::Vector{G}, operators) where {G <: GaussianBase}
    bs = BasisSet(basis)
    symmetry = _basis_symmetry(bs)
    any(g -> g isa ProjectedGaussian, basis) && symmetry === nothing && throw(ArgumentError("all initial basis functions must share the same symmetry"))
    st = BasisState(G; rank = _stochastic_rank(bs), symmetry)
    invariant = _hamiltonian_invariant(operators, symmetry)
    for g in basis
        cols = _candidate_columns(g, st.basis, operators, invariant)
        cols === nothing && error("invalid or numerically null function while rebuilding basis state")
        commit!(st, g, cols) === nothing &&
            error("linearly dependent basis while rebuilding state")
    end
    return st
end

# The (k−1)-function state with function `i` removed, re-committed from the
# cached S/H columns — no matrix-element recomputation.  O(k³), small constant.
function rebuild_without(st::BasisState{G}, i::Integer) where {G}
    k = nfuns(st)
    idx = setdiff(1:k, i)
    r = BasisState(G; rank = st.rank, symmetry = st.symmetry)
    for (m, j) in enumerate(idx)
        prev = idx[1:(m - 1)]
        cols = (st.S[prev, j], st.H[prev, j], st.S[j, j], st.H[j, j])
        commit!(r, st.basis[j], cols) === nothing &&
            error("linear dependence while rebuilding without function $i")
    end
    r.draw = st.draw
    return r
end

# Warm start: rebuild a BasisState from a Solution's basis.  The draw counter
# resumes after the points `sol` already consumed, so no candidate is redrawn.
function _solution_basis_state(sol::Solution, operators)
    fns = getfield(sol, :basis).functions
    rank = _stochastic_rank(sol.basis)
    G = _basis_symmetry(sol.basis) === nothing ? (rank == 0 ? Rank0Gaussian : Rank1Gaussian) : ProjectedGaussian
    st = BasisState(G[g for g in fns], operators)
    st.draw = getfield(sol, :draws)
    return st
end

function _stochastic_rank(basis::BasisSet)
    all(g -> _unprojected(g) isa Rank0Gaussian, basis) && return 0
    all(g -> _unprojected(g) isa Rank1Gaussian, basis) && return 1
    throw(ArgumentError("stochastic methods require a uniformly Rank0Gaussian or Rank1Gaussian basis"))
end
