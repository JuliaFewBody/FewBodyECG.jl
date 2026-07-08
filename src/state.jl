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
