using FewBodyHamiltonians
using LinearAlgebra

function _compute_overlap_element(bra::GaussianBase, ket::GaussianBase)
    return _compute_matrix_element(bra, ket)
end

"""
    build_overlap_matrix(basis)

Return the ECG overlap matrix `S` with entries `<g_i|g_j>` for a `BasisSet`.
"""
function build_overlap_matrix(basis::BasisSet{<:GaussianBase})
    n = length(basis.functions)
    T = eltype(parent(first(basis.functions).A))
    S = Matrix{T}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_overlap_element(basis.functions[i], basis.functions[j])
        S[i, j] = val
        S[j, i] = val
    end
    return S
end

function _build_operator_matrix(basis::BasisSet{<:GaussianBase}, op::FewBodyHamiltonians.Operator)
    n = length(basis.functions)
    T = eltype(parent(first(basis.functions).A))
    H = Matrix{T}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_matrix_element(basis.functions[i], basis.functions[j], op)
        H[i, j] = val
        H[j, i] = val
    end
    return H
end

"""
    build_hamiltonian_matrix(basis, operators)

Return the Hamiltonian matrix assembled from all operator matrix elements over
`basis`. `operators` may be an `Operators` builder or a vector of operator
terms.
"""
function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    n = length(basis.functions)
    T = eltype(parent(first(basis.functions).A))
    H = zeros(T, n, n)
    for op in operators
        H .+= _build_operator_matrix(basis, op)
    end
    return H
end

"""
    solve_generalized_eigenproblem(H, S; max_condition=1e12, regularization=0)

Solve the symmetric generalized eigenproblem `H*c = E*S*c`, returning
eigenvalues and `S`-orthonormal eigenvectors.
"""
function solve_generalized_eigenproblem(
        H::AbstractMatrix{<:Real},
        S::AbstractMatrix{<:Real};
        max_condition::Real = 1.0e12,
        regularization::Real = 0.0
    )

    if any(!isfinite, H)
        error("Hamiltonian matrix H contains NaN or Inf values")
    end
    if any(!isfinite, S)
        error("Overlap matrix S contains NaN or Inf values")
    end

    H_sym = Symmetric((H + H') / 2)
    S_sym = Symmetric((S + S') / 2)

    cond_S = cond(S_sym)
    if cond_S > max_condition
        if regularization == 0.0
            regularization = maximum(abs.(diag(S_sym))) * 1.0e-10
        end
    end

    if regularization > 0
        S_sym = Symmetric(Matrix(S_sym) + regularization * I)
    end

    if !isposdef(S_sym)
        @warn "Overlap matrix not positive definite, adding regularization"
        ε = maximum(abs.(diag(S_sym))) * 1.0e-8
        S_sym = Symmetric(Matrix(S_sym) + ε * I)

        if !isposdef(S_sym)
            error("Overlap matrix not positive definite even after regularization")
        end
    end

    # Solve the generalised symmetric eigenvalue problem H c = λ S c via
    # LAPACK's divide-and-conquer driver (dsygvd).  This is more reliable than
    # manually factorising S and back-transforming, and returns eigenvectors
    # normalised so that vᵀ S v = I.
    local evals, vecs
    try
        F = eigen(H_sym, S_sym)
        evals = real.(F.values)
        vecs = real.(F.vectors)
    catch e
        @error "Generalised eigenvalue decomposition failed" exception = e
        rethrow(e)
    end

    if any(!isfinite, evals) || any(!isfinite, vecs)
        error("Eigenvalues or eigenvectors contain NaN or Inf")
    end

    return evals, vecs
end

function normalized_overlap(A::GaussianBase, B::GaussianBase)
    overlap_12 = _compute_matrix_element(A, B)
    overlap_11 = _compute_matrix_element(A, A)
    overlap_22 = _compute_matrix_element(B, B)

    norm = sqrt(overlap_11 * overlap_22)

    if norm < eps(Float64)
        return 0.0
    end

    return abs(overlap_12) / norm
end

function is_linearly_independent(
        new_gaussian::GaussianBase,
        existing_basis::BasisSet{<:GaussianBase};
        threshold::Real = 0.95
    )

    0.0 < threshold < 1.0 || throw(ArgumentError("threshold must be in (0,1)"))

    for g_existing in existing_basis.functions
        overlap_norm = normalized_overlap(new_gaussian, g_existing)

        if overlap_norm > threshold
            return false
        end
    end

    return true
end

"""
    default_scale(masses)

Return the default Gaussian length scale inferred from the lightest finite
particle mass in atomic units.
"""
function default_scale(masses::Vector{<:Real})
    μ = minimum(masses[masses .< 1.0e10])
    return 1 / sqrt(μ)
end
