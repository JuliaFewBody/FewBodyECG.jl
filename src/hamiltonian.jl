using FewBodyHamiltonians
using LinearAlgebra

function _compute_overlap_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    return _compute_matrix_element(bra, ket)
end

function build_overlap_matrix(basis::BasisSet{<:GaussianBase})
    n = length(basis.functions)
    S = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_overlap_element(basis.functions[i]::Rank0Gaussian, basis.functions[j]::Rank0Gaussian)
        S[i, j] = val
        S[j, i] = val
    end
    return S
end

function _build_operator_matrix(basis::BasisSet{<:GaussianBase}, op::FewBodyHamiltonians.Operator)
    n = length(basis.functions)
    H = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_matrix_element(basis.functions[i]::Rank0Gaussian, basis.functions[j]::Rank0Gaussian, op)
        H[i, j] = val
        H[j, i] = val
    end
    return H
end

function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    n = length(basis.functions)
    H = zeros(Float64, n, n)
    for op in operators
        H .+= _build_operator_matrix(basis, op)
    end
    return H
end

function solve_generalized_eigenproblem(H::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(S); check = true)
    L = F.L
    A = (L \ H) / L'
    evals, evecs = eigen(Symmetric(A))
    vecs = L' \ evecs
    return real(evals), real(vecs)
end

function diagonalize(basis::BasisSet{<:GaussianBase}, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    H = build_hamiltonian_matrix(basis, operators)
    S = build_overlap_matrix(basis)
    return solve_generalized_eigenproblem(H, S)
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

function solve_ECG(
        operators::Vector{<:FewBodyHamiltonians.Operator},
        n::Int = 50;
        sampler = HaltonSample(),
        method::Symbol = :quasirandom,
        scale::Real = 0.2,
        threshold::Real = 0.95,
        max_attempts::Int = 10 * n,
        verbose::Bool = true
    )

    b₁ = float(scale)
    basis_fns = Rank0Gaussian[]
    E_hist = Float64[]
    vecs_list = Any[]

    w_list = [op.w for op in operators if op isa CoulombOperator]
    n_pairs = length(w_list)
    d = length(w_list[1])

    n_accepted = 0
    n_rejected = 0
    attempt = 0

    while n_accepted < n && attempt < max_attempts
        attempt += 1

        bij = generate_bij(method, attempt, n_pairs, b₁; qmc_sampler = sampler)
        A = _generate_A_matrix(bij, w_list)
        s = generate_shift(method, attempt, d, scale; qmc_sampler = sampler)
        candidate = Rank0Gaussian(A, s)

        if !isempty(basis_fns)
            existing_basis = BasisSet{Rank0Gaussian}(basis_fns)
            if !is_linearly_independent(candidate, existing_basis; threshold = threshold)
                n_rejected += 1
                verbose && @warn "Rejected basis function $attempt (overlap > $threshold)"
                continue
            end
        end

        push!(basis_fns, candidate)
        n_accepted += 1

        basis = BasisSet{Rank0Gaussian}(basis_fns)
        H = build_hamiltonian_matrix(basis, operators)
        S = build_overlap_matrix(basis)

        λs, Us = solve_generalized_eigenproblem(H, S)
        E0 = minimum(λs)

        push!(E_hist, E0)
        push!(vecs_list, Us)
        verbose && @info "Step $n_accepted" E₀ = E0 attempts = attempt rejected = n_rejected
    end

    if n_accepted < n
        @warn "Only generated $n_accepted of $n requested basis functions" rejected = n_rejected
    end

    Emin = last(E_hist)
    @info "Optimization complete" E₀ = Emin n_basis = n_accepted
    return SolverResults(basis_fns, n_accepted, operators, method, sampler, b₁, Emin, E_hist, vecs_list)
end
