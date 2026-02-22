using FewBodyHamiltonians
using LinearAlgebra

function _compute_overlap_element(bra::GaussianBase, ket::GaussianBase)
    return _compute_matrix_element(bra, ket)
end

function build_overlap_matrix(basis::BasisSet{<:GaussianBase})
    n = length(basis.functions)
    S = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_overlap_element(basis.functions[i], basis.functions[j])
        S[i, j] = val
        S[j, i] = val
    end
    return S
end

function _build_operator_matrix(basis::BasisSet{<:GaussianBase}, op::FewBodyHamiltonians.Operator)
    n = length(basis.functions)
    H = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_matrix_element(basis.functions[i], basis.functions[j], op)
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
        @warn "Overlap matrix poorly conditioned (κ=$cond_S), adding regularization"
        if regularization == 0.0
            regularization = maximum(abs.(diag(S_sym))) * 1.0e-10
        end
    end

    if regularization > 0
        S_sym = S_sym + regularization * I
    end

    if !isposdef(S_sym)
        @warn "Overlap matrix not positive definite, adding regularization"
        ε = maximum(abs.(diag(S_sym))) * 1.0e-8
        S_sym = S_sym + ε * I

        if !isposdef(S_sym)
            error("Overlap matrix not positive definite even after regularization")
        end
    end

    # Cholesky decomposition with error handling
    local F
    try
        F = cholesky(S_sym)
    catch e
        @error "Cholesky decomposition failed" exception = e
        @error "Overlap matrix info" condition = cond(S_sym) min_eigval = minimum(eigvals(S_sym))
        rethrow(e)
    end

    L = F.L

    # Transform to standard eigenvalue problem
    A = (L \ Matrix(H_sym)) / L'
    A_sym = Symmetric((A + A') / 2)

    # Check for NaN/Inf after transformation
    if any(!isfinite, A_sym)
        error("Transformed matrix contains NaN or Inf after Cholesky transformation")
    end

    # Solve standard eigenvalue problem
    local evals, evecs
    try
        evals, evecs = eigen(A_sym)
    catch e
        @error "Eigenvalue decomposition failed" exception = e
        @error "Transformed matrix info" condition = cond(A_sym)
        rethrow(e)
    end

    # Transform eigenvectors back
    vecs = L' \ evecs

    # Ensure real
    evals_real = real.(evals)
    vecs_real = real.(vecs)

    return evals_real, vecs_real
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

function default_scale(masses::Vector{<:Real})
    μ = minimum(masses[masses .< 1.0e10])
    return 1 / sqrt(μ)
end

function solve_ECG(
        operators::Vector{<:FewBodyHamiltonians.Operator},
        n::Int = 50;
        sampler = HaltonSample(),
        method::Symbol = :quasirandom,
        scale::Real = 0.2,
        threshold::Real = 0.95,
        max_attempts::Int = 10 * n,
        max_condition::Real = 1.0e12,
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

        # Check linear independence
        if !isempty(basis_fns)
            existing_basis = BasisSet{Rank0Gaussian}(basis_fns)
            if !is_linearly_independent(candidate, existing_basis; threshold = threshold)
                n_rejected += 1
                verbose && @warn "Rejected basis function $attempt (overlap > $threshold)"
                continue
            end
        end

        push!(basis_fns, candidate)

        local H, S, λs, Us
        try
            basis = BasisSet{Rank0Gaussian}(basis_fns)
            H = build_hamiltonian_matrix(basis, operators)
            S = build_overlap_matrix(basis)

            # Check for NaN/Inf BEFORE eigensolve
            if any(!isfinite, H)
                @warn "Hamiltonian contains NaN/Inf at step $(n_accepted + 1), rejecting basis function"
                pop!(basis_fns)
                n_rejected += 1
                continue
            end

            if any(!isfinite, S)
                @warn "Overlap contains NaN/Inf at step $(n_accepted + 1), rejecting basis function"
                pop!(basis_fns)
                n_rejected += 1
                continue
            end

            # Check condition number
            cond_S = cond(Symmetric(S))
            if cond_S > max_condition
                @warn "Overlap poorly conditioned (κ=$cond_S) at step $(n_accepted + 1), rejecting"
                pop!(basis_fns)
                n_rejected += 1
                continue
            end

            # Try to solve
            λs, Us = solve_generalized_eigenproblem(H, S; max_condition)

        catch e
            @warn "Failed at step $(n_accepted + 1): $e"
            pop!(basis_fns)  # Remove problematic function
            n_rejected += 1
            continue
        end

        n_accepted += 1
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
