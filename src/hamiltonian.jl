using FewBodyHamiltonians
using LinearAlgebra

function _compute_overlap_element(bra::GaussianBase, ket::GaussianBase)
    return _compute_matrix_element(bra, ket)
end

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

function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    n = length(basis.functions)
    T = eltype(parent(first(basis.functions).A))
    H = zeros(T, n, n)
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

function default_scale(masses::Vector{<:Real})
    μ = minimum(masses[masses .< 1.0e10])
    return 1 / sqrt(μ)
end

"""
    solve_ECG(operators, n=50; kwargs...) -> SolverResults

Build an ECG basis of `n` `Rank0Gaussian` functions using **stochastic greedy
search** and return the ground-state energy.

Candidate Gaussians are generated from a quasi-random sequence (Halton by
default).  Each candidate is accepted if it is linearly independent from the
existing basis (normalised overlap < `threshold`) and does not make the overlap
matrix ill-conditioned.  The ground-state energy after each accepted function
is stored in `SolverResults.energies`.

# Arguments
- `operators` : `Vector{<:Operator}` — kinetic + Coulomb operators (see [`KineticOperator`](@ref), [`CoulombOperator`](@ref)).
- `n`         : target number of basis functions (default 50).

# Keyword arguments
| keyword         | default        | description |
|:----------------|:---------------|:------------|
| `sampler`       | `HaltonSample()` | QuasiMonteCarlo sampler for generating candidates |
| `method`        | `:quasirandom` | `:quasirandom` or `:random` |
| `scale`         | `0.2`          | characteristic Gaussian width (a.u.) |
| `threshold`     | `0.95`         | normalised overlap above which a candidate is rejected |
| `max_attempts`  | `10n`          | maximum number of candidate draws |
| `max_condition` | `1e12`         | maximum condition number of the overlap matrix |
| `verbose`       | `true`         | print per-step info messages |

# Example

```julia
using FewBodyECG
masses = [1.0e15, 1.0]   # hydrogen atom (fixed nucleus)
Λmat = Λ(masses)
_, U = _jacobi_transform(masses)
w = U' * [1.0, -1.0]
ops = Operator[KineticOperator(Λmat); CoulombOperator(-1.0, w)]
sr = solve_ECG(ops, 30; scale=1.0, verbose=false)
println(sr.ground_state)   # ≈ -0.5 Ha
```
"""
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

    # Pre-allocate full matrices; fill one row/column per accepted function.
    # S_full[j,j] doubles as a cache of self-overlaps for the independence check.
    H_full = zeros(Float64, n, n)
    S_full = zeros(Float64, n, n)

    n_accepted = 0
    n_rejected = 0
    attempt = 0

    while n_accepted < n && attempt < max_attempts
        attempt += 1

        bij = generate_bij(method, attempt, n_pairs, b₁; qmc_sampler = sampler)
        A = _generate_A_matrix(bij, w_list)
        s = generate_shift(method, attempt, d, scale; qmc_sampler = sampler)
        candidate = Rank0Gaussian(A, s)

        k  = n_accepted   # current accepted count
        ki = k + 1        # index if this candidate is accepted

        # Compute new diagonal overlap (needed for independence check).
        s_diag = _compute_matrix_element(candidate, candidate)

        # Compute new overlap column; check linear independence in the same pass.
        s_col = Vector{Float64}(undef, k)
        for j in 1:k
            s_col[j] = _compute_matrix_element(candidate, basis_fns[j])
        end
        if k > 0
            # S_full[j,j] holds the self-overlap of the j-th accepted function.
            max_norm = maximum(j -> abs(s_col[j]) / sqrt(s_diag * S_full[j, j]), 1:k)
            if max_norm > threshold
                n_rejected += 1
                verbose && @warn "Rejected basis function $attempt (overlap > $threshold)"
                continue
            end
        end

        # Compute new Hamiltonian column.
        h_col = Vector{Float64}(undef, k)
        for j in 1:k
            h_col[j] = sum(_compute_matrix_element(candidate, basis_fns[j], op) for op in operators)
        end
        h_diag = sum(_compute_matrix_element(candidate, candidate, op) for op in operators)

        # Reject before touching the matrices if any element is non-finite.
        if !isfinite(s_diag) || !isfinite(h_diag) ||
                (k > 0 && (!all(isfinite, s_col) || !all(isfinite, h_col)))
            @warn "NaN/Inf in matrix elements at step $ki, rejecting basis function"
            n_rejected += 1
            continue
        end

        # Fill the new row/column into the pre-allocated matrices.
        for j in 1:k
            S_full[ki, j] = s_col[j]
            S_full[j, ki] = s_col[j]
            H_full[ki, j] = h_col[j]
            H_full[j, ki] = h_col[j]
        end
        S_full[ki, ki] = s_diag
        H_full[ki, ki] = h_diag

        # Extract ki×ki submatrices (copy needed: eigensolver may alias internally).
        H_k = H_full[1:ki, 1:ki]
        S_k = S_full[1:ki, 1:ki]

        # Condition check on the overlap submatrix.
        cond_S = cond(Symmetric(S_k))
        if cond_S > max_condition
            @warn "Overlap poorly conditioned (κ=$cond_S) at step $ki, rejecting"
            n_rejected += 1
            continue
        end

        local λs, Us
        try
            λs, Us = solve_generalized_eigenproblem(H_k, S_k; max_condition)
        catch e
            @warn "Failed at step $ki: $e"
            n_rejected += 1
            continue
        end

        E0 = minimum(λs)

        # Variational principle: adding any linearly independent function to the
        # basis cannot raise the ground-state energy.  If it does, the candidate
        # is numerically near-degenerate with the existing basis (not caught by
        # the overlap / condition-number checks above), so reject it.
        if n_accepted > 0 && E0 > E_hist[end] + 1.0e-10
            @warn "Candidate raises energy at step $ki, rejecting" ΔE = E0 - E_hist[end]
            n_rejected += 1
            continue
        end

        push!(basis_fns, candidate)
        n_accepted += 1
        push!(E_hist, E0)
        push!(vecs_list, Us)
        verbose && @info "Step $n_accepted" E₀ = E0 attempts = attempt rejected = n_rejected
    end

    if n_accepted < n
        @warn "Only generated $n_accepted of $n requested basis functions" rejected = n_rejected
    end

    Emin = last(E_hist)
    @info "Optimization complete" E₀ = Emin n_basis = n_accepted
    return SolverResults(basis_fns, n_accepted, operators, method, sampler, b₁, Emin, E_hist, vecs_list, E_hist)
end
