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
        verbose::Bool = true,
        state::Int = 1
    )
    state >= 1 || throw(ArgumentError("state must be >= 1, got $state"))

    b₁ = float(scale)
    basis_fns = Rank0Gaussian[]
    E_hist = Float64[]
    vecs_list = Any[]

    w_list = [op.w for op in operators if op isa Union{CoulombOperator, GaussianOperator}]
    n_pairs = length(w_list)
    d = length(w_list[1])

    # Pre-allocate full matrices; fill one row/column per accepted function.
    # S_full[j,j] doubles as a cache of self-overlaps for the independence check.
    H_full = zeros(Float64, n, n)
    S_full = zeros(Float64, n, n)

    n_accepted = 0
    n_rejected = 0
    attempt = 0
    E_target_last = Inf   # last accepted energy of the target state specifically

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
                if verbose == true @warn "Overlap poorly conditioned (κ=$cond_S) at step $ki, rejecting" end
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

        # Target eigenvalue: use the requested state when available, otherwise
        # fall back to the highest available eigenvalue during early build-up.
        target_idx = min(state, length(λs))
        E0 = λs[target_idx]

        # Variational principle: the k-th eigenvalue is an upper bound to the
        # k-th exact energy, so adding a linearly independent function cannot
        # raise it.  Only compare against a previous value of the SAME eigenvalue
        # (target_idx == state) to avoid spurious rejections during build-up
        # when transitioning from tracking a lower eigenvalue to the target one.
        if target_idx == state && isfinite(E_target_last) && E0 > E_target_last + 1.0e-10
            @warn "Candidate raises energy at step $ki, rejecting" ΔE = E0 - E_target_last
            n_rejected += 1
            continue
        end

        push!(basis_fns, candidate)
        n_accepted += 1
        push!(E_hist, E0)
        push!(vecs_list, Us)
        if target_idx == state
            E_target_last = E0
        end
        verbose && @info "Step $n_accepted" E₀ = E0 attempts = attempt rejected = n_rejected
    end

    if n_accepted < n
        @warn "Only generated $n_accepted of $n requested basis functions" rejected = n_rejected
    end

    Emin = last(E_hist)
    @info "Optimization complete" E₀ = Emin n_basis = n_accepted state = state
    return SolverResults(basis_fns, n_accepted, operators, method, sampler, b₁, Emin, state, E_hist, vecs_list, E_hist)
end

"""
    Operators

Accumulates `KineticOperator` and `CoulombOperator` terms to build a Hamiltonian.
Handles Jacobi coordinate transforms internally so that callers can work
with physical particle indices rather than Jacobi-frame weight vectors.

# Constructors

    Operators()                    # system-unaware; add pre-built operators with `+=`
    Operators(masses)              # system-aware; enables string/index shorthand
    Operators(masses, charges)     # fully automatic; enables `ops += "Coulomb"` shorthand

# System-aware interface

Particle indices follow the original ordering of `masses`.  All Jacobi
transforms are computed internally.

```julia
ops = Operators([m₁, m₂, m₃])
ops += "Kinetic"
ops += ("Coulomb", 1, 2, +1.0)   # pair (1,2) with coupling coefficient +1.0
ops += ("Coulomb", 1, 3, -1.0)
```

When charges are also supplied, the fully-automatic shorthand `ops += "Coulomb"`
adds all ``N(N-1)/2`` pairwise terms with coefficients ``q_i q_j``:

```julia
# Helium atom: nucleus (Z=2), two electrons
ops = Operators([1e15, 1.0, 1.0], [+2, -1, -1])
ops += "Kinetic"
ops += "Coulomb"   # adds (1,2)→-2, (1,3)→-2, (2,3)→+1 automatically
```

# System-unaware interface (pre-built operators)

```julia
ops = Operators()
ops += KineticOperator(Λmat)
ops += CoulombOperator(-1.0, w)
```

Both interfaces can be mixed freely.  Pass `ops` directly to [`solve_ECG`](@ref),
[`solve_ECG_variational`](@ref), [`solve_ECG_sequential`](@ref), or
`build_hamiltonian_matrix`.  Use [`coulomb_weights`](@ref) to retrieve
the Jacobi-frame weight vectors for manual basis construction.
"""
mutable struct Operators
    terms::Vector{FewBodyHamiltonians.Operator}
    masses::Union{Nothing, Vector{Float64}}
    charges::Union{Nothing, Vector{Float64}}
    _U::Union{Nothing, Matrix{Float64}}
end

Operators() = Operators(FewBodyHamiltonians.Operator[], nothing, nothing, nothing)

function Operators(masses::Vector{<:Real})
    _, U = _jacobi_transform(Float64.(masses))
    return Operators(FewBodyHamiltonians.Operator[], Float64.(masses), nothing, U)
end

"""
    Operators(masses, charges)

Create an `Operators` for a system with given particle `masses` and `charges`.
Enables the fully automatic shorthand `ops += "Coulomb"`, which adds all
``N(N-1)/2`` pairwise Coulomb interactions with coefficients ``q_i q_j``.

```julia
# H⁻: proton (charge +1) + two electrons (charge -1)
ops = Operators([1e15, 1.0, 1.0], [+1, -1, -1])
ops += "Kinetic"
ops += "Coulomb"   # adds (1,2)→-1, (1,3)→-1, (2,3)→+1 automatically
```
"""
function Operators(masses::Vector{<:Real}, charges::Vector{<:Real})
    length(masses) == length(charges) ||
        throw(ArgumentError("masses and charges must have the same length"))
    _, U = _jacobi_transform(Float64.(masses))
    return Operators(FewBodyHamiltonians.Operator[], Float64.(masses), Float64.(charges), U)
end

function Base.:+(ops::Operators, op::FewBodyHamiltonians.Operator)
    push!(ops.terms, op)
    return ops
end

function Base.:+(ops::Operators, name::AbstractString)
    if name == "Kinetic"
        ops.masses !== nothing ||
            throw(ArgumentError("\"Kinetic\" requires Operators(masses)."))
        push!(ops.terms, KineticOperator(ops.masses))
    elseif name == "Coulomb"
        ops.charges !== nothing ||
            throw(ArgumentError(
                "\"Coulomb\" without indices requires Operators(masses, charges). " *
                "Use ops += \"Coulomb\", i, j, coeff for explicit pairs."
            ))
        N = length(ops.masses)
        for i in 1:N, j in (i + 1):N
            e_ij = zeros(Float64, N)
            e_ij[i] = 1.0
            e_ij[j] = -1.0
            w = ops._U' * e_ij
            push!(ops.terms, CoulombOperator(ops.charges[i] * ops.charges[j], w))
        end
    else
        throw(ArgumentError(
            "Unknown operator \"$name\". Supported: \"Kinetic\", \"Coulomb\"."
        ))
    end
    return ops
end

function Base.:+(ops::Operators, term::Tuple{<:AbstractString, <:Integer, <:Integer, <:Real, <:Real})
    name, i, j, coeff, γ = term
    name == "Gaussian" ||
        throw(ArgumentError("Unknown operator \"$name\". Supported: \"Gaussian\"."))
    ops.masses !== nothing ||
        throw(ArgumentError("\"Gaussian\" requires Operators(masses)."))
    i != j || throw(ArgumentError("Particle indices must be distinct, got i = j = $i."))
    N = length(ops.masses)
    1 ≤ i ≤ N || throw(ArgumentError("Particle index i=$i out of range [1, $N]."))
    1 ≤ j ≤ N || throw(ArgumentError("Particle index j=$j out of range [1, $N]."))
    Float64(γ) > 0 || throw(ArgumentError("γ must be positive, got γ = $γ."))
    e_ij = zeros(Float64, N)
    e_ij[i] = 1.0
    e_ij[j] = -1.0
    w = ops._U' * e_ij
    push!(ops.terms, GaussianOperator(Float64(coeff), Float64(γ), w))
    return ops
end

function Base.:+(ops::Operators, term::Tuple{<:AbstractString, <:Integer, <:Integer, <:Real})
    name, i, j, coeff = term
    name == "Coulomb" ||
        throw(ArgumentError("Unknown operator \"$name\". Supported: \"Coulomb\"."))
    ops.masses !== nothing ||
        throw(ArgumentError(
            "String-based \"Coulomb\" requires Operators(masses)."
        ))
    i != j || throw(ArgumentError("Particle indices must be distinct, got i = j = $i."))
    N = length(ops.masses)
    1 ≤ i ≤ N || throw(ArgumentError("Particle index i=$i out of range [1, $N]."))
    1 ≤ j ≤ N || throw(ArgumentError("Particle index j=$j out of range [1, $N]."))
    e_ij = zeros(Float64, N)
    e_ij[i] = 1.0
    e_ij[j] = -1.0
    w = ops._U' * e_ij
    push!(ops.terms, CoulombOperator(Float64(coeff), w))
    return ops
end

Base.length(ops::Operators) = length(ops.terms)
Base.iterate(ops::Operators) = iterate(ops.terms)
Base.iterate(ops::Operators, state) = iterate(ops.terms, state)
Base.getindex(ops::Operators, i::Int) = ops.terms[i]
Base.eltype(::Type{Operators}) = FewBodyHamiltonians.Operator

function Base.show(io::IO, ops::Operators)
    n = length(ops.terms)
    if ops.masses !== nothing && ops.charges !== nothing
        header = "Operators(masses=$(round.(ops.masses; sigdigits=3)), charges=$(ops.charges))"
    elseif ops.masses !== nothing
        header = "Operators($(length(ops.masses))-body)"
    else
        header = "Operators"
    end
    println(io, "$header with $n term$(n == 1 ? "" : "s"):")
    for op in ops.terms
        if op isa KineticOperator
            println(io, "  + Kinetic")
        elseif op isa CoulombOperator
            println(io, "  + $(op.coefficient) × Coulomb(w = $(round.(op.w; digits = 3)))")
        elseif op isa GaussianOperator
            println(io, "  + $(op.coefficient) × Gaussian(γ = $(round(op.γ; digits = 3)), w = $(round.(op.w; digits = 3)))")
        else
            println(io, "  + $(typeof(op))")
        end
    end
end

"""
    coulomb_weights(ops::Operators) -> Vector{Vector{Float64}}

Return the Jacobi-frame weight vectors for every `CoulombOperator` in `ops`,
in the order they were added. Useful for manual basis construction:

```julia
w_jac = coulomb_weights(ops)
A = _generate_A_matrix(bij, w_jac)
```
"""
coulomb_weights(ops::Operators) = [op.w for op in ops.terms if op isa CoulombOperator]

function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, ops::Operators)
    return build_hamiltonian_matrix(basis, ops.terms)
end

function solve_ECG(ops::Operators, n::Int = 50; kwargs...)
    return solve_ECG(ops.terms, n; kwargs...)
end

function solve_ECG_variational(ops::Operators, n::Int = 50; kwargs...)
    return solve_ECG_variational(ops.terms, n; kwargs...)
end

function solve_ECG_sequential(ops::Operators, n::Int = 50; kwargs...)
    return solve_ECG_sequential(ops.terms, n; kwargs...)
end
