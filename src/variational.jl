using OptimKit
using LinearAlgebra
using QuasiMonteCarlo
using ForwardDiff

function _chol_to_params(L::AbstractMatrix)
    n = size(L, 1)
    params = Float64[]
    for j in 1:n
        for i in j:n
            push!(params, i == j ? log(L[i, j]) : L[i, j])
        end
    end
    return params
end

function _params_to_matrix(θ::AbstractVector, n::Int)
    T = eltype(θ)
    L = zeros(T, n, n)
    idx = 1
    for j in 1:n
        for i in j:n
            L[i, j] = (i == j) ? exp(θ[idx]) : θ[idx]
            idx += 1
        end
    end
    return Symmetric(L * L')
end

function _encode_basis(basis::BasisSet{<:Rank0Gaussian})
    params = Float64[]
    for g in basis.functions
        C = cholesky(Symmetric(Matrix(g.A)))
        append!(params, _chol_to_params(Matrix(C.L)))
        append!(params, Float64.(g.s))   # shift vector (unconstrained)
    end
    return params
end

# Decode a flat parameter vector back into a BasisSet{Rank0Gaussian}.
# Layout per Gaussian: [n_chol Cholesky params | n_dim shift params].
function _decode_basis(θ::AbstractVector, n_basis::Int, n_dim::Int)
    T = eltype(θ)
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_per = n_chol + n_dim
    fns = Vector{Rank0Gaussian{T, Matrix{T}, Vector{T}}}(undef, n_basis)
    for i in 1:n_basis
        start = (i - 1) * n_per + 1
        A = _params_to_matrix(θ[start:(start + n_chol - 1)], n_dim)
        s = θ[(start + n_chol):(start + n_per - 1)]
        fns[i] = Rank0Gaussian(Matrix(A), Vector(s))
    end
    return BasisSet(fns)
end

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

# Minimum generalised eigenvalue λ_min of (H, S).
# By the variational principle λ_min ≥ E₀ for any basis, so minimising
# over basis parameters converges to the exact ground-state energy.
function _energy_loss(
        θ::AbstractVector,
        n_basis::Int,
        n_dim::Int,
        operators::AbstractVector{<:FewBodyHamiltonians.Operator};
        regularization::Real = 1.0e-10
    )
    basis = _decode_basis(θ, n_basis, n_dim)
    H = build_hamiltonian_matrix(basis, operators)
    S = build_overlap_matrix(basis)
    evals, _ = solve_generalized_eigenproblem(H, S; regularization = regularization)
    return minimum(evals)
end

# Tr(S⁻¹H) = sum of all generalised eigenvalues.
# For a basis that already approximates the ground state well this provides
# a smooth surrogate for the energy, but for a random or warm-started basis
# whose upper eigenvalues are large and positive the optimizer can reach a
# degenerate near-zero minimum by spreading the Gaussians out.  Prefer
# loss_type = :energy unless you know the initial trace is already negative.
function _trace_loss(
        θ::AbstractVector,
        n_basis::Int,
        n_dim::Int,
        operators::AbstractVector{<:FewBodyHamiltonians.Operator};
        regularization::Real = 1.0e-10
    )
    basis = _decode_basis(θ, n_basis, n_dim)
    H = build_hamiltonian_matrix(basis, operators)
    S = build_overlap_matrix(basis)
    S_reg = Symmetric(S + regularization * I)
    return tr(S_reg \ H)
end

# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

"""
    solve_ECG_variational(operators, n; kwargs...) -> SolverResults

Minimise a variational loss over the parameters of a `Rank0Gaussian` ECG
basis using any OptimKit.jl optimisation algorithm.

Two loss types are available via the `loss_type` keyword:

* **`:energy`** (default) — minimise the lowest generalised eigenvalue
  `λ_min` of `Hc = λSc`.  By the variational principle `λ_min ≥ E₀`
  for any basis, so its global minimum is the exact ground-state energy.
  This is the recommended choice and is equivalent to the standard
  Rayleigh-Ritz variational principle applied to the Gaussian parameters.
  Gradients are computed via ForwardDiff and the Hellmann-Feynman theorem
  (differentiating through H and S only, not the eigensolver).

* **`:trace`** — minimise `Tr(S⁻¹H)`, the sum of all generalised
  eigenvalues.  For a basis already close to the physical optimum this
  can accelerate convergence, but for a randomly initialised basis
  whose upper eigenvalues are large and positive the optimizer may find
  a degenerate near-zero minimum (all Gaussians spread to infinity).
  Prefer `:trace` only when warm-starting from a stochastic result.

The `A` matrices of the basis functions are parameterised through their
Cholesky factors (log-diagonal) which keeps them positive-definite for
any parameter vector, giving a smooth unconstrained problem.  Shift
vectors are fixed at zero, which is appropriate for ground states with
spherical symmetry.

After optimisation the full generalised eigenvalue problem `Hc = λSc`
is solved once to produce the ground-state energy and eigenvectors, so
all `SolverResults`-based utilities (`ψ₀`, `correlation_function`, etc.)
work unchanged.

# Arguments
- `operators` : `Vector{<:Operator}` — kinetic + Coulomb operators
- `n`         : number of basis functions (default 50)

# Keyword arguments
| keyword          | default        | description |
|:-----------------|:---------------|:------------|
| `loss_type`      | `:energy`      | `:energy` (λ_min) or `:trace` (Tr(S⁻¹H)) |
| `initial_basis`  | `nothing`      | warm-start `BasisSet{<:Rank0Gaussian}`; fresh QMC basis built when `nothing` |
| `scale`          | `0.2`          | length scale for QMC initialisation (ignored if `initial_basis` given) |
| `optimizer`      | `nothing`      | any OptimKit algorithm (e.g. `LBFGS()`, `ConjugateGradient()`); when `nothing` an L-BFGS is built from `max_iterations`, `gradient_tol`, and `verbose` |
| `max_iterations` | `500`          | max iterations for the default optimizer (ignored if `optimizer` is given) |
| `gradient_tol`   | `1e-6`         | gradient-norm tolerance for the default optimizer (ignored if `optimizer` is given) |
| `regularization` | `1e-10`        | Tikhonov shift added to S |
| `verbose`        | `true`         | print solver info messages; also sets verbosity of the default optimizer |

# Example

```julia
using FewBodyECG, OptimKit
masses = [1.0e15, 1.0, 1.0]
Λmat   = Λ(masses)
J, U   = _jacobi_transform(masses)
w_list = [[1,-1,0],[1,0,-1],[0,1,-1]]
w_raw  = [U'*w for w in w_list]
ops    = Operator[KineticOperator(Λmat);
                  [CoulombOperator(c,w) for (c,w) in zip([-1.,-1.,1.], w_raw)]...]

# Default L-BFGS (convenience kwargs control it)
sr = solve_ECG_variational(ops, 30; scale=1.0, max_iterations=300, verbose=false)
println(sr.ground_state)

# Pass any OptimKit algorithm directly
sr2 = solve_ECG_variational(ops, 30; scale=1.0,
                            optimizer=LBFGS(; maxiter=500, gradtol=1e-8, verbosity=2))
println(sr2.ground_state)

# Warm-start refinement with conjugate gradient
sr0   = solve_ECG(ops, 30; scale=1.0, verbose=false)
basis0 = BasisSet(Rank0Gaussian[sr0.basis_functions...])
sr_cg = solve_ECG_variational(ops, 30; initial_basis=basis0, loss_type=:trace,
                              optimizer=ConjugateGradient(; maxiter=200, verbosity=0))
println(sr_cg.ground_state)
```
"""
function solve_ECG_variational(
        operators::Vector{<:FewBodyHamiltonians.Operator},
        n::Int = 50;
        loss_type::Symbol = :energy,
        initial_basis::Union{BasisSet{<:Rank0Gaussian}, Nothing} = nothing,
        scale::Real = 0.2,
        optimizer = nothing,
        max_iterations::Int = 500,
        gradient_tol::Real = 1.0e-6,
        regularization::Real = 1.0e-10,
        verbose::Bool = true
    )

    loss_type in (:energy, :trace) ||
        throw(ArgumentError("loss_type must be :energy or :trace, got :$loss_type"))

    # Infer the Jacobi-coordinate dimension from the kinetic operator.
    n_dim = size(first(op for op in operators if op isa KineticOperator).K, 1)
    n_chol = n_dim * (n_dim + 1) ÷ 2   # Cholesky params per Gaussian
    n_per  = n_chol + n_dim             # total params per Gaussian (A + shift)

    # ---- build initial basis ------------------------------------------------
    if initial_basis !== nothing
        length(initial_basis.functions) == n || throw(ArgumentError(
            "initial_basis has $(length(initial_basis.functions)) functions, expected $n"
        ))
        basis_init = initial_basis
    else
        w_list = [op.w for op in operators if op isa CoulombOperator]
        b1 = float(scale)
        fns = Rank0Gaussian[]
        for i in 1:n
            bij = generate_bij(:quasirandom, i, length(w_list), b1)
            A = _generate_A_matrix(bij, w_list)
            push!(fns, Rank0Gaussian(A, zeros(n_dim)))
        end
        basis_init = BasisSet(fns)
    end

    θ0 = _encode_basis(basis_init)

    # Pre-allocate GradientConfig with a tuned chunk size.
    # Grouping ~5 Gaussians per chunk gives ~10 passes for n=50 in 2D
    # (vs the ForwardDiff default of 13), with larger gains for higher n_dim.
    _chunk = min(n_per * 5, length(θ0))
    _grad_cfg = ForwardDiff.GradientConfig(nothing, θ0, ForwardDiff.Chunk(_chunk))

    # Accumulates the objective value at every successful primal fg evaluation.
    # Reduced to a cumulative minimum after optimisation so convergence_history()
    # returns a monotone curve.
    energy_log = Float64[]

    # ---- combined value + ForwardDiff gradient (OptimKit interface) ---------
    # The LBFGS line search probes regions that can produce degenerate A
    # matrices; returning (Inf, zero-gradient) acts as an infinite-cost barrier.
    function fg(θ::AbstractVector)
        if loss_type === :energy
            # Primal: solve Float64 eigenproblem for λ_min and eigenvector c.
            local val::Float64, c::Vector{Float64}
            try
                basis_f64 = _decode_basis(θ, n, n_dim)
                H = build_hamiltonian_matrix(basis_f64, operators)
                S = build_overlap_matrix(basis_f64)
                evals, evecs = solve_generalized_eigenproblem(H, S; regularization)
                idx = argmin(evals)
                val = evals[idx]
                c = evecs[:, idx]
            catch
                return Inf, zeros(Float64, length(θ))
            end
            isfinite(val) || return Inf, zeros(Float64, length(θ))
            push!(energy_log, val)
            # Gradient via Hellmann-Feynman: ∂λ/∂θ = cᵀ(∂H/∂θ − λ·∂S/∂θ)c
            G = try
                ForwardDiff.gradient(θ, _grad_cfg, Val(false)) do θ_ad
                    basis_ad = _decode_basis(θ_ad, n, n_dim)
                    H_ad = build_hamiltonian_matrix(basis_ad, operators)
                    S_ad = build_overlap_matrix(basis_ad)
                    dot(c, H_ad * c) - val * dot(c, S_ad * c)
                end
            catch
                zeros(Float64, length(θ))
            end
            return val, G
        else  # :trace
            local val_t::Float64
            try
                basis_f64 = _decode_basis(θ, n, n_dim)
                H = build_hamiltonian_matrix(basis_f64, operators)
                S = build_overlap_matrix(basis_f64)
                v = tr((S + regularization * I) \ H)
                val_t = isfinite(v) ? v : Inf
            catch
                return Inf, zeros(Float64, length(θ))
            end
            isfinite(val_t) || return Inf, zeros(Float64, length(θ))
            push!(energy_log, val_t)
            G = try
                ForwardDiff.gradient(θ, _grad_cfg, Val(false)) do θ_ad
                    basis_ad = _decode_basis(θ_ad, n, n_dim)
                    H_ad = build_hamiltonian_matrix(basis_ad, operators)
                    S_ad = build_overlap_matrix(basis_ad)
                    tr((S_ad + regularization * I) \ H_ad)
                end
            catch
                zeros(Float64, length(θ))
            end
            return val_t, G
        end
    end

    # ---- optimise -----------------------------------------------------------
    method = if optimizer !== nothing
        optimizer
    else
        LBFGS(; maxiter = max_iterations, gradtol = float(gradient_tol),
                verbosity = verbose ? 2 : 0)
    end

    verbose && @info "Starting variational ECG optimisation" n_basis = n n_params = length(θ0) loss_type

    θ_opt, f_opt, _, _, _ = optimize(fg, θ0, method)

    verbose && @info "Optimisation done" loss = f_opt

    # ---- reconstruct basis and solve eigenproblem ---------------------------
    basis_opt = _decode_basis(θ_opt, n, n_dim)
    H_opt = build_hamiltonian_matrix(basis_opt, operators)
    S_opt = build_overlap_matrix(basis_opt)
    evals, evecs = solve_generalized_eigenproblem(H_opt, S_opt)
    ground_state = minimum(evals)

    @info "Variational ECG complete" E₀ = ground_state n_basis = n

    # Reduce energy_log to a cumulative minimum so convergence_history() returns
    # a monotone decreasing curve regardless of line-search noise.
    fg_history = isempty(energy_log) ? Float64[] : accumulate(min, energy_log)

    return SolverResults(
        Vector{GaussianBase}(basis_opt.functions),
        n,
        operators,
        :variational,
        HaltonSample(),   # placeholder: no stochastic sampler is used
        float(scale),
        ground_state,
        [ground_state],   # single-point; no greedy build-up history
        [evecs],
        fg_history,
    )
end
