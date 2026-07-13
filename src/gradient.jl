using OptimKit
using LinearAlgebra
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
        append!(params, Float64.(vec(parent(g.s))))   # N×3 shift, column-major
    end
    return params
end

# Decode a flat parameter vector back into a BasisSet{Rank0Gaussian}.
# Layout per Gaussian: [n_chol Cholesky params | 3·n_dim shift params (N×3)].
function _decode_basis(θ::AbstractVector, n_basis::Int, n_dim::Int)
    T = eltype(θ)
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_shift = 3 * n_dim
    n_per = n_chol + n_shift
    fns = Vector{Rank0Gaussian{T, Matrix{T}, Matrix{T}}}(undef, n_basis)
    for i in 1:n_basis
        start = (i - 1) * n_per + 1
        A = _params_to_matrix(θ[start:(start + n_chol - 1)], n_dim)
        s = reshape(θ[(start + n_chol):(start + n_per - 1)], n_dim, 3)
        fns[i] = Rank0Gaussian(Matrix(A), Matrix(s))
    end
    return BasisSet(fns)
end
# Core LBFGS engine.  θ0 === nothing ⇒ fresh QMC basis of n functions at
# `scale`.  Returns the optimised basis, the cumulative-min fg history, and
# the final gradient norm from OptimKit's normgradhistory.
function _variational_engine(
        terms, n::Int, θ0, scale::Float64,
        maxiter::Int, gtol::Float64, verbose::Bool;
        state::Int = 1, shift_init::Symbol = :zeros
    )
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    n_chol = n_dim * (n_dim + 1) ÷ 2   # Cholesky params per Gaussian
    n_per = n_chol + 3 * n_dim         # total params per Gaussian (A + N×3 shift)
    regularization = 1.0e-10

    if θ0 === nothing
        w_list = _pairwise_weights(terms)
        fns = Rank0Gaussian[]
        for i in 1:n
            bij = generate_bij(:quasirandom, i, length(w_list), scale)
            A = _generate_A_matrix(bij, w_list)
            s = shift_init === :zeros ? zeros(n_dim, 3) : generate_shift(:quasirandom, i, n_dim, scale)
            push!(fns, Rank0Gaussian(A, s))
        end
        θ0 = _encode_basis(BasisSet(fns))
    end

    # Pre-allocate GradientConfig with a tuned chunk size.
    # Grouping ~5 Gaussians per chunk gives ~10 passes for n=50 in 2D
    # (vs the ForwardDiff default of 13), with larger gains for higher n_dim.
    _chunk = min(n_per * 5, length(θ0))
    _grad_cfg = ForwardDiff.GradientConfig(nothing, θ0, ForwardDiff.Chunk(_chunk))

    # Accumulates objective values, then stores the cumulative minimum as the
    # method's monotone energy history.
    energy_log = Float64[]

    # ---- combined value + ForwardDiff gradient (OptimKit interface) ---------
    # The LBFGS line search probes regions that can produce degenerate A
    # matrices; returning (Inf, zero-gradient) acts as an infinite-cost barrier.
    function fg(θ::AbstractVector)
        # Primal: solve Float64 eigenproblem for the target eigenpair.
        local val::Float64, c::Vector{Float64}
        try
            basis_f64 = _decode_basis(θ, n, n_dim)
            H = build_hamiltonian_matrix(basis_f64, terms)
            S = build_overlap_matrix(basis_f64)
            evals, evecs = solve_generalized_eigenproblem(H, S; regularization)
            idx = min(state, length(evals))
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
                H_ad = build_hamiltonian_matrix(basis_ad, terms)
                S_ad = build_overlap_matrix(basis_ad)
                dot(c, H_ad * c) - val * dot(c, S_ad * c)
            end
        catch
            zeros(Float64, length(θ))
        end
        # ForwardDiff through a near-singular overlap can yield silent NaNs;
        # treat a non-finite gradient as a zero-gradient barrier.
        all(isfinite, G) || (G = zeros(Float64, length(θ)))
        return val, G
    end

    # OptimKit emits @warn for linesearch bisection failures that it handles
    # gracefully internally.  Suppress them to keep output clean.
    x, _, _, _, normgradhistory = Base.CoreLogging.with_logger(
        Base.CoreLogging.ConsoleLogger(Base.stderr, Base.CoreLogging.Error)
    ) do
        optimize(fg, θ0, LBFGS(; maxiter, gradtol = gtol, verbosity = verbose ? 2 : 0))
    end
    basis = _decode_basis(x, n, n_dim)
    fg_hist = isempty(energy_log) ? Float64[] : accumulate(min, energy_log)
    return basis, fg_hist, float(last(normgradhistory))
end
# Core sequential (SVM-style) engine: at each step k = k0+1, …, n draw
# `candidates` quasi-random Gaussians, keep the one giving the lowest
# pre-optimisation target-state energy, then jointly LBFGS-optimise all k
# functions' parameters.  θ0 === nothing ⇒ start from an empty basis
# (k0 = 0); otherwise θ0 seeds `θ_running` and growth continues from
# `length(θ0) ÷ n_per` functions.  `gradnorm` is the final gradient norm
# from the LAST step's optimize call.
function _sequential_engine(
        terms, n::Int, θ0, scale::Float64, candidates::Int,
        maxiter_step::Int, gtol::Float64, verbose::Bool;
        state::Int = 1, shift_init::Symbol = :zeros
    )
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_per = n_chol + 3 * n_dim
    w_list = _pairwise_weights(terms)
    regularization = 1.0e-10

    method = LBFGS(; maxiter = maxiter_step, gradtol = gtol, verbosity = 0)

    energy_log = Float64[]   # all fg values across all steps → cummin history
    step_hist = Float64[]    # target-state energy after each step's optimisation
    θ_running = θ0 === nothing ? Float64[] : copy(θ0)
    k0 = θ0 === nothing ? 0 : length(θ0) ÷ n_per
    gradnorm = NaN

    for step in (k0 + 1):n
        k = step   # basis size after this step

        # ── candidate selection ──────────────────────────────────────────────
        # Sample `candidates` quasi-random Gaussians; keep the one giving the
        # lowest target-state energy before optimisation.
        best_E_cand = Inf
        best_θ_cand = Float64[]

        for c in 1:candidates
            attempt = (step - 1) * candidates + c
            bij = generate_bij(:quasirandom, attempt, length(w_list), scale)
            A = _generate_A_matrix(bij, w_list)
            s = shift_init === :zeros ? zeros(n_dim, 3) : generate_shift(:quasirandom, attempt, n_dim, scale)
            cand = Rank0Gaussian(A, s)
            θ_c = _encode_basis(BasisSet([cand]))
            θ_t = [θ_running; θ_c]
            try
                b_t = _decode_basis(θ_t, k, n_dim)
                H_t = build_hamiltonian_matrix(b_t, terms)
                S_t = build_overlap_matrix(b_t)
                ev_t, _ = solve_generalized_eigenproblem(H_t, S_t; regularization)
                E_t = ev_t[min(state, length(ev_t))]
                if E_t < best_E_cand
                    best_E_cand = E_t
                    best_θ_cand = θ_c
                end
            catch
                continue
            end
        end

        # If every candidate failed, the accumulated basis has become singular
        # (functions collapsed onto each other during optimisation — common when
        # `scale` is too large for the system).  Rather than crash, stop here and
        # return the functions built so far.
        if isempty(best_θ_cand)
            verbose && @warn "Sequential search stopped at step $step: all $candidates candidates failed (overlap likely singular). Returning the $(step - 1) functions built so far; try a smaller `scale`."
            break
        end

        θ_running = [θ_running; best_θ_cand]

        # ── optimise all k-function parameters ──────────────────────────────
        _chunk = min(n_per * 5, length(θ_running))
        _grad_cfg = ForwardDiff.GradientConfig(
            nothing, θ_running,
            ForwardDiff.Chunk(_chunk)
        )
        step_log = Float64[]

        # Build the fg closure for the current k-function basis.
        # Capture k, _grad_cfg, step_log, and other constants by reference;
        # each loop iteration creates a fresh set of these locals.
        fg_k = (θ::AbstractVector) -> begin
            local val::Float64, c::Vector{Float64}
            try
                b = _decode_basis(θ, k, n_dim)
                H = build_hamiltonian_matrix(b, terms)
                S = build_overlap_matrix(b)
                ev, ev_vecs = solve_generalized_eigenproblem(H, S; regularization)
                idx = min(state, length(ev))
                val = ev[idx]
                c = ev_vecs[:, idx]
            catch
                return Inf, zeros(Float64, length(θ))
            end
            isfinite(val) || return Inf, zeros(Float64, length(θ))
            push!(step_log, val)
            G = try
                ForwardDiff.gradient(θ, _grad_cfg, Val(false)) do θ_ad
                    b_ad = _decode_basis(θ_ad, k, n_dim)
                    H_ad = build_hamiltonian_matrix(b_ad, terms)
                    S_ad = build_overlap_matrix(b_ad)
                    dot(c, H_ad * c) - val * dot(c, S_ad * c)
                end
            catch
                zeros(Float64, length(θ))
            end
            all(isfinite, G) || (G = zeros(Float64, length(θ)))
            return val, G
        end

        θ_opt, _, _, _, normgradhistory = Base.CoreLogging.with_logger(
            Base.CoreLogging.ConsoleLogger(Base.stderr, Base.CoreLogging.Error)
        ) do
            optimize(fg_k, θ_running, method)
        end
        θ_running = θ_opt
        gradnorm = float(last(normgradhistory))
        append!(energy_log, step_log)

        # Record the target-state energy after this step's full optimisation.
        b_k = _decode_basis(θ_running, k, n_dim)
        H_k = build_hamiltonian_matrix(b_k, terms)
        S_k = build_overlap_matrix(b_k)
        ev_k, _ = solve_generalized_eigenproblem(H_k, S_k; regularization)
        push!(step_hist, ev_k[min(state, length(ev_k))])

        verbose && @info "Step $step/$n" E = last(step_hist) fg_evals = length(step_log)
    end

    # `n_built` may be < n if the search stopped early (singular basis).
    n_built = length(θ_running) ÷ n_per
    n_built >= 1 || error("Sequential selection produced no basis functions")
    basis = _decode_basis(θ_running, n_built, n_dim)
    fg_hist = isempty(energy_log) ? Float64[] : accumulate(min, energy_log)
    return basis, step_hist, fg_hist, gradnorm
end
