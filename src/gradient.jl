using OptimKit: optimize
import ForwardDiff

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

function _energy_gradient(θ, n, n_dim, terms, grad_cfg, state, regularization, bad_gradient)
    local energy::Float64, coefficients::Vector{Float64}
    try
        basis = _decode_basis(θ, n, n_dim)
        H = build_hamiltonian_matrix(basis, terms)
        S = build_overlap_matrix(basis)
        values, vectors = solve_generalized_eigenproblem(H, S; regularization)
        index = min(state, length(values))
        energy = values[index]
        coefficients = vectors[:, index]
    catch
        return Inf, zeros(Float64, length(θ))
    end
    isfinite(energy) || return Inf, zeros(Float64, length(θ))

    gradient = try
        ForwardDiff.gradient(θ, grad_cfg, Val(false)) do θ_ad
            basis = _decode_basis(θ_ad, n, n_dim)
            H = build_hamiltonian_matrix(basis, terms)
            S = build_overlap_matrix(basis)
            dot(coefficients, H * coefficients) -
                energy * dot(coefficients, S * coefficients)
        end
    catch e
        throw(
            ArgumentError(
                "automatic differentiation failed; operator matrix elements must " *
                    "support ForwardDiff.Dual values: $(sprint(showerror, e))"
            )
        )
    end
    if !all(isfinite, gradient)
        bad_gradient[] = gradient
        return Inf, zeros(Float64, length(θ))
    end
    return energy, gradient
end

function _check_optimization_result(x, energy, bad_gradient)
    isfinite(energy) && return
    bad_gradient[] === nothing && throw(
        DomainError(x, "optimization ended at an invalid parameter point")
    )
    throw(
        DomainError(
            bad_gradient[],
            "automatic differentiation returned a non-finite gradient"
        )
    )
end

function _gradient_report(gradnorm, gradtol, ΔE, cond_S)
    converged = gradnorm < gradtol
    return ConvergenceReport(
        converged, converged ? :stationarity : :max_steps, ΔE, gradtol, 0,
        gradnorm, cond_S,
        [
            "stationary point of the parameter optimisation; " *
                "the variational upper bound still applies",
        ]
    )
end

function _solution_from_basis(basis::BasisSet, ctx::_SolveCtx, stages)
    H = build_hamiltonian_matrix(basis, ctx.terms)
    S = build_overlap_matrix(basis)
    values, vectors = solve_generalized_eigenproblem(H, S)
    state = _require_available_state(ctx.state, length(values))
    return Solution(
        values, basis, vectors, ctx.terms,
        state, stages, last(stages).report
    )
end

function _solve(
        terms, masses, alg::GVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_per = n_chol + 3 * n_dim

    if init === nothing
        alg.basis === nothing && throw(
            ArgumentError("cold GVM requires `basis`; use GVM(basis = n)")
        )
        n = something(alg.basis)
        scale = _resolve_scale(alg.scale === nothing ? :auto : alg.scale, ctx.masses)
        functions = Rank0Gaussian[]
        for i in 1:n
            bij = generate_bij(:quasirandom, i, length(ctx.w_list), scale)
            A = _generate_A_matrix(bij, ctx.w_list)
            push!(functions, Rank0Gaussian(A, zeros(n_dim, 3)))
        end
        θ = _encode_basis(BasisSet(functions))
    else
        alg.scale === nothing || throw(
            ArgumentError("warm GVM uses the basis from `init`; omit `scale`")
        )
        n = length(init.basis.functions)
        alg.basis === nothing || alg.basis == n || throw(
            ArgumentError(
                "init has $n functions but GVM specifies basis = $(alg.basis); " *
                    "omit `basis` or set basis = $n"
            )
        )
        θ = _encode_basis(BasisSet(Rank0Gaussian[g for g in init.basis.functions]))
    end

    chunk = min(n_per * 5, length(θ))
    grad_cfg = ForwardDiff.GradientConfig(nothing, θ, ForwardDiff.Chunk(chunk))
    bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    fg = x -> _energy_gradient(
        x, n, n_dim, ctx.terms, grad_cfg, ctx.state, 1.0e-10, bad_gradient
    )
    report_iteration = function (x, energy, gradient, iteration)
        verbose && @info "GVM: iteration $iteration" energy gradnorm = norm(gradient)
        return x, energy, gradient
    end

    x, energy, gradient, _, history = optimize(
        fg, θ, alg.optimizer; finalize! = report_iteration
    )
    _check_optimization_result(x, energy, bad_gradient)

    basis = _decode_basis(x, n, n_dim)
    energy_history = accumulate(min, history[:, 1])
    ΔE = length(energy_history) ≥ 2 ? abs(energy_history[end - 1] - energy_history[end]) : NaN
    report = _gradient_report(
        norm(gradient), alg.optimizer.gradtol, ΔE,
        cond(Symmetric(build_overlap_matrix(basis)))
    )
    return _solution_from_basis(
        basis, ctx, [StageResult(alg, energy_history, report)]
    )
end

function _solve(
        terms, masses, alg::DynamicGVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose)
    scale = _resolve_scale(alg.scale, ctx.masses)
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_per = n_chol + 3 * n_dim
    θ = Float64[]

    if init !== nothing
        initial_size = length(init.basis.functions)
        initial_size < alg.basis || throw(
            ArgumentError(
                "init already has $initial_size functions but DynamicGVM grows to " *
                    "basis = $(alg.basis); set basis > $initial_size or use GVM() " *
                    "to re-optimise"
            )
        )
        θ = _encode_basis(BasisSet(Rank0Gaussian[g for g in init.basis.functions]))
    end

    initial_size = length(θ) ÷ n_per
    energy_history = Float64[]
    gradnorm = NaN

    for step in (initial_size + 1):alg.basis
        best_energy = Inf
        best_candidate = Float64[]

        for candidate in 1:alg.candidates
            attempt = (step - 1) * alg.candidates + candidate
            bij = generate_bij(:quasirandom, attempt, length(ctx.w_list), scale)
            A = _generate_A_matrix(bij, ctx.w_list)
            candidate_parameters = _encode_basis(
                BasisSet([Rank0Gaussian(A, zeros(n_dim, 3))])
            )
            trial_parameters = [θ; candidate_parameters]
            try
                basis = _decode_basis(trial_parameters, step, n_dim)
                H = build_hamiltonian_matrix(basis, ctx.terms)
                S = build_overlap_matrix(basis)
                values, _ = solve_generalized_eigenproblem(H, S; regularization = 1.0e-10)
                energy = values[min(ctx.state, length(values))]
                if energy < best_energy
                    best_energy = energy
                    best_candidate = candidate_parameters
                end
            catch
                continue
            end
        end

        if isempty(best_candidate)
            verbose && @warn "Sequential search stopped at step $step: all $(alg.candidates) candidates failed (overlap likely singular). Returning the $(step - 1) functions built so far; try a smaller `scale`."
            break
        end

        append!(θ, best_candidate)
        chunk = min(n_per * 5, length(θ))
        grad_cfg = ForwardDiff.GradientConfig(nothing, θ, ForwardDiff.Chunk(chunk))
        bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        fg = x -> _energy_gradient(
            x, step, n_dim, ctx.terms, grad_cfg, ctx.state, 1.0e-10, bad_gradient
        )
        report_iteration = function (x, energy, gradient, iteration)
            verbose && @info "DynamicGVM: step $step/$(alg.basis), iteration $iteration" energy gradnorm = norm(gradient)
            return x, energy, gradient
        end

        θ, energy, gradient, evaluations, _ = optimize(
            fg, θ, alg.optimizer; finalize! = report_iteration
        )
        _check_optimization_result(θ, energy, bad_gradient)
        gradnorm = norm(gradient)
        push!(energy_history, energy)
        verbose && @info "DynamicGVM: completed step $step/$(alg.basis)" energy evaluations
    end

    n_built = length(θ) ÷ n_per
    n_built ≥ 1 || error("Sequential selection produced no basis functions")
    basis = _decode_basis(θ, n_built, n_dim)
    ΔE = length(energy_history) ≥ 2 ? energy_history[end - 1] - energy_history[end] : NaN
    report = _gradient_report(
        gradnorm, alg.optimizer.gradtol, ΔE,
        cond(Symmetric(build_overlap_matrix(basis)))
    )
    return _solution_from_basis(
        basis, ctx, [StageResult(alg, energy_history, report)]
    )
end
