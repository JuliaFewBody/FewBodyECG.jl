using OptimKit: optimize

# ForwardDiff compiles the energy gradient once per chunk size, so a fixed
# chunk lets every system reuse one compiled gradient.  Larger chunks are no
# faster for ECG energies.
const _GRADIENT_CHUNK = 12

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

struct _GradientLayout{R, S}
    symmetry::S
    shifts::Bool
end

function _gradient_layout(rank::Int, symmetry; shifts = rank == 0 && symmetry === nothing)
    rank in (0, 1) || throw(ArgumentError("gradient methods support rank 0 or 1"))
    return _GradientLayout{rank, typeof(symmetry)}(symmetry, shifts)
end

function _gradient_layout(basis::BasisSet)
    isempty(basis.functions) && throw(ArgumentError("gradient basis must be nonempty"))
    rank = if all(g -> _unprojected(g) isa Rank0Gaussian, basis)
        0
    elseif all(g -> _unprojected(g) isa Rank1Gaussian, basis)
        1
    else
        throw(ArgumentError("gradient methods require a uniformly Rank0Gaussian or Rank1Gaussian basis"))
    end
    symmetry = _basis_symmetry(basis)
    any(g -> g isa ProjectedGaussian, basis) && symmetry === nothing &&
        throw(ArgumentError("all gradient basis functions must share the same symmetry"))
    for g in basis
        _gradient_norm(g)
    end
    shifts = (rank == 0 && symmetry === nothing) || any(g -> any(!iszero, _unprojected(g).s), basis)
    return _gradient_layout(rank, symmetry; shifts)
end

_gradient_rank(::_GradientLayout{R}) where {R} = R
_gradient_parameters(d, layout) = d * (d + 1) ÷ 2 + 3d * (layout.shifts + _gradient_rank(layout))

function _gradient_norm(g)
    value = _compute_matrix_element(g, g)
    isfinite(value) && value > 0 || throw(ArgumentError("gradient basis contains a zero or non-finite norm"))
    if g isa ProjectedGaussian
        value > 1.0e-12 * _compute_matrix_element(g.orbital, g.orbital) ||
            throw(ArgumentError("gradient basis contains a numerically null projected function"))
    end
    return value
end

_encode_basis(basis::BasisSet) = _encode_basis(basis, _gradient_layout(basis))
function _encode_basis(basis::BasisSet, layout::_GradientLayout{R}) where {R}
    params = Float64[]
    for projected in basis
        g = _unprojected(projected)
        C = cholesky(Symmetric(Matrix(g.A)))
        append!(params, _chol_to_params(Matrix(C.L)))
        layout.shifts && append!(params, Float64.(vec(g.s)))
        R == 1 && append!(params, Float64.(vec(g.a)))
    end
    return params
end

function _decode_basis(
        θ::AbstractVector, n_basis::Int, n_dim::Int,
        layout::_GradientLayout{R} = _gradient_layout(0, nothing)
    ) where {R}
    T = eltype(θ)
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_shift = layout.shifts ? 3n_dim : 0
    n_per = _gradient_parameters(n_dim, layout)
    fns = map(1:n_basis) do i
        start = (i - 1) * n_per + 1
        A = _params_to_matrix(θ[start:(start + n_chol - 1)], n_dim)
        offset = start + n_chol
        s = layout.shifts ? reshape(θ[offset:(offset + n_shift - 1)], n_dim, 3) : zeros(T, n_dim, 3)
        if R == 0
            g = Rank0Gaussian(Matrix(A), s)
        else
            a = reshape(θ[(offset + n_shift):(start + n_per - 1)], n_dim, 3)
            g = Rank1Gaussian(Matrix(A), a / norm(a), s)
        end
        return layout.symmetry === nothing ? g : ProjectedGaussian(g, layout.symmetry)
    end
    return BasisSet(fns)
end

function _gradient_matrices(basis, terms)
    norms = [_gradient_norm(g) for g in basis]
    D = Diagonal(1 ./ sqrt.(norms))
    S = build_overlap_matrix(basis)
    H = build_hamiltonian_matrix(basis, terms)
    return D * H * D, D * S * D, D
end

function _gradient_eigenproblem(basis, terms; regularization = 0.0)
    H, S, D = _gradient_matrices(basis, terms)
    isposdef(Symmetric(S)) && cond(Symmetric(S)) < 1.0e12 ||
        throw(ArgumentError("gradient basis overlap is singular or too ill-conditioned"))
    values, vectors = solve_generalized_eigenproblem(H, S; regularization, max_condition = Inf)
    return values, vectors, D
end

function _energy_gradient(
        θ, n, n_dim, terms, grad_cfg, state, regularization, bad_gradient,
        layout = _gradient_layout(0, nothing)
    )
    local energy::Float64, coefficients::Vector{Float64}
    try
        basis = _decode_basis(θ, n, n_dim, layout)
        values, vectors, _ = _gradient_eigenproblem(basis, terms; regularization)
        index = min(state, length(values))
        energy = values[index]
        coefficients = vectors[:, index]
    catch
        return Inf, zeros(Float64, length(θ))
    end
    isfinite(energy) || return Inf, zeros(Float64, length(θ))

    gradient = try
        ForwardDiff.gradient(θ, grad_cfg, Val(false)) do θ_ad
            basis = _decode_basis(θ_ad, n, n_dim, layout)
            H, S, _ = _gradient_matrices(basis, terms)
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

function _solution_from_basis(basis::BasisSet, ctx::_SolveCtx, stages, draws)
    values, vectors, D = _gradient_eigenproblem(basis, ctx.terms)
    state = _require_available_state(ctx.state, length(values))
    return Solution(
        values, basis, D * vectors, ctx.terms,
        state, stages, last(stages).report, ctx.units, draws, ctx.hamiltonian
    )
end

function _gradient_layout(alg::GradientMethod, init, ctx)
    layout = init === nothing ? _gradient_layout(something(alg.rank, 0), ctx.symmetry) : _gradient_layout(init.basis)
    alg.rank === nothing || alg.rank == _gradient_rank(layout) ||
        throw(ArgumentError("gradient rank=$(alg.rank) does not match the initial basis"))
    return layout
end

function _gradient_stream(layout, draws = 0)
    rank = _gradient_rank(layout)
    G = layout.symmetry === nothing ? (rank == 0 ? Rank0Gaussian : Rank1Gaussian) : ProjectedGaussian
    stream = BasisState(G; rank, symmetry = layout.symmetry)
    stream.draw = draws
    return stream
end

function _initial_gradient_basis(ctx, layout, n, scale, widths)
    n > 0 || throw(ArgumentError("GVM basis size must be positive"))
    stream = _gradient_stream(layout)
    for sampling_widths in (widths === nothing ? (:linear, :wide) : (widths,))
        functions = [_draw_candidate!(stream, scale, HaltonSample(), sampling_widths, ctx.w_list, ctx.d) for _ in 1:n]
        basis = BasisSet(functions)
        try
            _gradient_eigenproblem(basis, ctx.terms)
            note = stream.draw == n ? nothing : "cold GVM selected $n functions from $(stream.draw) draws (widths = :$sampling_widths)"
            return basis, stream.draw, note
        catch
            # A larger pool lets pivoted Cholesky select independent functions.
        end
        for _ in 1:(9n)
            push!(functions, _draw_candidate!(stream, scale, HaltonSample(), sampling_widths, ctx.w_list, ctx.d))
        end
        filter!(functions) do g
            try
                _gradient_norm(g)
                return true
            catch
                return false
            end
        end
        length(functions) ≥ n || continue
        S = build_overlap_matrix(BasisSet(functions))
        D = Diagonal(1 ./ sqrt.(diag(S)))
        factorization = cholesky(Symmetric(D * S * D), RowMaximum(); check = false)
        factorization.rank ≥ n || continue
        basis = BasisSet(functions[factorization.p[1:n]])
        try
            _gradient_eigenproblem(basis, ctx.terms)
            return basis, stream.draw, "cold GVM selected $n functions from $(stream.draw) draws (widths = :$sampling_widths)"
        catch
            continue
        end
    end
    throw(ArgumentError("could not initialize $n independent Gaussians; try wider widths, fewer functions, or an SVM warm start, and check rank and exchange signs"))
end

function _solve(
        terms, masses, alg::GVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing, symmetry = nothing
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian, symmetry)
    layout = _gradient_layout(alg, init, ctx)
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)

    scale_note = nothing
    initial_note = nothing
    if init === nothing
        alg.basis === nothing && throw(
            ArgumentError("cold GVM requires `basis`; use GVM(basis = n)")
        )
        n = something(alg.basis)
        scale, scale_note = _solve_scale(something(alg.scale, :adaptive), ctx, nothing; rank = _gradient_rank(layout))
        basis, draws, initial_note = _initial_gradient_basis(ctx, layout, n, scale, alg.widths)
        layout = _gradient_layout(basis)
        θ = _encode_basis(basis, layout)
    else
        alg.scale === nothing || throw(
            ArgumentError("warm GVM uses the basis from `init`; omit `scale`")
        )
        alg.widths === nothing || throw(
            ArgumentError("warm GVM uses the basis from `init`; omit `widths`")
        )
        n = length(init.basis.functions)
        alg.basis === nothing || alg.basis == n || throw(
            ArgumentError(
                "init has $n functions but GVM specifies basis = $(alg.basis); " *
                    "omit `basis` or set basis = $n"
            )
        )
        θ = _encode_basis(init.basis, layout)
        draws = init.draws
    end

    chunk = min(_GRADIENT_CHUNK, length(θ))
    # A named tag composes with the nested shift derivatives of polarized bases.
    grad_cfg = ForwardDiff.GradientConfig(_energy_gradient, θ, ForwardDiff.Chunk(chunk))
    bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    fg = x -> _energy_gradient(
        x, n, n_dim, ctx.terms, grad_cfg, ctx.state, 0.0, bad_gradient, layout
    )
    report_iteration = function (x, energy, gradient, iteration)
        verbose && @info "GVM: iteration $iteration" energy gradnorm = norm(gradient)
        return x, energy, gradient
    end

    x, energy, gradient, _, history = optimize(
        fg, θ, alg.optimizer; finalize! = report_iteration
    )
    _check_optimization_result(x, energy, bad_gradient)

    basis = _decode_basis(x, n, n_dim, layout)
    energy_history = accumulate(min, history[:, 1])
    ΔE = length(energy_history) ≥ 2 ? abs(energy_history[end - 1] - energy_history[end]) : NaN
    report = _gradient_report(
        norm(gradient), alg.optimizer.gradtol, ΔE,
        cond(Symmetric(build_overlap_matrix(basis)))
    )
    _add_note!(report, initial_note)
    return _solution_from_basis(
        basis, ctx, [StageResult(alg, energy_history, _add_note!(report, scale_note))], draws
    )
end

function _solve(
        terms, masses, alg::DynamicGVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing, symmetry = nothing
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian, symmetry)
    layout = _gradient_layout(alg, init, ctx)
    scale, scale_note = _solve_scale(alg.scale, ctx, init; rank = _gradient_rank(layout))
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    n_per = _gradient_parameters(n_dim, layout)
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
        θ = _encode_basis(init.basis, layout)
    end

    initial_size = length(θ) ÷ n_per
    stream = _gradient_stream(layout, init === nothing ? 0 : init.draws)
    energy_history = Float64[]
    gradnorm = NaN

    for step in (initial_size + 1):alg.basis
        best_energy = Inf
        best_candidate = Float64[]

        for _ in 1:alg.candidates
            candidate = _draw_candidate!(stream, scale, HaltonSample(), alg.widths, ctx.w_list, n_dim)
            candidate_parameters = _encode_basis(
                BasisSet([candidate]), layout
            )
            trial_parameters = [θ; candidate_parameters]
            try
                basis = _decode_basis(trial_parameters, step, n_dim, layout)
                values, _, _ = _gradient_eigenproblem(basis, ctx.terms)
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
        chunk = min(_GRADIENT_CHUNK, length(θ))
        grad_cfg = ForwardDiff.GradientConfig(_energy_gradient, θ, ForwardDiff.Chunk(chunk))
        bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        fg = x -> _energy_gradient(
            x, step, n_dim, ctx.terms, grad_cfg, ctx.state, 0.0, bad_gradient, layout
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
    basis = _decode_basis(θ, n_built, n_dim, layout)
    ΔE = length(energy_history) ≥ 2 ? energy_history[end - 1] - energy_history[end] : NaN
    report = _gradient_report(
        gradnorm, alg.optimizer.gradtol, ΔE,
        cond(Symmetric(build_overlap_matrix(basis)))
    )
    return _solution_from_basis(
        basis, ctx, [StageResult(alg, energy_history, _add_note!(report, scale_note))], stream.draw
    )
end
