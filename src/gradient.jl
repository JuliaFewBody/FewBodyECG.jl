using OptimKit: optimize

# Cap the dual width for large parameter blocks.
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

# Parameter layout of a gradient basis: rank `R` and exchange symmetry type `S`
# as type parameters, so decoding yields a concretely typed basis.  `shifts`
# says whether the shifts are optimized; unshifted projected or Rank1 bases stay
# unshifted to keep their angular character.
struct _GradientLayout{R, S}
    symmetry::S
    shifts::Bool
end

function _GradientLayout(rank::Integer, symmetry; shifts = rank == 0 && symmetry === nothing)
    rank in (0, 1) || throw(ArgumentError("gradient methods support rank 0 or 1, got $rank"))
    return _GradientLayout{Int(rank), typeof(symmetry)}(symmetry, shifts)
end

# Layout of an existing basis.
function _gradient_layout(basis::BasisSet)
    isempty(basis.functions) && throw(ArgumentError("gradient basis must be nonempty"))
    all(_admissible_gradient_function, basis) ||
        throw(ArgumentError("gradient basis contains a null or non-finite function"))
    rank = _basis_rank(basis)
    symmetry = _shared_symmetry(basis)
    shifts = (rank == 0 && symmetry === nothing) || any(g -> any(!iszero, _unprojected(g).s), basis)
    return _GradientLayout(rank, symmetry; shifts)
end

# Layout requested by a solver call: from `init` when given, else from `alg.rank`.
function _resolve_gradient_layout(alg::GradientMethod, init, ctx)
    layout = init === nothing ? _GradientLayout(something(alg.rank, 0), ctx.symmetry) : _gradient_layout(init.basis)
    alg.rank === nothing || alg.rank == _gradient_rank(layout) ||
        throw(ArgumentError("gradient rank=$(alg.rank) does not match the initial basis"))
    return layout
end

_gradient_rank(::_GradientLayout{R}) where {R} = R
_gradient_parameters(d, layout) = d * (d + 1) ÷ 2 + 3d * (layout.shifts + _gradient_rank(layout))

# A function the optimizer can evaluate: finite, positive-definite widths, a
# finite nonzero norm, and not annihilated by its projector.  Trial points with
# other functions get energy `Inf`, from which the line search backtracks.
function _admissible_gradient_function(g)
    A = _unprojected(g).A
    all(isfinite, A) && isposdef(A) || return false
    norm2 = _compute_matrix_element(g, g)
    return isfinite(norm2) && norm2 > 0 && _survives_projection(g, norm2)
end

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

function _decode_gradient_function(θ, n_dim, layout::_GradientLayout{R}) where {R}
    n_chol = n_dim * (n_dim + 1) ÷ 2
    n_shift = layout.shifts ? 3n_dim : 0
    A = Matrix(_params_to_matrix(view(θ, 1:n_chol), n_dim))
    offset = n_chol + 1
    s = layout.shifts ? reshape(view(θ, offset:(offset + n_shift - 1)), n_dim, 3) : zeros(eltype(θ), n_dim, 3)
    if R == 0
        g = Rank0Gaussian(A, s)
    else
        a = reshape(view(θ, (offset + n_shift):lastindex(θ)), n_dim, 3)
        g = Rank1Gaussian(A, a / norm(a), s)
    end
    return layout.symmetry === nothing ? g : ProjectedGaussian(g, layout.symmetry)
end

function _decode_basis(θ::AbstractVector, n_basis::Int, n_dim::Int, layout::_GradientLayout)
    n_per = _gradient_parameters(n_dim, layout)
    fns = map(1:n_basis) do i
        start = (i - 1) * n_per + 1
        return _decode_gradient_function(view(θ, start:(start + n_per - 1)), n_dim, layout)
    end
    return BasisSet(fns)
end

# H and S in coordinates where every function has unit norm, and the scaling D.
function _gradient_matrices(basis, terms)
    D = Diagonal([1 / sqrt(_compute_matrix_element(g, g)) for g in basis])
    return D * build_hamiltonian_matrix(basis, terms) * D, D * build_overlap_matrix(basis) * D, D
end

# Eigenpairs of a gradient basis in unit-norm coordinates, with the scaling D;
# `nothing` when a function is null or the overlap is too ill-conditioned to
# solve without regularization.
function _gradient_eigenproblem(basis, terms)
    all(_admissible_gradient_function, basis) || return nothing
    H, S, D = _gradient_matrices(basis, terms)
    λmin, λmax = extrema(eigvals(Hermitian(S)))
    λmin > 0 && λmax < 1.0e12 * λmin || return nothing
    F = eigen(Hermitian(H), Hermitian(S))
    return F.values, F.vectors, D
end

# Index of the optimized state.  DynamicGVM starts from fewer functions than
# `state`; until the basis is large enough it optimizes the highest state.
_target_index(state, values) = min(state, length(values))

function _gradient_config(θ, n_dim, layout, terms)
    # External operators may require matching scalar types on bra and ket.
    # Permutation terms can fill the two triangles independently.
    rowwise = all(terms) do op
        parentmodule(typeof(op)) === (@__MODULE__) && !(op isa Union{PermutationOperator, _PermutedOperator})
    end
    n = rowwise ? _gradient_parameters(n_dim, layout) : length(θ)
    chunk = ForwardDiff.Chunk(min(_GRADIENT_CHUNK, n))
    diff = ForwardDiff.GradientConfig(_energy_gradient, view(θ, 1:n), chunk)
    return (; diff, rowwise, invariant = _hamiltonian_invariant(terms, layout.symmetry))
end

_gradient_residual(bra, ket, terms, invariant, energy) =
    _hamiltonian_element(bra, ket, terms; invariant) - energy * _compute_overlap_element(bra, ket)

# A function's parameters enter only its own row and column of H and S.
function _row_energy(x, i, n_dim, layout, basis, terms, invariant, c, energy)
    g = _decode_gradient_function(x, n_dim, layout)
    ci = c[i]
    total = real(conj(ci) * (_gradient_residual(g, g, terms, invariant, energy) * ci))
    for (j, h) in pairs(basis.functions)
        j == i && continue
        # Match the lower-triangle orientation used by matrix assembly.
        if j < i
            total += 2real(conj(ci) * (_gradient_residual(g, h, terms, invariant, energy) * c[j]))
        else
            total += 2real(conj(c[j]) * (_gradient_residual(h, g, terms, invariant, energy) * ci))
        end
    end
    return total
end

function _energy_gradient(θ, n, n_dim, terms, grad_cfg, state, bad_gradient, layout)
    basis = _decode_basis(θ, n, n_dim, layout)
    problem = _gradient_eigenproblem(basis, terms)
    problem === nothing && return Inf, zeros(Float64, length(θ))
    values, vectors, D = problem
    index = _target_index(state, values)
    energy, coefficients = values[index], vectors[:, index]
    isfinite(energy) || return Inf, zeros(Float64, length(θ))

    gradient = try
        if grad_cfg.rowwise
            # The derivative of D cancels in the eigenvalue derivative.
            c = D * coefficients
            n_per = _gradient_parameters(n_dim, layout)
            result = similar(θ)
            for i in eachindex(basis.functions)
                block = ((i - 1) * n_per + 1):(i * n_per)
                result[block] = ForwardDiff.gradient(view(θ, block), grad_cfg.diff, Val(false)) do x
                    _row_energy(x, i, n_dim, layout, basis, terms, grad_cfg.invariant, c, energy)
                end
            end
            result
        else
            ForwardDiff.gradient(θ, grad_cfg.diff, Val(false)) do θ_ad
                H, S, _ = _gradient_matrices(_decode_basis(θ_ad, n, n_dim, layout), terms)
                dot(coefficients, H * coefficients) - energy * dot(coefficients, S * coefficients)
            end
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

# Optimize the parameters θ of an n-function basis.  Returns the optimizer's
# (x, energy, gradient, evaluations, history); `prefix` labels verbose output.
function _optimize_parameters(θ, n, n_dim, ctx, layout, optimizer, prefix)
    grad_cfg = _gradient_config(θ, n_dim, layout, ctx.terms)
    bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    fg = x -> _energy_gradient(x, n, n_dim, ctx.terms, grad_cfg, ctx.state, bad_gradient, layout)
    report_iteration = function (x, energy, gradient, iteration)
        ctx.verbose && @info "$(prefix)iteration $iteration" energy gradnorm = norm(gradient)
        return x, energy, gradient
    end
    result = optimize(fg, θ, optimizer; finalize! = report_iteration)
    _check_optimization_result(result[1], result[2], bad_gradient)
    return result
end

function _gradient_solution(basis::BasisSet, ctx::_SolveCtx, alg, history, gradnorm, draws, notes...)
    problem = _gradient_eigenproblem(basis, ctx.terms)
    problem === nothing && throw(ArgumentError("optimized basis contains a null function or an ill-conditioned overlap"))
    values, vectors, D = problem
    ΔE = length(history) ≥ 2 ? history[end - 1] - history[end] : NaN
    report = _gradient_report(gradnorm, alg.optimizer.gradtol, ΔE, cond(Symmetric(build_overlap_matrix(basis))))
    foreach(note -> _add_note!(report, note), notes)
    return Solution(
        values, basis, D * vectors, ctx.terms, _require_available_state(ctx.state, length(values)),
        [StageResult(alg, history, report)], report, ctx.units, draws, ctx.hamiltonian
    )
end

_jacobi_dimension(ctx) = size(first(op for op in ctx.terms if op isa KineticOperator).K, 1)

function _gradient_stream(layout, draws = 0)
    rank = _gradient_rank(layout)
    stream = BasisState(_basis_type(rank, layout.symmetry); rank, symmetry = layout.symmetry)
    stream.draw = draws
    return stream
end

# Cold start: n quasi-random functions or, if they are ill-conditioned, the n
# most independent of 10n draws (pivoted Cholesky of the unit-norm overlap).
# With `widths` unset, `:wide` sampling is tried after `:linear`.
function _initial_gradient_basis(ctx, layout, n, scale, widths)
    n > 0 || throw(ArgumentError("GVM basis size must be positive"))
    stream = _gradient_stream(layout)
    draw(w) = _draw_candidate!(stream, scale, HaltonSample(), w, ctx.w_list)
    note(w) = stream.draw == n ? nothing : "cold GVM selected $n functions from $(stream.draw) draws (widths = :$w)"
    for sampling_widths in (widths === nothing ? (:linear, :wide) : (widths,))
        basis = BasisSet([draw(sampling_widths) for _ in 1:n])
        _gradient_eigenproblem(basis, ctx.terms) === nothing || return basis, stream.draw, note(sampling_widths)

        pool = filter(_admissible_gradient_function, [basis.functions; [draw(sampling_widths) for _ in 1:(9n)]])
        length(pool) ≥ n || continue
        S = build_overlap_matrix(BasisSet(pool))
        D = Diagonal(1 ./ sqrt.(diag(S)))
        pivoted = cholesky(Symmetric(D * S * D), RowMaximum(); check = false)
        pivoted.rank ≥ n || continue
        basis = BasisSet(pool[pivoted.p[1:n]])
        _gradient_eigenproblem(basis, ctx.terms) === nothing || return basis, stream.draw, note(sampling_widths)
    end
    throw(ArgumentError("could not initialize $n independent Gaussians; try wider widths, fewer functions, or an SVM warm start, and check rank and exchange signs"))
end

function _solve(
        terms, masses, alg::GVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing, symmetry = nothing
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian, symmetry)
    layout = _resolve_gradient_layout(alg, init, ctx)
    n_dim = _jacobi_dimension(ctx)

    scale_note = initial_note = nothing
    if init === nothing
        alg.basis === nothing && throw(
            ArgumentError("cold GVM requires `basis`; use GVM(basis = n)")
        )
        n = something(alg.basis)
        scale, scale_note = _solve_scale(something(alg.scale, :adaptive), ctx, nothing; rank = _gradient_rank(layout))
        basis, draws, initial_note = _initial_gradient_basis(ctx, layout, n, scale, alg.widths)
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
    _require_available_state(ctx.state, n)

    x, _, gradient, _, history = _optimize_parameters(θ, n, n_dim, ctx, layout, alg.optimizer, "GVM: ")
    return _gradient_solution(
        _decode_basis(x, n, n_dim, layout), ctx, alg, accumulate(min, history[:, 1]), norm(gradient), draws,
        initial_note, scale_note
    )
end

function _solve(
        terms, masses, alg::DynamicGVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing, symmetry = nothing
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian, symmetry)
    layout = _resolve_gradient_layout(alg, init, ctx)
    _require_available_state(ctx.state, alg.basis)
    scale, scale_note = _solve_scale(alg.scale, ctx, init; rank = _gradient_rank(layout))
    n_dim = _jacobi_dimension(ctx)
    θ = Float64[]
    initial_size = 0
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

    stream = _gradient_stream(layout, init === nothing ? 0 : init.draws)
    energy_history = Float64[]
    gradnorm = NaN
    for step in (initial_size + 1):alg.basis
        best_energy, best_candidate = Inf, nothing
        for _ in 1:alg.candidates
            g = _draw_candidate!(stream, scale, HaltonSample(), alg.widths, ctx.w_list)
            candidate = _encode_basis(BasisSet([g]), layout)
            problem = _gradient_eigenproblem(_decode_basis([θ; candidate], step, n_dim, layout), ctx.terms)
            problem === nothing && continue
            values = first(problem)
            energy = values[_target_index(ctx.state, values)]
            if energy < best_energy
                best_energy, best_candidate = energy, candidate
            end
        end
        best_candidate === nothing && throw(
            ArgumentError(
                "DynamicGVM step $step: all $(alg.candidates) candidates give a null function or an " *
                    "ill-conditioned overlap; try a smaller `scale` or more `candidates`"
            )
        )

        append!(θ, best_candidate)
        θ, energy, gradient, evaluations, _ = _optimize_parameters(
            θ, step, n_dim, ctx, layout, alg.optimizer, "DynamicGVM: step $step/$(alg.basis), "
        )
        gradnorm = norm(gradient)
        push!(energy_history, energy)
        verbose && @info "DynamicGVM: completed step $step/$(alg.basis)" energy evaluations
    end

    return _gradient_solution(
        _decode_basis(θ, alg.basis, n_dim, layout), ctx, alg, energy_history, gradnorm, stream.draw, scale_note
    )
end
