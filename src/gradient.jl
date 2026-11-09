using OptimKit: optimize

# ForwardDiff compiles the energy gradient once per chunk size, so a fixed
# chunk lets every system reuse one compiled gradient.  Larger chunks are no
# faster for ECG energies.  Parameter vectors shorter than one chunk are padded
# with unused entries, so small bases use the same chunk as large ones.
const _GRADIENT_CHUNK = 12

_padded_length(θ) = max(_GRADIENT_CHUNK, length(θ))

# A named tag (rather than `nothing`) orders the gradient's duals outside the
# `_ShiftTag` duals of polarized matrix elements, which nest inside it.
_gradient_config(θ) = ForwardDiff.GradientConfig(
    _energy_gradient, zeros(_padded_length(θ)), ForwardDiff.Chunk(_GRADIENT_CHUNK)
)

function _energy_gradient(θ, layout, n, terms, grad_cfg, state, regularization, bad_gradient)
    local energy::Float64, coefficients::Vector{Float64}
    try
        basis = _decode_basis(layout, θ, n)
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

    n_θ = length(θ)
    θ_padded = [θ; zeros(_padded_length(θ) - n_θ)]
    gradient = try
        ForwardDiff.gradient(θ_padded, grad_cfg, Val(false)) do θ_ad
            basis = _decode_basis(layout, view(θ_ad, 1:n_θ), n)
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
    gradient = gradient[1:n_θ]
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
    H = build_hamiltonian_matrix(basis, ctx.terms)
    S = build_overlap_matrix(basis)
    values, vectors = solve_generalized_eigenproblem(H, S)
    state = _require_available_state(ctx.state, length(values))
    return Solution(
        values, basis, vectors, ctx.terms,
        state, stages, last(stages).report, ctx.units, draws, ctx.hamiltonian
    )
end

# Selection state that draws candidates of `layout`'s rank and symmetry from the
# quasi-random stream, continuing after its first `draw` points.
function _gradient_draw_state(layout::_GradientLayout, draw::Integer)
    st = BasisState(
        _stochastic_type(layout.rank, layout.symmetry);
        rank = layout.rank, symmetry = layout.symmetry
    )
    st.draw = draw
    return st
end

# The next drawn candidate that the symmetry projection does not annihilate.
function _next_admissible!(st::BasisState, scale, widths, w_list; max_draws = 100)
    for _ in 1:max_draws
        cand = _draw_candidate!(st, scale, HaltonSample(), widths, w_list)
        _survives_projection(cand, real(_compute_overlap_element(cand, cand))) && return cand
    end
    throw(ArgumentError("no admissible projected candidates; check rank and exchange signs"))
end

function _solve(
        terms, masses, alg::GVM;
        state = 1, tol = 1.0e-4, window = 20, init = nothing, verbose = false,
        units, hamiltonian = nothing, symmetry = nothing
    )
    ctx = _ctx(terms, masses; state, tol, window, verbose, units, init, hamiltonian, symmetry)
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    layout = _gradient_layout(alg.rank, ctx.symmetry, n_dim; init)

    scale_note = nothing
    if init === nothing
        alg.basis === nothing && throw(
            ArgumentError("cold GVM requires `basis`; use GVM(basis = n)")
        )
        n = something(alg.basis)
        scale, scale_note = _solve_scale(something(alg.scale, :adaptive), ctx, nothing; rank = layout.rank)
        st = _gradient_draw_state(layout, 0)
        widths = something(alg.widths, :linear)
        functions = [_next_admissible!(st, scale, widths, ctx.w_list) for _ in 1:n]
        θ = _encode_basis(layout, BasisSet(functions))
        draws = st.draw
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
        θ = _encode_basis(layout, init.basis)
        draws = init.draws
    end

    grad_cfg = _gradient_config(θ)
    bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    fg = x -> _energy_gradient(
        x, layout, n, ctx.terms, grad_cfg, ctx.state, 1.0e-10, bad_gradient
    )
    report_iteration = function (x, energy, gradient, iteration)
        verbose && @info "GVM: iteration $iteration" energy gradnorm = norm(gradient)
        return x, energy, gradient
    end

    x, energy, gradient, _, history = optimize(
        fg, θ, alg.optimizer; finalize! = report_iteration
    )
    _check_optimization_result(x, energy, bad_gradient)

    basis = _decode_basis(layout, x, n)
    energy_history = accumulate(min, history[:, 1])
    ΔE = length(energy_history) ≥ 2 ? abs(energy_history[end - 1] - energy_history[end]) : NaN
    report = _gradient_report(
        norm(gradient), alg.optimizer.gradtol, ΔE,
        cond(Symmetric(build_overlap_matrix(basis)))
    )
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
    n_dim = size(first(op for op in terms if op isa KineticOperator).K, 1)
    layout = _gradient_layout(alg.rank, ctx.symmetry, n_dim; init)
    scale, scale_note = _solve_scale(alg.scale, ctx, init; rank = layout.rank)
    n_per = _n_params(layout)
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
        θ = _encode_basis(layout, init.basis)
    end

    initial_size = length(θ) ÷ n_per
    st = _gradient_draw_state(layout, init === nothing ? 0 : init.draws)
    energy_history = Float64[]
    gradnorm = NaN

    for step in (initial_size + 1):alg.basis
        best_energy = Inf
        best_candidate = Float64[]

        for _ in 1:alg.candidates
            cand = _next_admissible!(st, scale, alg.widths, ctx.w_list)
            candidate_parameters = _encode_basis(layout, BasisSet([cand]))
            trial_parameters = [θ; candidate_parameters]
            try
                basis = _decode_basis(layout, trial_parameters, step)
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
        grad_cfg = _gradient_config(θ)
        bad_gradient = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        fg = x -> _energy_gradient(
            x, layout, step, ctx.terms, grad_cfg, ctx.state, 1.0e-10, bad_gradient
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
    basis = _decode_basis(layout, θ, n_built)
    ΔE = length(energy_history) ≥ 2 ? energy_history[end - 1] - energy_history[end] : NaN
    report = _gradient_report(
        gradnorm, alg.optimizer.gradtol, ΔE,
        cond(Symmetric(build_overlap_matrix(basis)))
    )
    return _solution_from_basis(
        basis, ctx, [StageResult(alg, energy_history, _add_note!(report, scale_note))], st.draw
    )
end
