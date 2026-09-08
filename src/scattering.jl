using SpecialFunctions: logabsgamma, digamma

function _channel_gaussian(A, ℓ, w)
    s = zeros(length(w))
    ℓ == 0 && return Rank0Gaussian(A, s)
    ℓ == 1 && return Rank1Gaussian(A, w, s)
    ℓ == 2 && return Rank2Gaussian(A, hcat(w, s, s), hcat(s, w, s), s)
    throw(ArgumentError("ℓ must be 0, 1, or 2"))
end

"""
    ScatteringChannel(fragment_a, fragment_b;
        ℓ, threshold, interaction_range, inelastic_threshold=nothing)

Describe a neutral, central, short-range elastic scattering channel between two
nonempty, disjoint fragments. Supported orbital angular momenta are `ℓ = 0:2`.
`threshold` is the elastic threshold, `interaction_range` is a positive
characteristic range, and `inelastic_threshold`, when supplied, must lie
strictly above `threshold`.

Use consistent units with `ℏ=1`: thresholds are energies and
`interaction_range` is a length. With `Operators(masses)`, the fragments must
partition all particles, whose masses must be positive. The caller supplies
the composite-fragment threshold and ensures scalar internal fragment states
and a central, short-range elastic interaction; these properties cannot all be
inferred from arbitrary radial potentials. Invalid channel inputs raise
`ArgumentError`. See [Scattering from trapped spectra](@ref) for a complete example.
"""
struct ScatteringChannel
    fragments::Tuple{Vector{Int}, Vector{Int}}
    ℓ::Int
    threshold::Float64
    interaction_range::Float64
    inelastic_threshold::Union{Nothing, Float64}

    function ScatteringChannel(
            fragment_a,
            fragment_b;
            ℓ,
            threshold,
            interaction_range,
            inelastic_threshold = nothing,
        )
        ℓ in 0:2 || throw(ArgumentError("ℓ must be 0, 1, or 2"))
        A, B = collect(Int, fragment_a), collect(Int, fragment_b)
        isempty(A) && throw(ArgumentError("fragment_a must be nonempty"))
        isempty(B) && throw(ArgumentError("fragment_b must be nonempty"))
        all(>(0), A) && all(>(0), B) ||
            throw(ArgumentError("fragment indices must be positive"))
        allunique(A) && allunique(B) ||
            throw(ArgumentError("fragment indices must be unique"))
        isempty(intersect(A, B)) || throw(ArgumentError("fragments must be disjoint"))

        elastic = Float64(threshold)
        isfinite(elastic) || throw(ArgumentError("threshold must be finite"))
        range = Float64(interaction_range)
        isfinite(range) && range > 0 ||
            throw(ArgumentError("interaction_range must be finite and positive"))
        inelastic = isnothing(inelastic_threshold) ? nothing : Float64(inelastic_threshold)
        if !isnothing(inelastic)
            isfinite(inelastic) ||
                throw(ArgumentError("inelastic_threshold must be finite"))
            inelastic > elastic || throw(
                ArgumentError("inelastic_threshold must be above threshold")
            )
        end

        return new((A, B), Int(ℓ), elastic, range, inelastic)
    end
end

function _channel_kinematics(ops::Operators, channel::ScatteringChannel)
    ops.masses === nothing &&
        throw(ArgumentError("scattering requires Operators(masses)"))
    masses = ops.masses
    A, B = channel.fragments
    sort(vcat(A, B)) == collect(eachindex(masses)) ||
        throw(ArgumentError("fragments must partition all particles"))
    all(>(0), masses) || throw(ArgumentError("all particle masses must be positive"))
    MA, MB = sum(masses[A]), sum(masses[B])
    d = zeros(Float64, length(masses))
    d[A] .= masses[A] ./ MA
    d[B] .= -masses[B] ./ MB
    return (; reduced_mass = MA * MB / (MA + MB), weight = ops._U' * d)
end

function _particle_pair(ops::Operators, w::AbstractVector)
    masses = ops.masses
    U = ops._U
    length(w) == size(U, 2) || return nothing
    all(isfinite, w) || return nothing
    weight_norm = norm(w)
    isfinite(weight_norm) || return nothing
    difference = zeros(Float64, length(masses))
    matches = Tuple{Float64, Tuple{Int, Int}, Float64}[]
    for i in firstindex(masses):(lastindex(masses) - 1), j in (i + 1):lastindex(masses)
        fill!(difference, 0.0)
        difference[i] = 1.0
        difference[j] = -1.0
        candidate = U' * difference
        candidate_norm = norm(candidate)
        residual_minus = norm(w - candidate)
        residual_plus = norm(w + candidate)
        scale = max(weight_norm, candidate_norm)
        tolerance = sqrt(eps(Float64)) * scale
        all(isfinite, (candidate_norm, residual_minus, residual_plus, tolerance)) ||
            continue
        if residual_minus ≤ tolerance || residual_plus ≤ tolerance
            push!(matches, (min(residual_minus, residual_plus), (i, j), scale))
        end
    end
    isempty(matches) && return nothing
    sort!(matches; by = first)
    residual, pair, scale = first(matches)
    if length(matches) > 1
        next_residual, _, next_scale = matches[2]
        # A unique exact match retains its identity. Approximate distances
        # carry roundoff at the weight scale, not at the tiny residual scale.
        ambiguous = iszero(residual) ? iszero(next_residual) :
            next_residual - residual ≤ 8eps(Float64) * max(scale, next_scale)
        ambiguous && return nothing
    end
    return pair
end

function _validate_scattering_hamiltonian(ops::Operators, channel::ScatteringChannel; check_channel = true)
    check_channel && _channel_kinematics(ops, channel)
    count(op -> op isa KineticOperator, ops) == 1 ||
        throw(ArgumentError("scattering requires exactly one KineticOperator"))
    K = only(op.K for op in ops if op isa KineticOperator)
    expected = Λ(ops.masses)
    size(K) == size(expected) && all(isfinite, K) && issymmetric(K) &&
        isapprox(K, expected; rtol = 64eps(Float64), atol = 0) || throw(
        ArgumentError("kinetic matrix must be finite, symmetric, and match Λ(ops.masses) in size and values; use KineticOperator(ops.masses)")
    )

    A, B = channel.fragments
    for op in ops
        if op isa KineticOperator
            continue
        elseif op isa CoulombOperator
            iszero(op.coefficient) && continue
            pair = _particle_pair(ops, op.w)
            isnothing(pair) && throw(
                ArgumentError("cannot identify the particle pair for CoulombOperator")
            )
            i, j = pair
            (i in A && j in A) || (i in B && j in B) || throw(
                ArgumentError("cross-fragment Coulomb interactions are not supported")
            )
        elseif op isa OscillatorOperator
            pair = _particle_pair(ops, op.w)
            isnothing(pair) && throw(
                ArgumentError("cannot identify the particle pair for OscillatorOperator")
            )
            i, j = pair
            (i in A && j in A) || (i in B && j in B) || throw(
                ArgumentError("cross-fragment oscillator interactions are not supported")
            )
        elseif op isa GaussianOperator
            iszero(op.coefficient) && continue
            isfinite(op.coefficient) || throw(
                ArgumentError("GaussianOperator coefficient must be finite")
            )
            isfinite(op.γ) && op.γ > 0 || throw(
                ArgumentError("GaussianOperator γ must be finite and positive")
            )
            length(op.w) == size(ops._U, 2) &&
                all(isfinite, op.w) && any(!iszero, op.w) || throw(
                ArgumentError("GaussianOperator weight must be finite and nonzero")
            )
        elseif op isa Union{NumericalPotential, ManyBodyGaussianOperator}
            continue
        else
            throw(ArgumentError("operator $(typeof(op)) is not supported for scattering"))
        end
    end
    return nothing
end

"""
    TrapSpectrum(channel, trap_lengths, energies; reduced_mass, ...)

Record absolute trapped eigenenergies and the kinematics and solver diagnostics
needed for scattering inference. `trap_lengths` use
`b = sqrt(ℏ / (2μω))`, a factor `sqrt(2)` smaller than the conventional
oscillator length. Rows of `energies` correspond to trap lengths and columns to
the consecutive `levels = 1:n`.

The keyword constructor accepts independently computed spectra in consistent
`ℏ=1` units. It subtracts `channel.threshold` to obtain `relative_energies` and
derives `frequencies` and `range_ratios`. `reduced_mass` and lengths must be
finite and positive. `converged` and `energy_changes` match the energy matrix;
`condition_numbers` and `basis_sizes` have one entry per trap. Invalid shapes
raise `DimensionMismatch`; invalid values or levels raise `ArgumentError`.
Converged energies must lie below a supplied inelastic threshold.

The default `converged=trues(size(energies))` is the caller's assertion of
convergence, not an independent check. Missing changes and conditions default
to `NaN`, sizes to zero, `tolerance=NaN`, `window=0`, and `algorithm=nothing`.
`channel_weight` defaults to an empty vector when Jacobi kinematics are unknown.
"""
struct TrapSpectrum
    channel::ScatteringChannel
    reduced_mass::Float64
    channel_weight::Vector{Float64}
    trap_lengths::Vector{Float64}
    range_ratios::Vector{Float64}
    frequencies::Vector{Float64}
    levels::UnitRange{Int}
    energies::Matrix{Float64}
    relative_energies::Matrix{Float64}
    converged::BitMatrix
    energy_changes::Matrix{Float64}
    condition_numbers::Vector{Float64}
    basis_sizes::Vector{Int}
    tolerance::Float64
    window::Int
    algorithm::Union{Nothing, SolverMethod}
end

function TrapSpectrum(
        channel::ScatteringChannel,
        trap_lengths,
        energies;
        reduced_mass,
        channel_weight = Float64[],
        levels = 1:size(energies, 2),
        converged = trues(size(energies)),
        energy_changes = fill(NaN, size(energies)),
        condition_numbers = fill(NaN, length(trap_lengths)),
        basis_sizes = zeros(Int, length(trap_lengths)),
        tolerance = NaN,
        window = 0,
        algorithm = nothing,
    )
    mass = Float64(reduced_mass)
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError("reduced_mass must be finite and positive"))

    lengths = collect(Float64, trap_lengths)
    all(x -> isfinite(x) && x > 0, lengths) ||
        throw(ArgumentError("trap_lengths must be finite and positive"))
    energy_matrix = Matrix{Float64}(energies)
    ntraps, nlevels = size(energy_matrix)
    ntraps == length(lengths) || throw(
        DimensionMismatch("energies must have one row per trap length")
    )
    levels isa UnitRange || throw(ArgumentError("levels must be a UnitRange"))
    levels == 1:nlevels || throw(ArgumentError("levels must equal 1:$nlevels"))

    size(converged) == size(energy_matrix) || throw(
        DimensionMismatch("converged must match the energies shape")
    )
    size(energy_changes) == size(energy_matrix) || throw(
        DimensionMismatch("energy_changes must match the energies shape")
    )
    length(condition_numbers) == ntraps || throw(
        DimensionMismatch("condition_numbers must have one entry per trap length")
    )
    length(basis_sizes) == ntraps || throw(
        DimensionMismatch("basis_sizes must have one entry per trap length")
    )

    tolerance_value = Float64(tolerance)
    isnan(tolerance_value) || isfinite(tolerance_value) && tolerance_value > 0 ||
        throw(ArgumentError("tolerance must be NaN or finite and positive"))
    window isa Integer && window ≥ 0 ||
        throw(ArgumentError("window must be a nonnegative integer"))
    algorithm isa Union{Nothing, SolverMethod} ||
        throw(ArgumentError("algorithm must be nothing or a SolverMethod"))

    convergence_matrix = BitMatrix(converged)
    if !isnothing(channel.inelastic_threshold)
        for i in eachindex(energy_matrix, convergence_matrix)
            convergence_matrix[i] &&
                !(energy_matrix[i] < channel.inelastic_threshold) &&
                throw(ArgumentError("converged energies must be below inelastic_threshold"))
        end
    end

    range_ratios = lengths ./ channel.interaction_range
    frequencies = 1 ./ (2 .* mass .* lengths .^ 2)
    relative_energies = energy_matrix .- channel.threshold
    return TrapSpectrum(
        channel,
        mass,
        collect(Float64, channel_weight),
        lengths,
        range_ratios,
        frequencies,
        Int(first(levels)):Int(last(levels)),
        energy_matrix,
        relative_energies,
        convergence_matrix,
        Matrix{Float64}(energy_changes),
        collect(Float64, condition_numbers),
        collect(Int, basis_sizes),
        tolerance_value,
        Int(window),
        algorithm,
    )
end

function _trapped_levels(st, levels)
    values = fill(NaN, length(levels))
    available = 1:min(last(levels), length(st.eig.ε))
    values[available] = st.eig.ε[available]
    return values
end

function _trapped_stage(st, alg::SVM, ctx, levels, make_gaussian)
    scale = _resolve_scale(alg.scale, ctx.masses)
    histories = [Float64[] for _ in levels]
    for iteration in 1:alg.basis
        bestE, best, bestcols = Inf, nothing, nothing
        for _ in 1:alg.candidates
            cand = _draw_candidate!(st, scale, alg.sampler, ctx.w_list, make_gaussian)
            cols = _candidate_columns(cand, st.basis, ctx.terms)
            cols === nothing && continue
            E = _score_candidate_sum(st.eig, cols...; levels, min_resid_ratio = alg.indep_tol)
            E === nothing && continue
            if E < bestE
                bestE, best, bestcols = E, cand, cols
            end
        end
        accepted = best !== nothing && commit!(st, best, bestcols) !== nothing
        if accepted
            for level in 1:min(last(levels), length(st.eig.ε))
                push!(histories[level], st.eig.ε[level])
            end
        end
        if ctx.verbose
            energy = nfuns(st) == 0 ? NaN : sum(st.eig.ε[1:min(last(levels), end)])
            @info "SVM: iteration $iteration/$(alg.basis)" energy basis = nfuns(st) accepted
        end
    end
    changes = [
        length(hist) > ctx.window ? hist[end - ctx.window] - hist[end] : NaN
            for hist in histories
    ]
    return st, changes
end

function _trapped_stage(st, alg::Refine, ctx, levels, make_gaussian)
    scale = _resolve_scale(alg.scale, ctx.masses)
    changes = fill(NaN, length(levels))
    for iteration in 1:alg.sweeps
        before = _trapped_levels(st, levels)
        for _ in 1:nfuns(st)
            # Winners are appended, so visit the first remaining member to
            # reconsider each starting basis slot exactly once per sweep.
            i = firstindex(st.basis)
            base = rebuild_without(st, i)
            idx = setdiff(1:nfuns(st), i)
            current = (st.S[idx, i], st.H[idx, i], st.S[i, i], st.H[i, i])
            bestE = something(_score_candidate_sum(base.eig, current...; levels), Inf)
            best, bestcols = st.basis[i], current
            for _ in 1:alg.candidates
                cand = _draw_candidate!(base, scale, alg.sampler, ctx.w_list, make_gaussian)
                cols = _candidate_columns(cand, base.basis, ctx.terms)
                cols === nothing && continue
                E = _score_candidate_sum(base.eig, cols...; levels, min_resid_ratio = alg.indep_tol)
                E === nothing && continue
                if E < bestE - 1.0e-12
                    bestE, best, bestcols = E, cand, cols
                end
            end
            commit!(base, best, bestcols) === nothing &&
                error("linearly dependent basis while restoring a trapped refinement slot")
            st = base
        end
        changes = before - _trapped_levels(st, levels)
        if ctx.verbose
            energy = nfuns(st) == 0 ? NaN : sum(st.eig.ε[1:min(last(levels), end)])
            @info "Refine: iteration $iteration/$(alg.sweeps)" energy
        end
    end
    return st, changes
end

"""
    trapped_spectrum(ops::Operators, channel, trap_lengths, algorithm=SVM();
        levels=1:2, tol=1e-4, window=20, verbose=false)

Solve independent harmonic traps in the definite orbital-angular-momentum
channel `ℓ = 0, 1, 2`, retaining all requested consecutive levels `1:n` in one
common basis. `trap_lengths` use `b = sqrt(1/(2μω))`; the added potential is
`R²/(8μb⁴)`. Input trap order is preserved.

The stochastic objective is the equal-weight sum of the selected energies.
Use `SVM` or a pipeline beginning with `SVM` and containing only `SVM` and
`Refine`. Each stage requires positive basis/sweep and candidate counts,
`scale=:auto` or a finite positive numeric scale, and `0 < indep_tol < 1`.
Default Halton sampling is deterministic. SVM convergence compares
each level with its value `window` accepted additions earlier; refinement uses
the final sweep. These changes indicate numerical saturation, not error bounds.
`tol` must be finite and positive, and `window` must be a nonnegative integer.
An entire trap row is fit-eligible only if all requested levels exist and have
`0 ≤ ΔE < tol`. Missing energies and unavailable changes are `NaN`; unconverged
energies, overlap condition numbers, and basis sizes remain in [`TrapSpectrum`](@ref).
Invalid or unsupported inputs raise `ArgumentError`.
"""
function trapped_spectrum(
        ops::Operators, channel::ScatteringChannel, trap_lengths,
        algorithm::SolverMethod = SVM();
        levels = 1:2, tol = 1.0e-4, window = 20, verbose = false,
    )::TrapSpectrum
    levels isa UnitRange && first(levels) == 1 && !isempty(levels) ||
        throw(ArgumentError("levels must be a nonempty UnitRange 1:n"))
    selected = 1:Int(last(levels))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    window isa Integer && window ≥ 0 ||
        throw(ArgumentError("window must be a nonnegative integer"))
    lengths = collect(Float64, trap_lengths)
    all(b -> isfinite(b) && b > 0, lengths) ||
        throw(ArgumentError("trap_lengths must be finite and positive"))
    stages = algorithm isa Pipeline ? algorithm.stages : (algorithm,)
    isempty(stages) && throw(ArgumentError("empty trapped pipeline; start with SVM"))
    first(stages) isa SVM || throw(
        ArgumentError("trapped spectra require an initial SVM stage; use SVM() → Refine() for refinement")
    )
    all(stage -> stage isa Union{SVM, Refine}, stages) || throw(
        ArgumentError("trapped spectra support only SVM and Refine; remove gradient or other unsupported stages")
    )
    for stage in stages
        if stage isa SVM
            stage.basis > 0 || throw(ArgumentError("SVM basis must be positive"))
        else
            stage.sweeps > 0 || throw(ArgumentError("Refine sweeps must be positive"))
        end
        stage.candidates > 0 || throw(ArgumentError("$(typeof(stage)) candidates must be positive"))
        stage.scale === :auto ||
            stage.scale isa Real && isfinite(stage.scale) && stage.scale > 0 ||
            throw(ArgumentError("$(typeof(stage)) scale must be :auto or a finite positive number"))
        isfinite(stage.indep_tol) && 0 < stage.indep_tol < 1 ||
            throw(ArgumentError("$(typeof(stage)) indep_tol must be finite and strictly between 0 and 1"))
    end
    kinematics = _channel_kinematics(ops, channel)
    _validate_scattering_hamiltonian(ops, channel; check_channel = false)
    μ, w = kinematics.reduced_mass, kinematics.weight
    ntraps, nlevels = length(lengths), length(selected)
    E = fill(NaN, ntraps, nlevels)
    changes = fill(NaN, ntraps, nlevels)
    eligible = falses(ntraps, nlevels)
    conditions = fill(NaN, ntraps)
    sizes = zeros(Int, ntraps)
    make_gaussian = A -> _channel_gaussian(A, channel.ℓ, w)
    G = typeof(make_gaussian(Matrix{Float64}(I, length(w), length(w))))
    for (i, b) in enumerate(lengths)
        terms = filter(ops.terms) do op
            !(op isa Union{CoulombOperator, GaussianOperator} && iszero(op.coefficient))
        end
        push!(terms, OscillatorOperator(1 / (8μ * b^4), w))
        ctx = _ctx(terms, ops.masses; state = last(selected), tol, window, verbose)
        # Many-body potentials need not expose pair weights. Complete any
        # missing sampling directions using physical particle separations.
        if rank(hcat(ctx.w_list...)) < ctx.d
            U = ops._U
            for particle in firstindex(ops.masses):(lastindex(ops.masses) - 1)
                push!(ctx.w_list, U[particle, :] - U[end, :])
            end
        end
        st = BasisState(G)
        for stage in stages
            st, ΔE = _trapped_stage(st, stage, ctx, selected, make_gaussian)
            changes[i, :] = ΔE
        end
        E[i, :] = _trapped_levels(st, selected)
        eligible[i, :] .= all(isfinite, E[i, :]) && all(x -> 0 ≤ x < tol, changes[i, :])
        sizes[i] = nfuns(st)
        conditions[i] = nfuns(st) == 0 ? NaN : cond(Symmetric(st.S))
    end
    return TrapSpectrum(
        channel, lengths, E; reduced_mass = μ, channel_weight = w, levels = selected,
        converged = eligible, energy_changes = changes, condition_numbers = conditions,
        basis_sizes = sizes, tolerance = tol, window, algorithm
    )
end

"""
    scattering_parameters(ops, channel, trap_lengths, algorithm=SVM();
        levels=1:2, ere_order=2, tol=1e-4, window=20, verbose=false)

Compose [`trapped_spectrum`](@ref) and [`fit_scattering_parameters`](@ref).
`ere_order = 0, 1, 2` fits through the constant, `k²`, or `k⁴` term,
respectively. The returned fit retains its source spectrum and diagnostics.
"""
function scattering_parameters(
        ops, channel, trap_lengths, algorithm::SolverMethod = SVM();
        levels = 1:2, ere_order = 2, tol = 1.0e-4, window = 20, verbose = false,
    )
    spectrum = trapped_spectrum(ops, channel, trap_lengths, algorithm; levels, tol, window, verbose)
    return fit_scattering_parameters(spectrum; ere_order)
end

function _trap_K(ℓ, E, μ, b)
    ω = 1 / (2μ * b^2)
    znum = 3 / 4 + ℓ / 2 - E / (2ω)
    zden = 1 / 4 - ℓ / 2 - E / (2ω)
    znum ≤ 0 && isinteger(znum) && throw(
        ArgumentError(
            "energy lies at a quantization numerator pole; use energies away from trap poles"
        )
    )
    zden ≤ 0 && isinteger(zden) && return 0.0
    lnum, snum = logabsgamma(znum)
    lden, sden = logabsgamma(zden)
    logmag = (ℓ + 1 / 2) * log(2) - (2ℓ + 1) * log(b) + lnum - lden
    sgn = (isodd(ℓ + 1) ? -1.0 : 1.0) * snum * sden
    return sgn * exp(logmag)
end

function _quantization_pole_distance(ℓ, E, μ, b)
    ω = 1 / (2μ * b^2)
    offset = E / (2ω) - (3 / 4 + ℓ / 2)
    return abs(offset - max(0.0, round(offset)))
end

"""
    ScatteringParameters

Record an effective-range fit of `K = c₀ + c₂ k² + c₄ k⁴`. The derived parameters are
`a = -1/c₀`, `r = 2c₂`, and `shape = c₄`; omitted terms are `nothing`.
Their dimensions are `L^(2ℓ+1)`, `L^(1-2ℓ)`, and `L^(3-2ℓ)`, respectively;
`coefficients` stores the retained polynomial coefficients in ascending order.
An exactly zero `c₀` uses `a = -Inf`, independent of floating-point zero sign.
`k2` and `K` retain the spectrum shape, while unused residuals and diagnostics
are `NaN`. Covariance describes regression residuals, not physical uncertainty.
`K` is also `NaN` for nonfinite energies and unused exact numerator poles.
Rows of `window_parameters` match `window_starts`; columns are `a`, `r`, and
`shape`, with `NaN` for omitted terms or rank-deficient windows.
"""
struct ScatteringParameters
    ℓ::Int
    coefficients::Vector{Float64}
    a::Float64
    r::Union{Nothing, Float64}
    shape::Union{Nothing, Float64}
    used::BitMatrix
    k2::Matrix{Float64}
    K::Matrix{Float64}
    residuals::Matrix{Float64}
    design_rank::Int
    design_condition::Float64
    covariance::Union{Nothing, Matrix{Float64}}
    pole_distance::Matrix{Float64}
    K_energy_sensitivity::Matrix{Float64}
    window_starts::Vector{Float64}
    window_parameters::Matrix{Float64}
    warnings::Vector{Pair{Symbol, String}}
    spectrum::TrapSpectrum
end

function _ere_fit(x, y, p)
    length(x) ≥ p || return nothing
    xscale = p == 1 ? 1.0 : maximum(abs, x)
    xscale > 0 || return nothing
    z = x ./ xscale
    X = hcat((z .^ j for j in 0:(p - 1))...)
    F = qr(X)
    R = Matrix(F.R)
    rnk = rank(R)
    rnk == p || return nothing
    βscaled = F \ y
    coefficients = [βscaled[j + 1] / xscale^j for j in 0:(p - 1)]
    residual = y - X * βscaled
    dof = length(y) - p
    covariance = if dof > 0
        σ² = sum(abs2, residual) / dof
        Rinv = UpperTriangular(R) \ Matrix{Float64}(I, p, p)
        cov_scaled = σ² * Rinv * Rinv'
        D = Diagonal([xscale^(-j) for j in 0:(p - 1)])
        D * cov_scaled * D
    else
        nothing
    end
    return (; coefficients, residual, rnk, condition = cond(R), covariance)
end

"""
    fit_scattering_parameters(spectrum::TrapSpectrum; ere_order=2)

Fit the real trap quantization function using converged energies relative to
the elastic threshold. Orders `0`, `1`, and `2` retain powers through `1`,
`k²`, and `k⁴`, respectively. Uses scaled QR; singular or undersampled fits
raise `ArgumentError`. Exactly determined fits have no covariance.
Return [`ScatteringParameters`](@ref), retaining the source spectrum and a
`used` mask that excludes unconverged entries. A used energy at a quantization
numerator pole, a nonfinite quantization value, or an energy at or above a
supplied inelastic threshold raises `ArgumentError`.

Warnings are diagnostics, not physical-validity cuts: `:near_quantization_pole`
means dimensionless pole distance `< 1e-4`; `:energy_sensitive` means the
propagated K change exceeds `max(abs(residual), sqrt(eps())*max(1,abs(K)))`;
`:ill_conditioned_fit` means scaled design condition `> 1/sqrt(eps())`;
`:marginal_convergence` means a used absolute energy change exceeds half the
available tolerance. `:unstable_trap_window` means the range of one parameter
over the final three viable windows exceeds 10% of its maximum absolute value.
Unequal estimates involving infinities also trigger this warning; identical
infinities do not.
Windows start at each distinct trap length in ascending order; those with too
few points or insufficient rank retain `NaN` parameters. Missing energy-change
data yields `NaN` sensitivity.
"""
function fit_scattering_parameters(spectrum::TrapSpectrum; ere_order = 2)
    ere_order isa Integer && ere_order in 0:2 ||
        throw(ArgumentError("ere_order must be 0, 1, or 2"))
    p = ere_order + 1
    ℓ, μ = spectrum.channel.ℓ, spectrum.reduced_mass
    used = copy(spectrum.converged)
    k2 = 2μ .* spectrum.relative_energies
    K = fill(NaN, size(k2))
    inelastic = spectrum.channel.inelastic_threshold
    for index in CartesianIndices(K)
        used[index] && !isnothing(inelastic) &&
            !(spectrum.energies[index] < inelastic) &&
            throw(ArgumentError("fitted energies must be below inelastic_threshold"))
        E = spectrum.relative_energies[index]
        if isfinite(E)
            b = spectrum.trap_lengths[index[1]]
            !used[index] && iszero(_quantization_pole_distance(ℓ, E, μ, b)) && continue
            K[index] = _trap_K(ℓ, E, μ, b)
        end
        used[index] && !(isfinite(k2[index]) && isfinite(K[index])) &&
            throw(ArgumentError("converged energies must give finite k² and quantization values"))
    end
    count(used) ≥ p || throw(ArgumentError("need at least $p converged points for this effective-range fit"))
    fit = _ere_fit(k2[used], K[used], p)
    isnothing(fit) && throw(ArgumentError("effective-range design matrix is rank deficient"))
    coefficients = fit.coefficients
    residuals = fill(NaN, size(K))
    residuals[used] = fit.residual
    pole_distance = fill(NaN, size(K))
    sensitivity = fill(NaN, size(K))
    for index in CartesianIndices(K)
        used[index] || continue
        E, b = spectrum.relative_energies[index], spectrum.trap_lengths[index[1]]
        pole_distance[index] = _quantization_pole_distance(ℓ, E, μ, b)
        derivative = _trap_K_energy_derivative(ℓ, E, μ, b, K[index])
        sensitivity[index] = abs(derivative * spectrum.energy_changes[index])
    end

    starts = sort(unique(spectrum.trap_lengths))
    window_parameters = fill(NaN, length(starts), 3)
    viable = Int[]
    for (i, start) in pairs(starts)
        mask = used .& (spectrum.trap_lengths .≥ start)
        window_fit = _ere_fit(k2[mask], K[mask], p)
        isnothing(window_fit) && continue
        c = window_fit.coefficients
        window_parameters[i, 1] = iszero(c[1]) ? -Inf : -1 / c[1]
        p ≥ 2 && (window_parameters[i, 2] = 2c[2])
        p == 3 && (window_parameters[i, 3] = c[3])
        push!(viable, i)
    end

    warnings = Pair{Symbol, String}[]
    any(<(1.0e-4), pole_distance[used]) && push!(
        warnings,
        :near_quantization_pole => "Used energies lie within 1e-4 of a quantization pole; inspect energy sensitivity."
    )
    any(index -> sensitivity[index] > max(abs(residuals[index]), sqrt(eps(Float64)) * max(1, abs(K[index]))), findall(used)) &&
        push!(warnings, :energy_sensitive => "Recorded energy changes propagate to K changes larger than fit residuals or numerical precision.")
    fit.condition > 1 / sqrt(eps(Float64)) && push!(
        warnings,
        :ill_conditioned_fit => "The scaled effective-range design is ill-conditioned; widen the energy range or reduce ere_order."
    )
    isfinite(spectrum.tolerance) && any(change -> abs(change) > spectrum.tolerance / 2, spectrum.energy_changes[used]) &&
        push!(warnings, :marginal_convergence => "Used energy changes exceed half the convergence tolerance; improve basis convergence.")
    if length(viable) ≥ 3
        for j in 1:p
            values = window_parameters[viable[(end - 2):end], j]
            scale = maximum(abs, values)
            unstable = if isinf(scale)
                !all(==(first(values)), values)
            else
                scale > 0 && maximum(values) - minimum(values) > 0.1scale
            end
            if unstable
                push!(warnings, :unstable_trap_window => "Parameters vary by more than 10% or have unequal infinite estimates over the final three viable trap windows; inspect the large-trap trend.")
                break
            end
        end
    end
    return ScatteringParameters(
        ℓ, coefficients, iszero(coefficients[1]) ? -Inf : -1 / coefficients[1],
        p ≥ 2 ? 2coefficients[2] : nothing,
        p == 3 ? coefficients[3] : nothing,
        used, k2, K, residuals, fit.rnk, fit.condition, fit.covariance,
        pole_distance, sensitivity, starts, window_parameters, warnings, spectrum,
    )
end

function _trap_K_energy_derivative(ℓ, E, μ, b, K)
    ω = 1 / (2μ * b^2)
    znum = 3 / 4 + ℓ / 2 - E / (2ω)
    zden = 1 / 4 - ℓ / 2 - E / (2ω)
    if zden ≤ 0 && isinteger(zden)
        # At K=0, differentiate reciprocal gamma: (1/Γ)'(-n)=(-1)^n n!.
        lnum, snum = logabsgamma(znum)
        lfactorial, _ = logabsgamma(1 - zden)
        logmag = (ℓ + 1 / 2) * log(2) - (2ℓ + 1) * log(b) + lnum + lfactorial - log(2ω)
        return (isodd(ℓ) ? -1.0 : 1.0) * snum * cospi(zden) * exp(logmag)
    end
    dlogK_dE = -(digamma(znum) - digamma(zden)) / (2ω)
    return K * dlogK_dE
end
