const SATURATION_CAVEAT =
    "basis saturation under this sampler — not a certificate of the exact eigenvalue"

"""
    ConvergenceReport

What a solver run can honestly certify.

- `converged::Bool`
- `criterion::Symbol` — `:saturation` (stochastic: ΔE over the last `window`
  additions below `tol`), `:stationarity` (gradient tolerance met),
  `:max_steps`, or `:early_stop`
- `ΔE::Float64` — tail energy change (Ha)
- `tol::Float64`, `window::Int` (0 for gradient methods)
- `gradnorm` — final gradient norm (`nothing` for stochastic methods)
- `cond_S::Float64` — final overlap condition number
- `notes::Vector{String}` — caveats and early-stop explanations
"""
struct ConvergenceReport
    converged::Bool
    criterion::Symbol
    ΔE::Float64
    tol::Float64
    window::Int
    gradnorm::Union{Nothing, Float64}
    cond_S::Float64
    notes::Vector{String}
end

"""
    StageResult(method, energies, report)

One pipeline stage: the method that ran, its per-step target-state energies,
and its convergence report.
"""
struct StageResult
    method::Method
    energies::Vector{Float64}
    report::ConvergenceReport
end

"""
    Solution

Result of [`solve`](@ref).  Fields: `E` (eigenvalues of the final basis,
ascending), `basis::BasisSet`, `coefficients` (generalized eigenvectors,
`cᵀSc = I`), `operators`, `state` (target eigenstate), `stages`
(length 1 unless a `Pipeline` ran), `convergence` (final report).
`sol.E₀` is the target-state energy `E[state]`.
"""
struct Solution
    E::Vector{Float64}
    basis::BasisSet
    coefficients::Matrix{Float64}
    operators::Vector{FewBodyHamiltonians.Operator}
    state::Int
    stages::Vector{StageResult}
    convergence::ConvergenceReport
end

function Base.getproperty(sol::Solution, s::Symbol)
    s === :E₀ && return getfield(sol, :E)[getfield(sol, :state)]
    return getfield(sol, s)
end
Base.propertynames(::Solution) = (fieldnames(Solution)..., :E₀)

"""
    converged(sol::Solution) -> Bool
    converged(report::ConvergenceReport) -> Bool
"""
converged(r::ConvergenceReport) = r.converged
converged(sol::Solution) = converged(getfield(sol, :convergence))

"""
    energies(sol::Solution)            -> Vector{Float64}
    energies(sol::Solution, i::Integer)

Per-step target-state energy history — concatenated across stages, or of
stage `i`.  Ready for plotting (see also `plot(sol)`).
"""
energies(sol::Solution) = reduce(vcat, (s.energies for s in getfield(sol, :stages)))
energies(sol::Solution, i::Integer) = getfield(sol, :stages)[i].energies

function _fmtE(x)
    # Format energy with sigdigits, preferring exponential for small numbers
    rounded = round(x, sigdigits = 8)
    s = string(rounded)
    # If we got a decimal like "0.0001" and original was in exp form, convert to exponential
    if abs(rounded) < 1.0e-3 && abs(rounded) > 0
        # Manually construct exponential notation
        exponent = floor(Int, log10(abs(rounded)))
        mantissa = rounded / (10.0^exponent)
        s = "$(round(mantissa, sigdigits = 2))e$(exponent)"
    end
    return s
end

function Base.show(io::IO, ::MIME"text/plain", r::ConvergenceReport)
    verdict = r.converged ? "✓" : "✗"
    desc = r.criterion === :saturation ?
        "$(verdict) saturated  ΔE = $(_fmtE(r.ΔE)) Ha over last $(r.window) additions (tol $(_fmtE(r.tol)))" :
        r.criterion === :stationarity ?
        "$(verdict) stationary  |∇E| = $(_fmtE(something(r.gradnorm, NaN))) (gtol $(_fmtE(r.tol)))" :
        r.criterion === :early_stop ? "✗ stopped early" : "✗ max steps reached"
    print(io, "ConvergenceReport: ", desc)
    for n in r.notes
        print(io, "\n  note: ", n)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sol::Solution)
    n = length(getfield(sol, :basis).functions)
    G = isempty(getfield(sol, :basis).functions) ? "Gaussian" :
        string(nameof(typeof(first(getfield(sol, :basis).functions))))
    r = getfield(sol, :convergence)
    println(
        io, "FewBodyECG solution — ", n, " × ", G, ", ",
        length(getfield(sol, :operators)), " operator terms"
    )
    println(io, "  method       ", join((s.method for s in getfield(sol, :stages)), " → "))
    println(io, "  E₀           ", _fmtE(sol.E₀), " Ha    (variational upper bound)")
    print(io, "  convergence  ")
    show(io, MIME"text/plain"(), r)
    println(io)
    if length(getfield(sol, :stages)) > 1
        chain = join(
            ("$(s.method): E→$(_fmtE(last(s.energies)))" for s in getfield(sol, :stages)),
            "  →  "
        )
        println(io, "  stages       ", chain)
    end
    print(
        io, "  conditioning cond(S) ≈ ", round(r.cond_S, sigdigits = 2),
        " — handled (whitened eigensolver)"
    )
    return nothing
end
