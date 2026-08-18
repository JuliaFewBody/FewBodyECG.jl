"""
    SolverMethod

Abstract supertype of all solver algorithms.  A method is a small struct of
algorithm-level options; problem-level options (`state`, `tol`, `window`,
`init`, `verbose`) live on [`solve`](@ref).  Adding a new method = defining a
new subtype plus `solve`/`step!` methods — pure multiple dispatch.
"""
abstract type SolverMethod end

"""
    StochasticMethod <: SolverMethod

Abstract supertype for stochastic basis-selection methods such as [`SVM`](@ref)
and [`Refine`](@ref).
"""
abstract type StochasticMethod <: SolverMethod end

"""
    GradientMethod <: SolverMethod

Abstract supertype for gradient-optimization methods such as [`GVM`](@ref) and
[`DynamicGVM`](@ref).
"""
abstract type GradientMethod <: SolverMethod end

"""
    SVM(basis; candidates = 25, scale = :auto, sampler = HaltonSample(), indep_tol = 1e-4)

Suzuki–Varga stochastic selection (Sect. 4.2.5).  At each of `basis` steps,
`candidates` quasi-random Gaussians are drawn and scored in O(k²) by the
incremental whitened eigensolver; the best admissible one is committed.
`candidates = 1` is the accept-first strategy.  `scale = :auto` resolves via
[`default_scale`](@ref) from the system's masses.
"""
Base.@kwdef struct SVM <: StochasticMethod
    basis::Int = 50
    candidates::Int = 25
    scale::Union{Float64, Symbol} = :auto
    sampler::Any = HaltonSample()
    indep_tol::Float64 = 1.0e-4
end
SVM(basis::Int; kw...) = SVM(; basis, kw...)

"""
    Refine(sweeps; candidates = 25, scale = :auto, sampler = HaltonSample(), indep_tol = 1e-4)

Suzuki–Varga cyclic refinement (Sect. 4.2.6, steps r1–r4): revisit each basis
function in turn, draw `candidates` replacements, keep the best of
{current, candidates}.  Requires an existing basis (`init =` or a pipeline).
"""
Base.@kwdef struct Refine <: StochasticMethod
    sweeps::Int = 1
    candidates::Int = 25
    scale::Union{Float64, Symbol} = :auto
    sampler::Any = HaltonSample()
    indep_tol::Float64 = 1.0e-4
end
Refine(sweeps::Int; kw...) = Refine(; sweeps, kw...)

"""
    GVM([basis]; scale = nothing, maxiter = 500, gtol = 1e-6)

Joint LBFGS optimisation of all Gaussian parameters (widths via log-Cholesky
encoding, plus shifts) using ForwardDiff/Hellmann–Feynman gradients.  A cold
start requires `basis`; a warm start infers it from `init` when omitted.
`scale` controls only cold-start sampling and must be omitted for warm starts.
"""
Base.@kwdef struct GVM <: GradientMethod
    basis::Union{Nothing, Int} = nothing
    scale::Union{Nothing, Float64, Symbol} = nothing
    maxiter::Int = 500
    gtol::Float64 = 1.0e-6
end
GVM(basis::Int; kw...) = GVM(; basis, kw...)

"""
    DynamicGVM(basis; candidates = 10, scale = :auto, maxiter_step = 100, gtol = 1e-6)

Per-step selection followed by joint LBFGS of the whole current basis
(SVM-style sequential growth).  `basis` is the final basis size, including any
functions supplied through `init`.
"""
Base.@kwdef struct DynamicGVM <: GradientMethod
    basis::Int = 15
    candidates::Int = 10
    scale::Union{Float64, Symbol} = :auto
    maxiter_step::Int = 100
    gtol::Float64 = 1.0e-6
end
DynamicGVM(basis::Int; kw...) = DynamicGVM(; basis, kw...)

"""
    Pipeline(stages)
    alg₁ → alg₂ → alg₃

Composition of methods run left to right; each stage warm-starts from the
previous stage's result.  Built with the `→` operator (`\\to<tab>`).
"""
struct Pipeline <: SolverMethod
    stages::Tuple{Vararg{SolverMethod}}
end

"""
    alg₁ → alg₂

Compose two solver methods into a left-to-right `Pipeline`.
"""
→(a::SolverMethod, b::SolverMethod) = Pipeline((a, b))
→(p::Pipeline, b::SolverMethod) = Pipeline((p.stages..., b))
→(a::SolverMethod, p::Pipeline) = Pipeline((a, p.stages...))
→(p::Pipeline, q::Pipeline) = Pipeline((p.stages..., q.stages...))

Base.show(io::IO, m::SVM) = print(io, "SVM(", m.basis, ")")
Base.show(io::IO, m::Refine) = print(io, "Refine(", m.sweeps, ")")
Base.show(io::IO, m::GVM) = print(io, "GVM(", something(m.basis, "warm"), ")")
Base.show(io::IO, m::DynamicGVM) = print(io, "DynamicGVM(", m.basis, ")")
Base.show(io::IO, p::Pipeline) = join(io, p.stages, " → ")

# Forward declaration: `solve` methods live in solve.jl (Task 4).  Defining
# the empty generic function here makes the Task-1 export well-defined.
function solve end

# Resolve `scale = :auto` against the system's masses (`nothing` when the
# operators were built without masses — then an explicit scale is required).
_resolve_scale(scale::Real, _) = float(scale)
function _resolve_scale(scale::Symbol, masses)
    scale === :auto || throw(ArgumentError("unknown scale $scale; use :auto or a number"))
    masses === nothing && throw(
        ArgumentError(
            "scale = :auto requires Operators(masses[, charges]); pass an explicit scale"
        )
    )
    return default_scale(collect(Float64, masses))
end
