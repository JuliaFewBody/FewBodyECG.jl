"""
    Method

Abstract supertype of all solver algorithms.  A method is a small struct of
algorithm-level options; problem-level options (`state`, `tol`, `window`,
`init`, `verbose`) live on [`solve`](@ref).  Adding a new method = defining a
new subtype plus `solve`/`step!` methods — pure multiple dispatch.
"""
abstract type Method end

"""
    GradientBackend

How gradients are obtained in the gradient-based methods.  `AutoDiff` (the
default and only v2.0 backend) uses ForwardDiff with Hellmann–Feynman
gradients.  Analytic gradients (Fedorov, Few-Body Syst 58:21, 2017) can be
added later as another subtype without interface changes.
"""
abstract type GradientBackend end

"""
    AutoDiff()

ForwardDiff-based gradient backend (Hellmann–Feynman theorem).
"""
struct AutoDiff <: GradientBackend end

"""
    SVM(basis; candidates = 25, scale = :auto, sampler = HaltonSample(), indep_tol = 1e-4)

Suzuki–Varga stochastic selection (Sect. 4.2.5).  At each of `basis` steps,
`candidates` quasi-random Gaussians are drawn and scored in O(k²) by the
incremental whitened eigensolver; the best admissible one is committed.
`candidates = 1` is the accept-first strategy.  `scale = :auto` resolves via
[`default_scale`](@ref) from the system's masses.
"""
Base.@kwdef struct SVM <: Method
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
Base.@kwdef struct Refine <: Method
    sweeps::Int = 1
    candidates::Int = 25
    scale::Union{Float64, Symbol} = :auto
    sampler::Any = HaltonSample()
    indep_tol::Float64 = 1.0e-4
end
Refine(sweeps::Int; kw...) = Refine(; sweeps, kw...)

"""
    Variational(basis; scale = :auto, maxiter = 500, gtol = 1e-6, gradient = AutoDiff())

Joint LBFGS optimisation of all Gaussian parameters (widths via log-Cholesky
encoding, plus shifts).  Cold-starts from a quasi-random basis unless
`solve(...; init = sol)` provides one.
"""
Base.@kwdef struct Variational <: Method
    basis::Int = 30
    scale::Union{Float64, Symbol} = :auto
    maxiter::Int = 500
    gtol::Float64 = 1.0e-6
    gradient::GradientBackend = AutoDiff()
end
Variational(basis::Int; kw...) = Variational(; basis, kw...)

"""
    GrowVariational(basis; candidates = 10, scale = :auto, maxiter_step = 100, gtol = 1e-6)

Per-step selection followed by joint LBFGS of the whole current basis
(SVM-style sequential growth).
"""
Base.@kwdef struct GrowVariational <: Method
    basis::Int = 15
    candidates::Int = 10
    scale::Union{Float64, Symbol} = :auto
    maxiter_step::Int = 100
    gtol::Float64 = 1.0e-6
end
GrowVariational(basis::Int; kw...) = GrowVariational(; basis, kw...)

"""
    Pipeline(stages)
    alg₁ → alg₂ → alg₃

Composition of methods run left to right; each stage warm-starts from the
previous stage's result.  Built with the `→` operator (`\\to<tab>`).
"""
struct Pipeline <: Method
    stages::Tuple{Vararg{Method}}
end

"""
    alg₁ → alg₂

Compose two solver methods into a left-to-right `Pipeline`.
"""
→(a::Method, b::Method) = Pipeline((a, b))
→(p::Pipeline, b::Method) = Pipeline((p.stages..., b))
→(a::Method, p::Pipeline) = Pipeline((a, p.stages...))
→(p::Pipeline, q::Pipeline) = Pipeline((p.stages..., q.stages...))

Base.show(io::IO, m::SVM) = print(io, "SVM(", m.basis, ")")
Base.show(io::IO, m::Refine) = print(io, "Refine(", m.sweeps, ")")
Base.show(io::IO, m::Variational) = print(io, "Variational(", m.basis, ")")
Base.show(io::IO, m::GrowVariational) = print(io, "GrowVariational(", m.basis, ")")
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
