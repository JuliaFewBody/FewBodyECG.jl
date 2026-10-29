# Physical Hamiltonian terms, indexed by particle, and their algebra.  Nothing
# here refers to Jacobi coordinates; `jacobi_frame.jl` transforms a
# `Hamiltonian` to the Jacobi-frame operators the solvers use.

"""
    Kinetic(masses; units = :atomic)

Kinetic energy ``-\\sum_i \\nabla_i^2 / 2m_i`` of all particles, with the
center-of-mass motion removed (``\\hbar = 1``).

`masses` defines the system: its length is the number of particles, and its
order fixes the particle indices used by every other term. `units` is
`:atomic`, `:nuclear`, or a [`UnitSystem`](@ref); it names the units of the
masses and couplings and labels the results, and converts nothing. A
[`Hamiltonian`](@ref) holds exactly one `Kinetic` term.

```julia
H = Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2, -1, -1])   # helium
```
"""
struct Kinetic <: FewBodyHamiltonians.KineticTerm
    masses::Vector{Float64}
    units::UnitSystem
    function Kinetic(masses::AbstractVector; units = :atomic)
        Base.require_one_based_indexing(masses)
        length(masses) ≥ 2 ||
            throw(ArgumentError("Kinetic needs at least two particles, got $(length(masses))"))
        all(m -> isfinite(m) && m > 0, masses) ||
            throw(ArgumentError("masses must be finite and positive, got $masses"))
        return new(Vector{Float64}(masses), _unitsystem(units))
    end
end

# Tags the inner constructors, which take every field positionally.  The public
# constructors take couplings and range parameters only by keyword.
struct _Fields end

function _check_pair(i, j)
    min(i, j) ≥ 1 || throw(ArgumentError("particle indices start at 1, got ($i, $j)"))
    i != j || throw(ArgumentError("particle indices must differ, got i = j = $i"))
    return nothing
end

function _check_exponent(exponent, name)
    exponent === nothing && throw(
        ArgumentError("$name(i, j) needs its range parameter: pass exponent = γ with γ > 0")
    )
    isfinite(exponent) && exponent > 0 ||
        throw(ArgumentError("exponent must be positive and finite, got $exponent"))
    return nothing
end

_check_coefficient(c) = isfinite(c) || throw(ArgumentError("coefficient must be finite, got $c"))

"""
    Coulomb(i, j; coefficient = 1)
    Coulomb(charges)

Coulomb pair potential `coefficient / rᵢⱼ` between particles `i` and `j`.

`Coulomb(charges)` is the sum over all pairs `i < j` of
`Coulomb(i, j; coefficient = charges[i] * charges[j])`, ordered
`(1,2), (1,3), …, (2,3), …`. The coupling `qᵢqⱼ` assumes atomic units
(``e^2/4\\pi\\varepsilon_0 = 1``), so it can only be solved with a `Kinetic`
in atomic units.
"""
struct Coulomb <: FewBodyHamiltonians.PotentialTerm
    i::Int
    j::Int
    coefficient::Float64
    n_charges::Int
    function Coulomb(::_Fields, i, j, coefficient, n_charges)
        _check_pair(i, j)
        _check_coefficient(coefficient)
        return new(i, j, coefficient, n_charges)
    end
end
Coulomb(i::Integer, j::Integer; coefficient::Real = 1) = Coulomb(_Fields(), i, j, coefficient, 0)

function Coulomb(charges::AbstractVector{<:Real})
    Base.require_one_based_indexing(charges)
    n = length(charges)
    n ≥ 2 || throw(ArgumentError("Coulomb(charges) needs at least two charges, got $n"))
    return Hamiltonian(
        FewBodyHamiltonians.Operator[
            Coulomb(_Fields(), i, j, charges[i] * charges[j], n) for i in 1:n for j in (i + 1):n
        ]
    )
end

"""
    Gaussian(i, j; coefficient = 1, exponent)

Gaussian pair potential `coefficient · exp(-exponent · rᵢⱼ²)`. `exponent`
has no default.

The product of Gaussian terms is a Gaussian in several pair distances,
`coefficient · exp(-Σₖ exponentₖ rₖ²)`, which writes many-body forces:

```julia
W₃ * Gaussian(1, 2; exponent = γ) * Gaussian(1, 3; exponent = γ) * Gaussian(2, 3; exponent = γ)
```
"""
struct Gaussian <: FewBodyHamiltonians.PotentialTerm
    coefficient::Float64
    factors::Vector{Tuple{Int, Int, Float64}}
    function Gaussian(::_Fields, coefficient, factors)
        _check_coefficient(coefficient)
        for (i, j, γ) in factors
            _check_pair(i, j)
            _check_exponent(γ, "Gaussian")
        end
        return new(coefficient, factors)
    end
end
function Gaussian(i::Integer, j::Integer; coefficient::Real = 1, exponent = nothing)
    _check_exponent(exponent, "Gaussian")
    return Gaussian(_Fields(), coefficient, [(i, j, exponent)])
end

"""
    ManyBodyGaussian(G; coefficient = 1)

Many-body Gaussian `coefficient · exp(-xᵀ G x)`, where `x` stacks the particle
positions and `G` is a symmetric `N × N` matrix with zero row sums, so the
potential depends only on relative positions. A pair Gaussian has
`G = exponent · eᵢⱼeᵢⱼᵀ` with `eᵢⱼ = eᵢ - eⱼ`; products of Gaussians add
their `G` matrices.
"""
struct ManyBodyGaussian <: FewBodyHamiltonians.PotentialTerm
    coefficient::Float64
    G::Matrix{Float64}
    function ManyBodyGaussian(::_Fields, coefficient, G)
        Base.require_one_based_indexing(G)
        _check_coefficient(coefficient)
        all(isfinite, G) || throw(ArgumentError("G must be finite"))
        # A G computed in floating point, such as J'WJ, is symmetric only to rounding.
        isapprox(G, G'; rtol = 1.0e-12) || throw(ArgumentError("G must be symmetric"))
        G = (G + G') / 2
        maximum(abs, sum(G; dims = 2)) ≤ 1.0e-12 * maximum(abs, G) || throw(
            ArgumentError(
                "G must have zero row sums (the potential must depend only on " *
                    "relative positions), got row sums $(vec(sum(G; dims = 2)))"
            )
        )
        return new(coefficient, G)
    end
end
ManyBodyGaussian(G::AbstractMatrix{<:Real}; coefficient::Real = 1) = ManyBodyGaussian(_Fields(), coefficient, G)

"""
    Oscillator(i, j; coefficient = 1)

Harmonic pair potential `coefficient · rᵢⱼ²`.
"""
struct Oscillator <: FewBodyHamiltonians.PotentialTerm
    i::Int
    j::Int
    coefficient::Float64
    function Oscillator(::_Fields, i, j, coefficient)
        _check_pair(i, j)
        _check_coefficient(coefficient)
        return new(i, j, coefficient)
    end
end
Oscillator(i::Integer, j::Integer; coefficient::Real = 1) = Oscillator(_Fields(), i, j, coefficient)

"""
    Custom(f, i, j; rtol = 1e-8, atol = 0.0, maxevals = 10_000)

Pair potential `f(rᵢⱼ)` given by a callable of the distance. Its matrix
elements are computed by adaptive radial quadrature with the given
tolerances; see [`NumericalPotential`](@ref). A scalar multiplier is kept as
the term's `coefficient`, so `c * Custom(f, i, j)` is the potential `c · f(rᵢⱼ)`.
"""
struct Custom{F} <: FewBodyHamiltonians.PotentialTerm
    f::F
    i::Int
    j::Int
    coefficient::Float64
    rtol::Float64
    atol::Float64
    maxevals::Int
    function Custom(::_Fields, f, i, j, coefficient, rtol, atol, maxevals)
        _check_pair(i, j)
        _check_coefficient(coefficient)
        return new{typeof(f)}(f, i, j, coefficient, rtol, atol, maxevals)
    end
end
Custom(f, i::Integer, j::Integer; rtol::Real = 1.0e-8, atol::Real = 0.0, maxevals::Integer = 10_000) =
    Custom(_Fields(), f, i, j, 1, rtol, atol, maxevals)

"""
    Tensor(i, j; coefficient = 1, exponent, traceless = true)

Gaussian-form tensor interaction between particles `i` and `j`, coupling
their pair coordinate (range `exponent`) to their spins; see
[`GaussianTensorOperator`](@ref). It needs a [`SpinGaussian`](@ref) basis and
is solved through `FewBodyECG.build_hamiltonian_matrix`.
"""
struct Tensor <: FewBodyHamiltonians.PotentialTerm
    i::Int
    j::Int
    coefficient::Float64
    exponent::Float64
    traceless::Bool
    function Tensor(::_Fields, i, j, coefficient, exponent, traceless)
        _check_pair(i, j)
        _check_coefficient(coefficient)
        _check_exponent(exponent, "Tensor")
        return new(i, j, coefficient, exponent, traceless)
    end
end
Tensor(i::Integer, j::Integer; coefficient::Real = 1, exponent = nothing, traceless::Bool = true) =
    Tensor(_Fields(), i, j, coefficient, exponent, traceless)

"""
    SpinOrbit(i, j; coefficient = 1, exponent)

Gaussian-form spin-orbit interaction between particles `i` and `j`, coupling
the orbital motion in their pair coordinate (range `exponent`) to their total
spin; see [`GaussianSpinOrbitOperator`](@ref). It needs a
[`SpinGaussian`](@ref) basis and is solved through
`FewBodyECG.build_hamiltonian_matrix`.
"""
struct SpinOrbit <: FewBodyHamiltonians.PotentialTerm
    i::Int
    j::Int
    coefficient::Float64
    exponent::Float64
    function SpinOrbit(::_Fields, i, j, coefficient, exponent)
        _check_pair(i, j)
        _check_coefficient(coefficient)
        _check_exponent(exponent, "SpinOrbit")
        return new(i, j, coefficient, exponent)
    end
end
SpinOrbit(i::Integer, j::Integer; coefficient::Real = 1, exponent = nothing) =
    SpinOrbit(_Fields(), i, j, coefficient, exponent)

const _PotentialTerm = Union{Coulomb, Gaussian, ManyBodyGaussian, Oscillator, Custom, Tensor, SpinOrbit}
const _PhysicalTerm = Union{Kinetic, _PotentialTerm}

"""
    Hamiltonian

A sum of physical terms ([`Kinetic`](@ref), [`Coulomb`](@ref),
[`Gaussian`](@ref), …), built with `+`, `-`, scalar `*` and `/`, and `sum`:

```julia
q = [+2, -1, -1]
H = Kinetic([1.0e15, 1.0, 1.0]) +
    sum(Coulomb(i, j; coefficient = q[i] * q[j]) for i in 1:3 for j in (i + 1):3)
```

The terms are kept in the order they were written; like terms are not merged.
A Hamiltonian passed to [`solve`](@ref) must contain exactly one `Kinetic`
term; partial sums without one are ordinary values.
"""
struct Hamiltonian
    terms::Vector{FewBodyHamiltonians.Operator}
    function Hamiltonian(terms)
        count(t -> t isa Kinetic, terms) ≤ 1 || throw(
            ArgumentError("a Hamiltonian holds one Kinetic term; got two Kinetic terms")
        )
        return new(terms)
    end
end

Base.length(H::Hamiltonian) = length(H.terms)
Base.iterate(H::Hamiltonian, state...) = iterate(H.terms, state...)
Base.getindex(H::Hamiltonian, i::Int) = H.terms[i]
Base.eltype(::Type{Hamiltonian}) = FewBodyHamiltonians.Operator

const _Sum = Union{_PhysicalTerm, Hamiltonian}

_terms(t::_PhysicalTerm) = FewBodyHamiltonians.Operator[t]
_terms(H::Hamiltonian) = H.terms

# The four signatures together resolve every combination of a term and a
# Hamiltonian without ambiguity against the raw-operator methods below.
for (A, B) in ((_Sum, _Sum), (_Sum, _PhysicalTerm), (_PhysicalTerm, _Sum), (_PhysicalTerm, _PhysicalTerm))
    @eval Base.:+(a::$A, b::$B) = Hamiltonian(vcat(_terms(a), _terms(b)))
    @eval Base.:-(a::$A, b::$B) = a + _scale(b, -1)
end

_mix_error(op) = ArgumentError(
    "physical terms cannot be combined with Jacobi-frame operators such as " *
        "$(nameof(typeof(op))); write the whole Hamiltonian with physical terms, " *
        "or pass a vector of Jacobi-frame operators to solve"
)
Base.:+(::_Sum, op::FewBodyHamiltonians.Operator) = throw(_mix_error(op))
Base.:+(op::FewBodyHamiltonians.Operator, ::_Sum) = throw(_mix_error(op))
Base.:-(::_Sum, op::FewBodyHamiltonians.Operator) = throw(_mix_error(op))
Base.:-(op::FewBodyHamiltonians.Operator, ::_Sum) = throw(_mix_error(op))

Base.:-(x::_Sum) = _scale(x, -1)
Base.:*(c::Real, x::_Sum) = _scale(x, c)
Base.:*(x::_Sum, c::Real) = _scale(x, c)
Base.:/(x::_Sum, c::Real) = _scale(x, inv(c))
Base.:*(c::Number, ::_Sum) = throw(ArgumentError("coefficients must be real, got $c"))
Base.:*(::_Sum, c::Number) = throw(ArgumentError("coefficients must be real, got $c"))

_scale(::Kinetic, c) = throw(
    ArgumentError("Kinetic takes no scalar factor; scale the masses instead: Kinetic(masses ./ $c)")
)
_scale(t::Coulomb, c) = Coulomb(_Fields(), t.i, t.j, c * t.coefficient, t.n_charges)
_scale(t::Gaussian, c) = Gaussian(_Fields(), c * t.coefficient, t.factors)
_scale(t::ManyBodyGaussian, c) = ManyBodyGaussian(_Fields(), c * t.coefficient, t.G)
_scale(t::Oscillator, c) = Oscillator(_Fields(), t.i, t.j, c * t.coefficient)
_scale(t::Tensor, c) = Tensor(_Fields(), t.i, t.j, c * t.coefficient, t.exponent, t.traceless)
_scale(t::SpinOrbit, c) = SpinOrbit(_Fields(), t.i, t.j, c * t.coefficient, t.exponent)
_scale(t::Custom, c) = Custom(_Fields(), t.f, t.i, t.j, c * t.coefficient, t.rtol, t.atol, t.maxevals)
_scale(H::Hamiltonian, c) = Hamiltonian(FewBodyHamiltonians.Operator[_scale(t, c) for t in H.terms])

# Particle-space exponent matrix Σ γ eᵢⱼeᵢⱼᵀ of a list of pair factors.
function _pair_matrix(factors, N)
    G = zeros(N, N)
    for (i, j, γ) in factors
        max(i, j) ≤ N || throw(ArgumentError("particle index $(max(i, j)) out of range 1:$N"))
        G[i, i] += γ
        G[j, j] += γ
        G[i, j] -= γ
        G[j, i] -= γ
    end
    return G
end

Base.:*(a::Gaussian, b::Gaussian) = Gaussian(_Fields(), a.coefficient * b.coefficient, vcat(a.factors, b.factors))
Base.:*(a::Gaussian, b::ManyBodyGaussian) =
    ManyBodyGaussian(_Fields(), a.coefficient * b.coefficient, b.G + _pair_matrix(a.factors, size(b.G, 1)))
Base.:*(a::ManyBodyGaussian, b::Gaussian) = b * a
function Base.:*(a::ManyBodyGaussian, b::ManyBodyGaussian)
    size(a.G) == size(b.G) ||
        throw(ArgumentError("many-body Gaussians act on $(size(a.G, 1)) and $(size(b.G, 1)) particles"))
    return ManyBodyGaussian(_Fields(), a.coefficient * b.coefficient, a.G + b.G)
end
Base.:*(a::_PhysicalTerm, b::_PhysicalTerm) = throw(
    ArgumentError(
        "the product of $(nameof(typeof(a))) and $(nameof(typeof(b))) has no analytic " *
            "matrix element; write the pair potential as Custom(f, i, j)"
    )
)

_term_string(t::Coulomb) = "Coulomb($(t.i), $(t.j))"
_term_string(t::Oscillator) = "Oscillator($(t.i), $(t.j))"
_term_string(t::Custom) = "Custom(f, $(t.i), $(t.j))"
_term_string(t::Gaussian) = join(("Gaussian($i, $j; exponent = $γ)" for (i, j, γ) in t.factors), " * ")
_term_string(t::ManyBodyGaussian) = "ManyBodyGaussian(G = $(t.G))"
_term_string(t::Tensor) =
    "Tensor($(t.i), $(t.j); exponent = $(t.exponent)$(t.traceless ? "" : ", traceless = false"))"
_term_string(t::SpinOrbit) = "SpinOrbit($(t.i), $(t.j); exponent = $(t.exponent))"

Base.show(io::IO, t::Kinetic) = print(io, "Kinetic(masses = ", t.masses, ")")
Base.show(io::IO, t::_PotentialTerm) =
    print(io, t.coefficient, " * ", _term_string(t))
Base.show(io::IO, H::Hamiltonian) = print(io, "Hamiltonian with ", length(H), " terms")

# The `Kinetic` term of `H`, or `nothing` when it has none.
function _kinetic(H::Hamiltonian)
    k = findfirst(t -> t isa Kinetic, H.terms)
    return k === nothing ? nothing : H.terms[k]
end

function _hamiltonian_header(H::Hamiltonian)
    K = _kinetic(H)
    K === nothing && return "Hamiltonian (no kinetic energy)"
    return "Hamiltonian ($(length(K.masses)) particles; $(K.units.energy), $(K.units.length))"
end

function Base.show(io::IO, ::MIME"text/plain", H::Hamiltonian)
    println(io, _hamiltonian_header(H), ":")
    for t in H.terms
        if t isa Kinetic
            println(io, "  ", t)
        else
            println(io, "  ", t.coefficient < 0 ? "- " : "+ ", abs(t.coefficient), " ", _term_string(t))
        end
    end
    return
end
