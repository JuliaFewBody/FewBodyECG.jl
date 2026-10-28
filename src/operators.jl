"""
    Operators

Accumulates kinetic and potential terms to build a Hamiltonian, including
`NumericalPotential` terms whose radial quadrature is handled internally.
Handles Jacobi coordinate transforms internally so that callers can work
with physical particle indices rather than Jacobi-frame weight vectors.

# Constructors

    Operators(; units = :atomic)                    # system-unaware; add pre-built operators with `+=`
    Operators(masses; units = :atomic)              # system-aware; enables string/index shorthand
    Operators(masses, charges; units = :atomic)     # fully automatic; enables `ops += "Coulomb"` shorthand

`units` is `:atomic`, `:nuclear`, or a [`UnitSystem`](@ref). It names the units
of the masses and couplings you supply; it labels results and converts nothing.

# Adding terms

`push!(ops, term)` appends `term` to `ops` in place and returns `ops`.
`ops + term` returns a new `Operators` and leaves `ops` unchanged, so
`ops += term` rebinds `ops` to the extended copy. Both accept every term form
shown below.

# System-aware interface

Particle indices follow the original ordering of `masses`.  All Jacobi
transforms are computed internally.

```julia
ops = Operators([m₁, m₂, m₃])
ops += "Kinetic"
ops += ("Coulomb", 1, 2, +1.0)   # pair (1,2) with coupling coefficient +1.0
ops += ("Coulomb", 1, 3, -1.0)
ops += (r -> -exp(-r^2), numerical, 1, 2)
```

When charges are also supplied, the fully-automatic shorthand `ops += "Coulomb"`
adds all ``N(N-1)/2`` pairwise terms with coefficients ``q_i q_j``:

```julia
# Helium atom: nucleus (Z=2), two electrons
ops = Operators([1e15, 1.0, 1.0], [+2, -1, -1])
ops += "Kinetic"
ops += "Coulomb"   # adds (1,2)→-2, (1,3)→-2, (2,3)→+1 automatically
```

# System-unaware interface (pre-built operators)

```julia
ops = Operators()
ops += KineticOperator(Λmat)
ops += CoulombOperator(-1.0, w)
```

Both interfaces can be mixed freely. Pass `ops` directly to [`solve`](@ref) or
`build_hamiltonian_matrix`. Use [`coulomb_weights`](@ref) to retrieve the
Jacobi-frame weight vectors for manual basis construction.
"""
mutable struct Operators
    terms::Vector{FewBodyHamiltonians.Operator}
    masses::Union{Nothing, Vector{Float64}}
    charges::Union{Nothing, Vector{Float64}}
    _U::Union{Nothing, Matrix{Float64}}
    units::UnitSystem
end

Operators(; units = :atomic) =
    Operators(FewBodyHamiltonians.Operator[], nothing, nothing, nothing, _unitsystem(units))

function Operators(masses::Vector{<:Real}; units = :atomic)
    _, U = jacobi_transform(Float64.(masses))
    return Operators(FewBodyHamiltonians.Operator[], Float64.(masses), nothing, U, _unitsystem(units))
end

"""
    Operators(masses, charges; units = :atomic)

Create an `Operators` for a system with given particle `masses` and `charges`.
Enables the fully automatic shorthand `ops += "Coulomb"`, which adds all
``N(N-1)/2`` pairwise Coulomb interactions with coefficients ``q_i q_j``.

```julia
# H⁻: proton (charge +1) + two electrons (charge -1)
ops = Operators([1e15, 1.0, 1.0], [+1, -1, -1])
ops += "Kinetic"
ops += "Coulomb"   # adds (1,2)→-1, (1,3)→-1, (2,3)→+1 automatically
```
"""
function Operators(masses::Vector{<:Real}, charges::Vector{<:Real}; units = :atomic)
    length(masses) == length(charges) ||
        throw(ArgumentError("masses and charges must have the same length"))
    _, U = jacobi_transform(Float64.(masses))
    return Operators(
        FewBodyHamiltonians.Operator[], Float64.(masses), Float64.(charges), U, _unitsystem(units)
    )
end

function Base.push!(ops::Operators, op::FewBodyHamiltonians.Operator)
    push!(ops.terms, op)
    return ops
end

function Base.push!(ops::Operators, name::AbstractString)
    if name == "Kinetic"
        ops.masses !== nothing ||
            throw(ArgumentError("\"Kinetic\" requires Operators(masses)."))
        push!(ops.terms, KineticOperator(ops.masses))
    elseif name == "Coulomb"
        ops.charges !== nothing ||
            throw(
            ArgumentError(
                "\"Coulomb\" without indices requires Operators(masses, charges). " *
                    "Use ops += \"Coulomb\", i, j, coeff for explicit pairs."
            )
        )
        # q_i q_j is the Coulomb coupling only where e²/4πε₀ = 1.
        ops.units == _UNIT_PRESETS.atomic || throw(
            ArgumentError(
                "\"Coulomb\" uses couplings q_i q_j, which assumes atomic units " *
                    "(e²/4πε₀ = 1); this system uses $(ops.units). Add each pair as " *
                    "ops += (\"Coulomb\", i, j, k * q_i * q_j) with k = e²/4πε₀ in your " *
                    "units (k ≈ 1.44 MeV fm for :nuclear)."
            )
        )
        N = length(ops.masses)
        for i in 1:N, j in (i + 1):N
            e_ij = zeros(Float64, N)
            e_ij[i] = 1.0
            e_ij[j] = -1.0
            w = ops._U' * e_ij
            push!(ops.terms, CoulombOperator(ops.charges[i] * ops.charges[j], w))
        end
    else
        throw(
            ArgumentError(
                "Unknown operator \"$name\". Supported: \"Kinetic\", \"Coulomb\"."
            )
        )
    end
    return ops
end

function Base.push!(ops::Operators, term::Tuple{<:AbstractString, <:Integer, <:Integer, <:Real, <:Real})
    name, i, j, coeff, γ = term
    name == "Gaussian" ||
        throw(ArgumentError("Unknown operator \"$name\". Supported: \"Gaussian\"."))
    ops.masses !== nothing ||
        throw(ArgumentError("\"Gaussian\" requires Operators(masses)."))
    i != j || throw(ArgumentError("Particle indices must be distinct, got i = j = $i."))
    N = length(ops.masses)
    1 ≤ i ≤ N || throw(ArgumentError("Particle index i=$i out of range [1, $N]."))
    1 ≤ j ≤ N || throw(ArgumentError("Particle index j=$j out of range [1, $N]."))
    Float64(γ) > 0 || throw(ArgumentError("γ must be positive, got γ = $γ."))
    e_ij = zeros(Float64, N)
    e_ij[i] = 1.0
    e_ij[j] = -1.0
    w = ops._U' * e_ij
    push!(ops.terms, GaussianOperator(Float64(coeff), Float64(γ), w))
    return ops
end

function Base.push!(ops::Operators, term::Tuple{<:AbstractString, <:Integer, <:Integer, <:Real})
    name, i, j, coeff = term
    name in ("Coulomb", "Oscillator") ||
        throw(ArgumentError("Unknown operator \"$name\". Supported: \"Coulomb\", \"Oscillator\"."))
    ops.masses !== nothing ||
        throw(
        ArgumentError(
            "String-based \"$name\" requires Operators(masses)."
        )
    )
    i != j || throw(ArgumentError("Particle indices must be distinct, got i = j = $i."))
    N = length(ops.masses)
    1 ≤ i ≤ N || throw(ArgumentError("Particle index i=$i out of range [1, $N]."))
    1 ≤ j ≤ N || throw(ArgumentError("Particle index j=$j out of range [1, $N]."))
    e_ij = zeros(Float64, N)
    e_ij[i] = 1.0
    e_ij[j] = -1.0
    w = ops._U' * e_ij
    if name == "Coulomb"
        push!(ops.terms, CoulombOperator(Float64(coeff), w))
    else
        push!(ops.terms, OscillatorOperator(Float64(coeff), w))
    end
    return ops
end

function Base.push!(
        ops::Operators,
        term::Tuple{F, NumericalPotentialMarker, I, J},
    ) where {F, I <: Integer, J <: Integer}
    f, _, i, j = term
    ops.masses !== nothing ||
        throw(ArgumentError("NumericalPotential shorthand requires Operators(masses)."))
    i != j || throw(ArgumentError("Particle indices must be distinct, got i = j = $i."))
    N = length(ops.masses)
    1 ≤ i ≤ N || throw(ArgumentError("Particle index i=$i out of range [1, $N]."))
    1 ≤ j ≤ N || throw(ArgumentError("Particle index j=$j out of range [1, $N]."))

    e_ij = zeros(Float64, N)
    e_ij[i] = 1.0
    e_ij[j] = -1.0
    w = ops._U' * e_ij
    push!(ops.terms, NumericalPotential(f, w))
    return ops
end

# `copy` duplicates the term list only; masses, charges, the Jacobi matrix, and
# the units are never mutated after construction, so the copy shares them.
Base.copy(ops::Operators) = Operators(copy(ops.terms), ops.masses, ops.charges, ops._U, ops.units)

Base.:+(ops::Operators, term) = push!(copy(ops), term)

Base.length(ops::Operators) = length(ops.terms)
Base.iterate(ops::Operators) = iterate(ops.terms)
Base.iterate(ops::Operators, state) = iterate(ops.terms, state)
Base.getindex(ops::Operators, i::Int) = ops.terms[i]
Base.eltype(::Type{Operators}) = FewBodyHamiltonians.Operator

function Base.show(io::IO, ops::Operators)
    n = length(ops.terms)
    if ops.masses !== nothing && ops.charges !== nothing
        header = "Operators(masses=$(round.(ops.masses; sigdigits = 3)), charges=$(ops.charges))"
    elseif ops.masses !== nothing
        header = "Operators($(length(ops.masses))-body)"
    else
        header = "Operators"
    end
    println(io, "$header with $n term$(n == 1 ? "" : "s") ($(ops.units.energy), $(ops.units.length)):")
    for op in ops.terms
        if op isa KineticOperator
            println(io, "  + Kinetic")
        elseif op isa CoulombOperator
            println(io, "  + $(op.coefficient) × Coulomb(w = $(round.(op.w; digits = 3)))")
        elseif op isa GaussianOperator
            println(io, "  + $(op.coefficient) × Gaussian(γ = $(round(op.γ; digits = 3)), w = $(round.(op.w; digits = 3)))")
        elseif op isa OscillatorOperator
            println(io, "  + $(op.coefficient) × Oscillator(w = $(round.(op.w; digits = 3)))")
        elseif op isa ManyBodyGaussianOperator
            println(io, "  + $(op.coefficient) × ManyBodyGaussian(W = $(round.(op.W; digits = 3)))")
        elseif op isa NumericalPotential
            println(
                io,
                "  + NumericalPotential(w = $(round.(op.w; digits = 3)), " *
                    "rtol = $(op.rtol), atol = $(op.atol))",
            )
        else
            println(io, "  + $(typeof(op))")
        end
    end
    return
end

"""
    FewBodyECG.coulomb_weights(ops::Operators) -> Vector{Vector{Float64}}

Return the Jacobi-frame weight vectors for every `CoulombOperator` in `ops`,
in the order they were added. Useful for manual basis construction: with pair
widths `b`, the correlation matrix ``A = \\sum_k w_k w_k^T / b_k^2`` gives a
`Rank0Gaussian` whose exponent is ``-\\sum_k r_k^2 / b_k^2`` in the pair
distances ``r_k``:

```julia
using FewBodyECG: coulomb_weights
w = coulomb_weights(ops)
b = [1.0, 2.0, 1.5]                       # one width per pair (three-body system)
A = sum(w[k] * w[k]' / b[k]^2 for k in eachindex(w, b))
g = Rank0Gaussian(A, zeros(length(w[1])))
```
"""
coulomb_weights(ops::Operators) = [op.w for op in ops.terms if op isa CoulombOperator]

function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, ops::Operators)
    return build_hamiltonian_matrix(basis, ops.terms)
end
