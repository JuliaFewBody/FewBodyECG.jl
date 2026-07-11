"""
    Operators

Accumulates `KineticOperator` and `CoulombOperator` terms to build a Hamiltonian.
Handles Jacobi coordinate transforms internally so that callers can work
with physical particle indices rather than Jacobi-frame weight vectors.

# Constructors

    Operators()                    # system-unaware; add pre-built operators with `+=`
    Operators(masses)              # system-aware; enables string/index shorthand
    Operators(masses, charges)     # fully automatic; enables `ops += "Coulomb"` shorthand

# System-aware interface

Particle indices follow the original ordering of `masses`.  All Jacobi
transforms are computed internally.

```julia
ops = Operators([m₁, m₂, m₃])
ops += "Kinetic"
ops += ("Coulomb", 1, 2, +1.0)   # pair (1,2) with coupling coefficient +1.0
ops += ("Coulomb", 1, 3, -1.0)
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
end

Operators() = Operators(FewBodyHamiltonians.Operator[], nothing, nothing, nothing)

function Operators(masses::Vector{<:Real})
    _, U = jacobi_transform(Float64.(masses))
    return Operators(FewBodyHamiltonians.Operator[], Float64.(masses), nothing, U)
end

"""
    Operators(masses, charges)

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
function Operators(masses::Vector{<:Real}, charges::Vector{<:Real})
    length(masses) == length(charges) ||
        throw(ArgumentError("masses and charges must have the same length"))
    _, U = jacobi_transform(Float64.(masses))
    return Operators(FewBodyHamiltonians.Operator[], Float64.(masses), Float64.(charges), U)
end

function Base.:+(ops::Operators, op::FewBodyHamiltonians.Operator)
    push!(ops.terms, op)
    return ops
end

function Base.:+(ops::Operators, name::AbstractString)
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

function Base.:+(ops::Operators, term::Tuple{<:AbstractString, <:Integer, <:Integer, <:Real, <:Real})
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

function Base.:+(ops::Operators, term::Tuple{<:AbstractString, <:Integer, <:Integer, <:Real})
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
    println(io, "$header with $n term$(n == 1 ? "" : "s"):")
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
        else
            println(io, "  + $(typeof(op))")
        end
    end
    return
end

"""
    coulomb_weights(ops::Operators) -> Vector{Vector{Float64}}

Return the Jacobi-frame weight vectors for every `CoulombOperator` in `ops`,
in the order they were added. Useful for manual basis construction:

```julia
w_jac = coulomb_weights(ops)
A = _generate_A_matrix(bij, w_jac)
```
"""
coulomb_weights(ops::Operators) = [op.w for op in ops.terms if op isa CoulombOperator]

function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, ops::Operators)
    return build_hamiltonian_matrix(basis, ops.terms)
end
