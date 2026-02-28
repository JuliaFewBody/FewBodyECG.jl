using FewBodyHamiltonians

"""
    GaussianBase

Abstract supertype for all explicitly correlated Gaussian basis functions.
Concrete subtypes differ by the rank of the polynomial prefactor:
`Rank0Gaussian` (plain Gaussian), `Rank1Gaussian` (linear prefactor),
`Rank2Gaussian` (quadratic prefactor).
"""
abstract type GaussianBase end

"""
    Rank0Gaussian(A, s)

Basis function ``g(\\mathbf{r}) = \\exp(-\\mathbf{r}^T A\\,\\mathbf{r} + \\mathbf{s}^T\\mathbf{r})``.

# Fields
- `A` : symmetric positive-definite ``n_{\\text{dim}} \\times n_{\\text{dim}}`` matrix controlling the Gaussian width and inter-particle correlations.
- `s` : shift vector ``\\mathbf{s} \\in \\mathbb{R}^{n_{\\text{dim}}}``; controls the location of the Gaussian maximum.
"""
struct Rank0Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}} <: GaussianBase
    A::Symmetric{T, M}
    s::V
    function Rank0Gaussian(A::AbstractMatrix{T}, s::AbstractVector{T}) where {T <: Real}
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        length(s) == size(A, 1) || throw(ArgumentError("length(s) != size(A,1)"))
        return new{T, typeof(A), typeof(s)}(Symmetric(A), s)
    end
end

struct Rank1Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}} <: GaussianBase
    A::Symmetric{T, M}
    a::V
    s::V
    function Rank1Gaussian(A::AbstractMatrix{T}, a::AbstractVector{T}, s::AbstractVector{T}) where {T <: Real}
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        (length(a) == size(A, 1) && length(s) == size(A, 1)) ||
            throw(ArgumentError("length(a) and length(s) must equal size(A,1)"))
        return new{T, typeof(A), typeof(a)}(Symmetric(A), a, s)
    end
end

struct Rank2Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}} <: GaussianBase
    A::Symmetric{T, M}
    a::V
    b::V
    s::V
    function Rank2Gaussian(A::AbstractMatrix{T}, a::AbstractVector{T}, b::AbstractVector{T}, s::AbstractVector{T}) where {T <: Real}
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        (length(a) == size(A, 1) && length(b) == size(A, 1) && length(s) == size(A, 1)) ||
            throw(ArgumentError("length(a), length(b), length(s) must equal size(A,1)"))
        return new{T, typeof(A), typeof(a)}(Symmetric(A), a, b, s)
    end
end

"""
    BasisSet(functions)

A collection of `GaussianBase` functions that form the variational basis.

# Fields
- `functions` : `Vector{G}` of basis functions, all of the same concrete `GaussianBase` subtype `G`.
"""
struct BasisSet{G <: GaussianBase}
    functions::Vector{G}
end

"""
    KineticOperator(K)
    KineticOperator(masses)

Kinetic-energy operator in Jacobi coordinates.

When constructed from a mass vector the Jacobi-transformed kinetic-energy
matrix ``\\Lambda = J M^{-1} J^T / 2`` is computed automatically via [`Λ`](@ref).

# Fields
- `K` : symmetric ``n_{\\text{dim}} \\times n_{\\text{dim}}`` kinetic-energy matrix (``\\Lambda``).
"""
struct KineticOperator{T <: Real} <: FewBodyHamiltonians.KineticTerm
    K::AbstractMatrix{T}
end

"""
    CoulombOperator(coefficient, w)

Two-body Coulomb (``1/r_{ij}``) interaction operator.

The inter-particle distance is ``|w^T \\mathbf{r}|`` where `w` is a weight
vector in Jacobi coordinates selecting the pair ``(i,j)``.  Construct `w`
by transforming the charge-difference vector with the inverse Jacobi matrix:
`w = U' * charge_vector`.

# Fields
- `coefficient` : coupling constant (e.g. ``q_i q_j``; negative for attraction).
- `w`           : weight vector in Jacobi coordinates.
"""
struct CoulombOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    w::AbstractVector{T}
end

struct ECG{G <: GaussianBase, O}
    basis::BasisSet{G}
    operators::Vector{O}
end

validate!(g::Rank0Gaussian) = (cholesky(g.A); g)
validate!(g::Rank1Gaussian) = (cholesky(g.A); g)
validate!(g::Rank2Gaussian) = (cholesky(g.A); g)
