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
- `A` : symmetric positive-definite ``n_{\\text{dim}} \\times n_{\\text{dim}}`` matrix controlling the Gaussian width and correlations.
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

const Polarization{T} = Union{AbstractVector{T}, AbstractMatrix{T}}

_pol_nrows(a::AbstractVector) = length(a)
_pol_nrows(a::AbstractMatrix) = size(a, 1)
_pol_ncomp(a::AbstractVector) = 1
_pol_ncomp(a::AbstractMatrix) = size(a, 2)
_pol_cols(a::AbstractVector) = reshape(a, :, 1)
_pol_cols(a::AbstractMatrix) = a

_polarization_components(a::Union{AbstractVector, AbstractMatrix}) = _pol_ncomp(a)
_polarization_matrix(a::Union{AbstractVector, AbstractMatrix}) = _pol_cols(a)

function _check_polarization_compat(
        a::Union{AbstractVector, AbstractMatrix},
        b::Union{AbstractVector, AbstractMatrix}
    )
    _pol_ncomp(a) == _pol_ncomp(b) ||
        throw(DimensionMismatch("polarizations must have the same number of components"))
    return nothing
end

function _polar_contract(
        a::Union{AbstractVector, AbstractMatrix},
        M::AbstractMatrix,
        b::Union{AbstractVector, AbstractMatrix}
    )
    _check_polarization_compat(a, b)
    A = _pol_cols(a)
    B = _pol_cols(b)
    return tr(transpose(A) * M * B)
end

function _polar_projection(
        a::Union{AbstractVector, AbstractMatrix},
        x::AbstractVector
    )
    A = _pol_cols(a)
    return transpose(A) * x
end

function _polar_project_dot(
        a::Union{AbstractVector, AbstractMatrix},
        x::AbstractVector,
        b::Union{AbstractVector, AbstractMatrix},
        y::AbstractVector
    )
    _check_polarization_compat(a, b)
    return dot(_polar_projection(a, x), _polar_projection(b, y))
end

"""
    Rank1Gaussian(A, a, s)

Rank-1 (p-wave-like) ECG basis function with linear prefactor.

`a` can be either:
- a vector of length `size(A,1)` (single polarization component), or
- a matrix of size `size(A,1) × ncomp` (multi-component polarization).
"""
struct Rank1Gaussian{
    T <: Real,
    M <: AbstractMatrix{T},
    P <: Polarization{T},
    V <: AbstractVector{T},
} <: GaussianBase
    A::Symmetric{T, M}
    a::P
    s::V
    function Rank1Gaussian(A::AbstractMatrix{T}, a::Polarization{T}, s::AbstractVector{T}) where {T <: Real}
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        _pol_nrows(a) == size(A, 1) ||
            throw(ArgumentError("size(a,1) (or length(a)) must equal size(A,1)"))
        length(s) == size(A, 1) ||
            throw(ArgumentError("length(s) must equal size(A,1)"))
        return new{T, typeof(A), typeof(a), typeof(s)}(Symmetric(A), a, s)
    end
end

"""
    Rank2Gaussian(A, a, b, s)

Rank-2 (d-wave-like) ECG basis function with quadratic prefactor.

`a` and `b` can each be either vectors or matrices. Their first dimension must
match `size(A,1)`. For matrix polarizations, `a` and `b` must have the same
number of columns (`ncomp`), enabling multi-component pure d-wave channels.
"""
struct Rank2Gaussian{
    T <: Real,
    M <: AbstractMatrix{T},
    P <: Polarization{T},
    Q <: Polarization{T},
    V <: AbstractVector{T},
} <: GaussianBase
    A::Symmetric{T, M}
    a::P
    b::Q
    s::V
    function Rank2Gaussian(A::AbstractMatrix{T}, a::Polarization{T}, b::Polarization{T}, s::AbstractVector{T}) where {T <: Real}
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        _pol_nrows(a) == size(A, 1) ||
            throw(ArgumentError("size(a,1) (or length(a)) must equal size(A,1)"))
        _pol_nrows(b) == size(A, 1) ||
            throw(ArgumentError("size(b,1) (or length(b)) must equal size(A,1)"))
        _pol_ncomp(a) == _pol_ncomp(b) ||
            throw(ArgumentError("a and b must have the same number of polarization components"))
        length(s) == size(A, 1) ||
            throw(ArgumentError("length(s) must equal size(A,1)"))
        return new{T, typeof(A), typeof(a), typeof(b), typeof(s)}(Symmetric(A), a, b, s)
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
