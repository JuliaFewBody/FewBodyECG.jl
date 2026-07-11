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
    _shift_matrix(s::AbstractVector) -> Matrix

Map a legacy length-`N` shift onto the paper's three-dimensional `N × 3`
supervector by placing it in the `z` Cartesian component (third column).
A matrix argument is returned unchanged.
"""
function _shift_matrix(s::AbstractVector{T}) where {T <: Real}
    result = zeros(T, length(s), 3)
    result[:, 3] .= s
    return result
end
_shift_matrix(s::AbstractMatrix{<:Real}) = s

"""
    _gaussian_data(A, s) -> (A::Matrix, s::Matrix)

Validate and normalise the exponent matrix `A` (square) and the `N × 3` shift
`s`, promoting both to a common element type.
"""
function _gaussian_data(A::AbstractMatrix{<:Real}, s::AbstractMatrix{<:Real})
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(s) == (size(A, 1), 3) ||
        throw(ArgumentError("s must have size (size(A,1), 3)"))
    T = promote_type(eltype(A), eltype(s))
    return Matrix{T}(A), Matrix{T}(s)
end

"""
    Rank0Gaussian(A, s)

Basis function ``g(\\mathbf{r}) = \\exp(-\\mathbf{r}^T A\\,\\mathbf{r} + \\operatorname{tr}(s^T \\mathbf{r}))``.

# Fields
- `A` : symmetric positive-definite ``n_{\\text{dim}} \\times n_{\\text{dim}}`` matrix controlling the Gaussian width and correlations.
- `s` : shift supervector of size ``n_{\\text{dim}} \\times 3``; row `i` is the Cartesian shift of Jacobi coordinate `i`. A length-`N` vector is accepted for compatibility and mapped to the `z` component.
"""
struct Rank0Gaussian{T <: Real, M <: AbstractMatrix{T}, S <: AbstractMatrix{T}} <: GaussianBase
    A::Symmetric{T, M}
    s::S
    function Rank0Gaussian(A::AbstractMatrix{<:Real}, s::AbstractMatrix{<:Real})
        Ad, sd = _gaussian_data(A, s)
        return new{eltype(Ad), typeof(Ad), typeof(sd)}(Symmetric(Ad), sd)
    end
end
Rank0Gaussian(A::AbstractMatrix{<:Real}, s::AbstractVector{<:Real}) =
    Rank0Gaussian(A, _shift_matrix(s))

const Polarization{T} = Union{AbstractVector{T}, AbstractMatrix{T}}

_pol_ncomp(a::AbstractVector) = 1
_pol_ncomp(a::AbstractMatrix) = size(a, 2)
_pol_cols(a::AbstractVector) = reshape(a, :, 1)
_pol_cols(a::AbstractMatrix) = a

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
        size(a, 1) == size(A, 1) ||
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
        size(a, 1) == size(A, 1) ||
            throw(ArgumentError("size(a,1) (or length(a)) must equal size(A,1)"))
        size(b, 1) == size(A, 1) ||
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

"""
    GaussianOperator(coefficient, γ, w)

Two-body Gaussian potential ``V(r_{ij}) = \\text{coefficient} \\cdot e^{-\\gamma r_{ij}^2}``
operator, where ``r_{ij} = |w^T \\mathbf{r}|`` is the inter-particle distance in Jacobi
coordinates selected by the weight vector `w`.

The matrix element reduces to an overlap with a shifted exponent matrix:
``S' = A + B + \\gamma\\, w w^T``, making evaluation exact and free of special functions.

# Fields
- `coefficient` : coupling constant (negative for attractive well).
- `γ`           : inverse-square range parameter (``\\gamma > 0``).
- `w`           : weight vector in Jacobi coordinates selecting the pair.
"""
struct GaussianOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    γ::T
    w::AbstractVector{T}
end

"""
    OscillatorOperator(coefficient, w)

Harmonic (oscillator) two-body potential ``V = \\text{coefficient}\\cdot|w^T\\mathbf{r}|^2``,
where ``w^T\\mathbf{r}`` is the inter-particle coordinate selected by the Jacobi
weight vector `w`.

# Fields
- `coefficient` : coupling constant.
- `w`           : weight vector in Jacobi coordinates selecting the pair.
"""
struct OscillatorOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    w::AbstractVector{T}
    function OscillatorOperator(coefficient::Real, w::AbstractVector{<:Real})
        T = promote_type(typeof(coefficient), eltype(w))
        return new{T}(T(coefficient), Vector{T}(w))
    end
end

"""
    ManyBodyGaussianOperator(coefficient, W)

Many-body Gaussian interaction ``V = \\text{coefficient}\\cdot\\exp(-\\mathbf{r}^T W\\,\\mathbf{r})``
with a symmetric positive-definite exponent matrix `W` acting on all Jacobi
coordinates at once (e.g. a repulsive regulator).

# Fields
- `coefficient` : coupling constant.
- `W`           : symmetric positive-definite exponent matrix.
"""
struct ManyBodyGaussianOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    W::AbstractMatrix{T}
    function ManyBodyGaussianOperator(coefficient::Real, W::AbstractMatrix{<:Real})
        issymmetric(W) || throw(ArgumentError("W must be symmetric"))
        isposdef(W) || throw(ArgumentError("W must be positive definite"))
        T = promote_type(typeof(coefficient), eltype(W))
        return new{T}(T(coefficient), Matrix{T}(W))
    end
end

"""
    SpinProjection

Spin-½ projection eigenstates: [`up`](@ref) (+½) and [`down`](@ref) (−½).
"""
@enum SpinProjection down = -1 up = 1

"""
    up :: SpinProjection

Spin-½ projection eigenstate with eigenvalue +½.
"""
up

"""
    down :: SpinProjection

Spin-½ projection eigenstate with eigenvalue −½.
"""
down

"""
    SpinState(projections)

Direct-product spin-½ state: one [`SpinProjection`](@ref) per particle site.
"""
struct SpinState
    projections::Vector{SpinProjection}
end
SpinState(projections::AbstractVector{SpinProjection}) = SpinState(collect(projections))

"""
    SpinGaussian(orbital, spin)

An explicitly correlated Gaussian with an attached direct-product spin state.
`orbital` is a [`Rank0Gaussian`](@ref); `spin` is a [`SpinState`](@ref).  Only
introduced to support the tensor and spin-orbit interactions; central operators
factor through the spin overlap.
"""
struct SpinGaussian{G <: Rank0Gaussian} <: GaussianBase
    orbital::G
    spin::SpinState
end

"""
    GaussianTensorOperator(coefficient, γ, w, i, j; traceless = true)

Gaussian-form tensor interaction coupling the coordinate `wᵀr` (range `γ`) to
the spins on sites `i` and `j`.  With `traceless = true` the rank-2 spatial
tensor `rₐr_b − ⅓r²δₐ_b` is used.
"""
struct GaussianTensorOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    γ::T
    w::AbstractVector{T}
    i::Int
    j::Int
    traceless::Bool
    function GaussianTensorOperator(
            coefficient::Real, γ::Real, w::AbstractVector{<:Real},
            i::Integer, j::Integer; traceless::Bool = true
        )
        γ > 0 || throw(ArgumentError("γ must be positive, got γ = $γ"))
        i != j || throw(ArgumentError("sites i and j must differ, got i = j = $i"))
        T = promote_type(typeof(coefficient), typeof(γ), eltype(w))
        return new{T}(T(coefficient), T(γ), Vector{T}(w), Int(i), Int(j), traceless)
    end
end

"""
    GaussianSpinOrbitOperator(coefficient, γ, w, i, j)

Gaussian-form spin-orbit interaction coupling the orbital motion in the
coordinate `wᵀr` (range `γ`) to the total spin `Sᵢ + Sⱼ`.  Produces complex
Hermitian matrix elements for shifted Gaussians.
"""
struct GaussianSpinOrbitOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    γ::T
    w::AbstractVector{T}
    i::Int
    j::Int
    function GaussianSpinOrbitOperator(
            coefficient::Real, γ::Real, w::AbstractVector{<:Real},
            i::Integer, j::Integer
        )
        γ > 0 || throw(ArgumentError("γ must be positive, got γ = $γ"))
        i != j || throw(ArgumentError("sites i and j must differ, got i = j = $i"))
        T = promote_type(typeof(coefficient), typeof(γ), eltype(w))
        return new{T}(T(coefficient), T(γ), Vector{T}(w), Int(i), Int(j))
    end
end

struct ECG{G <: GaussianBase, O}
    basis::BasisSet{G}
    operators::Vector{O}
end

validate!(g::Rank0Gaussian) = (cholesky(g.A); g)
validate!(g::Rank1Gaussian) = (cholesky(g.A); g)
validate!(g::Rank2Gaussian) = (cholesky(g.A); g)
