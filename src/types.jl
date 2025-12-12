using FewBodyHamiltonians

abstract type GaussianBase end

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

struct BasisSet{G <: GaussianBase}
    functions::Vector{G}
end

struct KineticOperator{T <: Real} <: FewBodyHamiltonians.KineticTerm
    K::AbstractMatrix{T}
end

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
