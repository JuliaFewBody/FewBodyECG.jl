using FewBodyHamiltonians

abstract type GaussianBase end

function _shift_matrix(s::AbstractVector{T}) where {T <: Real}
    return hcat(zeros(T, length(s), 2), s)
end

function _check_gaussian_data(A::AbstractMatrix, s::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(s) == (size(A, 1), 3) ||
        throw(ArgumentError("s must have size ($(size(A, 1)), 3)"))
    return nothing
end

function _gaussian_data(A::AbstractMatrix{<:Real}, s::AbstractMatrix{<:Real})
    _check_gaussian_data(A, s)
    T = promote_type(eltype(A), eltype(s))
    return Matrix{T}(A), Matrix{T}(s)
end

struct Rank0Gaussian{T <: Real, M <: AbstractMatrix{T}, S <: AbstractMatrix{T}} <: GaussianBase
    A::Symmetric{T, M}
    s::S
end

function Rank0Gaussian(A::AbstractMatrix{<:Real}, s::AbstractMatrix{<:Real})
    A_data, s_data = _gaussian_data(A, s)
    T = eltype(A_data)
    return Rank0Gaussian{T, typeof(A_data), typeof(s_data)}(Symmetric(A_data), s_data)
end

Rank0Gaussian(A::AbstractMatrix{<:Real}, s::AbstractVector{<:Real}) =
    Rank0Gaussian(A, _shift_matrix(s))

struct Rank1Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}, S <: AbstractMatrix{T}} <: GaussianBase
    A::Symmetric{T, M}
    a::V
    s::S
end

function Rank1Gaussian(
        A::AbstractMatrix{<:Real},
        a::AbstractVector{<:Real},
        s::AbstractMatrix{<:Real},
    )
    A_data, s_data = _gaussian_data(A, s)
    length(a) == size(A_data, 1) || throw(ArgumentError("length(a) must equal size(A, 1)"))
    T = promote_type(eltype(A_data), eltype(a))
    A_t, a_t, s_t = Matrix{T}(A_data), Vector{T}(a), Matrix{T}(s_data)
    return Rank1Gaussian{T, typeof(A_t), typeof(a_t), typeof(s_t)}(Symmetric(A_t), a_t, s_t)
end

Rank1Gaussian(A::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, s::AbstractVector{<:Real}) =
    Rank1Gaussian(A, a, _shift_matrix(s))

struct Rank2Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}, S <: AbstractMatrix{T}} <: GaussianBase
    A::Symmetric{T, M}
    a::V
    b::V
    s::S
end

function Rank2Gaussian(
        A::AbstractMatrix{<:Real},
        a::AbstractVector{<:Real},
        b::AbstractVector{<:Real},
        s::AbstractMatrix{<:Real},
    )
    A_data, s_data = _gaussian_data(A, s)
    (length(a) == size(A_data, 1) && length(b) == size(A_data, 1)) ||
        throw(ArgumentError("length(a) and length(b) must equal size(A, 1)"))
    T = promote_type(eltype(A_data), eltype(a), eltype(b))
    A_t, a_t, b_t, s_t = Matrix{T}(A_data), Vector{T}(a), Vector{T}(b), Matrix{T}(s_data)
    return Rank2Gaussian{T, typeof(A_t), typeof(a_t), typeof(s_t)}(
        Symmetric(A_t),
        a_t,
        b_t,
        s_t,
    )
end

Rank2Gaussian(
    A::AbstractMatrix{<:Real},
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
) = Rank2Gaussian(A, a, b, _shift_matrix(s))

@enum SpinProjection::Int8 down = -1 up = 1

struct SpinState
    projections::Vector{SpinProjection}
end

SpinState(projections::AbstractVector{SpinProjection}) = SpinState(collect(projections))

struct SpinGaussian{G <: GaussianBase} <: GaussianBase
    orbital::G
    spin::SpinState
end

spin_overlap(bra::SpinState, ket::SpinState) = bra.projections == ket.projections ? 1.0 : 0.0

function spin_element(bra::SpinState, ket::SpinState, site::Integer, component::Symbol)
    length(bra.projections) == length(ket.projections) ||
        throw(ArgumentError("spin states must have the same number of sites"))
    site in eachindex(bra.projections) || throw(ArgumentError("invalid spin site $site"))

    bra_spin, ket_spin = bra.projections[site], ket.projections[site]
    if component === :z
        bra_spin == ket_spin || return 0.0
        return bra_spin === up ? 0.5 : -0.5
    elseif component === :x
        return bra_spin == ket_spin ? 0.0 : 0.5
    elseif component === :y
        bra_spin == ket_spin && return 0.0im
        return bra_spin === up ? -0.5im : 0.5im
    end

    throw(ArgumentError("component must be :x, :y, or :z"))
end

struct BasisSet{G <: GaussianBase}
    functions::Vector{G}
end

struct KineticOperator{T <: Real, M <: AbstractMatrix{T}} <: FewBodyHamiltonians.KineticTerm
    K::M
end

struct CoulombOperator{T <: Real, V <: AbstractVector{T}} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    w::V
end

function _potential_parameters(coefficient::Real, gamma::Real, w::AbstractVector{<:Real})
    gamma > 0 || throw(ArgumentError("gamma must be positive"))
    isempty(w) && throw(ArgumentError("w must not be empty"))
    T = promote_type(typeof(coefficient), typeof(gamma), eltype(w))
    return T(coefficient), T(gamma), Vector{T}(w)
end

struct GaussianPotential{T <: Real, V <: AbstractVector{T}} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    gamma::T
    w::V
    function GaussianPotential{T, V}(coefficient::T, gamma::T, w::V) where {T <: Real, V <: AbstractVector{T}}
        gamma > zero(T) || throw(ArgumentError("gamma must be positive"))
        isempty(w) && throw(ArgumentError("w must not be empty"))
        return new{T, V}(coefficient, gamma, w)
    end
end

function GaussianPotential(coefficient::Real, gamma::Real, w::AbstractVector{<:Real})
    coefficient_t, gamma_t, w_t = _potential_parameters(coefficient, gamma, w)
    T = typeof(coefficient_t)
    return GaussianPotential{T, typeof(w_t)}(coefficient_t, gamma_t, w_t)
end

struct OscillatorPotential{T <: Real, V <: AbstractVector{T}} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    w::V
end

function OscillatorPotential(coefficient::Real, w::AbstractVector{<:Real})
    isempty(w) && throw(ArgumentError("w must not be empty"))
    T = promote_type(typeof(coefficient), eltype(w))
    w_t = Vector{T}(w)
    return OscillatorPotential{T, typeof(w_t)}(T(coefficient), w_t)
end

struct ManyBodyGaussianPotential{T <: Real, M <: AbstractMatrix{T}} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    W::Symmetric{T, M}
end

function ManyBodyGaussianPotential(coefficient::Real, W::AbstractMatrix{<:Real})
    size(W, 1) == size(W, 2) || throw(ArgumentError("W must be square"))
    issymmetric(W) || throw(ArgumentError("W must be symmetric"))
    isposdef(Symmetric(W)) || throw(ArgumentError("W must be positive definite"))
    T = promote_type(typeof(coefficient), eltype(W))
    W_t = Matrix{T}(W)
    return ManyBodyGaussianPotential{T, typeof(W_t)}(T(coefficient), Symmetric(W_t))
end

function _spin_potential_parameters(
        coefficient::Real,
        gamma::Real,
        w::AbstractVector{<:Real},
        i::Integer,
        j::Integer,
    )
    i > 0 && j > 0 && i != j || throw(ArgumentError("spin sites must be distinct positive integers"))
    c, g, w_data = _potential_parameters(coefficient, gamma, w)
    return c, g, w_data, Int(i), Int(j)
end

struct GaussianTensorPotential{T <: Real, V <: AbstractVector{T}} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    gamma::T
    w::V
    i::Int
    j::Int
    traceless::Bool
    function GaussianTensorPotential{T, V}(
            coefficient::T,
            gamma::T,
            w::V,
            i::Int,
            j::Int,
            traceless::Bool,
        ) where {T <: Real, V <: AbstractVector{T}}
        gamma > zero(T) || throw(ArgumentError("gamma must be positive"))
        isempty(w) && throw(ArgumentError("w must not be empty"))
        i > 0 && j > 0 && i != j ||
            throw(ArgumentError("spin sites must be distinct positive integers"))
        return new{T, V}(coefficient, gamma, w, i, j, traceless)
    end
end

function GaussianTensorPotential(
        coefficient::Real,
        gamma::Real,
        w::AbstractVector{<:Real},
        i::Integer,
        j::Integer;
        traceless::Bool = true,
    )
    coefficient_t, gamma_t, w_t, i_t, j_t =
        _spin_potential_parameters(coefficient, gamma, w, i, j)
    T = typeof(coefficient_t)
    return GaussianTensorPotential{T, typeof(w_t)}(
        coefficient_t,
        gamma_t,
        w_t,
        i_t,
        j_t,
        traceless,
    )
end

struct GaussianSpinOrbitPotential{T <: Real, V <: AbstractVector{T}} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    gamma::T
    w::V
    i::Int
    j::Int
    function GaussianSpinOrbitPotential{T, V}(
            coefficient::T,
            gamma::T,
            w::V,
            i::Int,
            j::Int,
        ) where {T <: Real, V <: AbstractVector{T}}
        gamma > zero(T) || throw(ArgumentError("gamma must be positive"))
        isempty(w) && throw(ArgumentError("w must not be empty"))
        i > 0 && j > 0 && i != j ||
            throw(ArgumentError("spin sites must be distinct positive integers"))
        return new{T, V}(coefficient, gamma, w, i, j)
    end
end

function GaussianSpinOrbitPotential(
        coefficient::Real,
        gamma::Real,
        w::AbstractVector{<:Real},
        i::Integer,
        j::Integer,
    )
    coefficient_t, gamma_t, w_t, i_t, j_t =
        _spin_potential_parameters(coefficient, gamma, w, i, j)
    T = typeof(coefficient_t)
    return GaussianSpinOrbitPotential{T, typeof(w_t)}(coefficient_t, gamma_t, w_t, i_t, j_t)
end

struct ECG{G <: GaussianBase, O}
    basis::BasisSet{G}
    operators::Vector{O}
end

validate!(g::Rank0Gaussian) = (cholesky(g.A); g)
validate!(g::Rank1Gaussian) = (cholesky(g.A); g)
validate!(g::Rank2Gaussian) = (cholesky(g.A); g)
