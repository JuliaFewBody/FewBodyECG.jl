module Types

using FewBodyHamiltonians

export Operator, Hamiltonian, Kinetic, Coulomb, Particle, System, GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, ECGBasis, MatrixElementResult

abstract type GaussianBase end

struct Particle
    mass::Real
    charge::Real
    spin::Union{Nothing, Real}
end

struct System
    particles::Vector{Particle}
    remove_com::Bool
end

struct Rank0Gaussian{T <: Real, M <: AbstractMatrix{T}} <: GaussianBase
    A::M
end

struct Rank1Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{<:AbstractVector{T}}} <: GaussianBase
    A::M
    a::V
end

struct Rank2Gaussian{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{<:AbstractVector{T}}} <: GaussianBase
    A::M
    a::V
    b::V
end

struct ECGBasis{F <: GaussianBase}
    functions::Vector{F}
end

struct MatrixElementResult{B <: GaussianBase, O <: Operator, T <: Real}
    bra::B
    ket::B
    operator::O
    value::T
end

end
