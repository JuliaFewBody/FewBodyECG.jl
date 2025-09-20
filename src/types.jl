module Types

using FewBodyHamiltonians

export GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet, Operator, KineticOperator, CoulombOperator, ECG, MatrixElementResult

abstract type GaussianBase end

struct Rank0Gaussian <: GaussianBase
    A::Matrix{Float64}
end

struct Rank1Gaussian <: GaussianBase
    A::Matrix{Float64}
    a::Vector{Vector{Float64}}
end

struct Rank2Gaussian <: GaussianBase
    A::Matrix{Float64}
    a::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
end

struct BasisSet
    functions::Vector{GaussianBase}
end


struct KineticOperator <: FewBodyHamiltonians.KineticTerm
    K::Matrix{Float64}
end

struct CoulombOperator <: FewBodyHamiltonians.PotentialTerm
    coefficient::Float64
    w::Vector{Float64}
end

struct ECG
    basis::BasisSet
    operators::Vector{Operator}
end

struct MatrixElementResult
    bra::GaussianBase
    ket::GaussianBase
    operator::Operator
    value::Float64
end

end
