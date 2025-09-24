using FewBodyHamiltonians

abstract type GaussianBase end

struct Rank0Gaussian <: GaussianBase
    A::Matrix{Float64}
end

struct Rank1Gaussian <: GaussianBase
    A::Matrix{Float64}
    a::Vector{Float64}
end

struct Rank2Gaussian <: GaussianBase
    A::Matrix{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
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
