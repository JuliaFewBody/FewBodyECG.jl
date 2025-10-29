using FewBodyHamiltonians

abstract type GaussianBase end

struct Rank0Gaussian <: GaussianBase
    A::Matrix{Number}
    s::Vector{Number}
end

struct Rank1Gaussian <: GaussianBase
    A::Matrix{Number}
    a::Vector{Number}
end

struct Rank2Gaussian <: GaussianBase
    A::Matrix{Number}
    a::Vector{Number}
    b::Vector{Number}
end

struct BasisSet
    functions::Vector{GaussianBase}
end

struct KineticOperator <: FewBodyHamiltonians.KineticTerm
    K::Matrix{Number}
end

struct CoulombOperator <: FewBodyHamiltonians.PotentialTerm
    coefficient::Float64
    w::Vector{Number}
end

struct ECG
    basis::BasisSet
    operators::Vector{Operator}
end
