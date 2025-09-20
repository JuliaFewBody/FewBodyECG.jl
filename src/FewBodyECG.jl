module FewBodyECG

using LinearAlgebra
using FewBodyHamiltonians

export ParticleSystem, default_b0

export generate_bij

export GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet, ECG, KineticOperator, CoulombOperator

const Operator = FewBodyHamiltonians.Operator

export Operator

export build_hamiltonian_matrix, build_overlap_matrix, solve_generalized_eigenproblem

export ψ₀

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("hamiltonian.jl")
include("sampling.jl")
include("utils.jl")


end
