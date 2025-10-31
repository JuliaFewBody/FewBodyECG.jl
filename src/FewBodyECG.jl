module FewBodyECG

using LinearAlgebra
using FewBodyHamiltonians

export Λ, _jacobi_transform

export generate_bij, _generate_A_matrix

export GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet, ECG, KineticOperator, CoulombOperator

const Operator = FewBodyHamiltonians.Operator

export Operator

export build_hamiltonian_matrix, build_overlap_matrix, solve_generalized_eigenproblem, solve_ECG, convergence

export ψ₀, SolverResults, convergence

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("hamiltonian.jl")
include("sampling.jl")
include("utils.jl")


end
