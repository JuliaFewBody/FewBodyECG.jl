module FewBodyECG

using LinearAlgebra
using FewBodyHamiltonians

export generate_A_matrix, default_b0, jacobi_transform, transform_coordinates, inverse_transform_coordinates, ParticleSystem, shift_vectors

export generate_bij, compute_ground_state_energy

export GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet, ECG, MatrixElementResult, KineticOperator, CoulombOperator

const Operator = FewBodyHamiltonians.Operator
export Operator

export build_hamiltonian_matrix, solve_generalized_eigenproblem

export ψ₀

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("hamiltonian.jl")
include("sampling.jl")
include("utils.jl")


end
