module FewBodyECG

using LinearAlgebra
using FewBodyHamiltonians

export Λ, _jacobi_transform

export generate_bij, _generate_A_matrix

export GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, SpinProjection, SpinState, SpinGaussian, up, down
export BasisSet, ECG, KineticOperator, CoulombOperator, GaussianPotential, OscillatorPotential
export ManyBodyGaussianPotential, GaussianTensorPotential, GaussianSpinOrbitPotential

const Operator = FewBodyHamiltonians.Operator

export Operator

export build_hamiltonian_matrix, build_overlap_matrix, solve_generalized_eigenproblem, solve_ECG, convergence

export ψ₀, SolverResults, convergence, correlation_function, plot_correlation

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("hamiltonian.jl")
include("sampling.jl")
include("utils/solver_results.jl")
include("utils/observables.jl")
include("utils/plotting.jl")


end
