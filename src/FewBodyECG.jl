module FewBodyECG

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("hamiltonian.jl")
include("sampling.jl")
include("optimization.jl")
include("utils.jl")

using .Types
using .Coordinates
using .MatrixElements
using .Hamiltonian
using .Sampling
using .Optimization
using .Utils

export generate_A_matrix, generate_bij, default_b0, jacobi_transform, transform_list, transform_coordinates, inverse_transform_coordinates, ParticleSystem, Particle, GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet, Operator, KineticEnergy, CoulombPotential, FewBodyHamiltonian, MatrixElementResult

export compute_matrix_element, build_overlap_matrix, build_operator_matrix,
    build_hamiltonian_matrix, solve_generalized_eigenproblem,
    generate_basis, compute_ground_state_energy, optimize_ground_state_energy

export ψ₀, plot_wavefunction, plot_density


end
