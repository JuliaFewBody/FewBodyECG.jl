module FewBodyECG


export generate_A_matrix, generate_bij, default_b0, jacobi_transform, transform_coordinates, inverse_transform_coordinates,
    ParticleSystem, shift_vectors,
    GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian,
    BasisSet, Operator, MatrixElementResult, KineticOperator, CoulombOperator,
    compute_matrix_element, build_overlap_matrix, build_operator_matrix,
    build_hamiltonian_matrix, solve_generalized_eigenproblem, compute_ground_state_energy,
    ψ₀, plot_wavefunction, plot_density

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("hamiltonian.jl")
include("sampling.jl")
include("optimization.jl")
include("utils.jl")


end 
