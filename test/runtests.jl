using Test
using LinearAlgebra
using FewBodyECG

@testset "FewBodyECG" begin

    include("Aqua.jl")
    include("test_sampling.jl")
    include("test_methods.jl")
    include("test_solution.jl")
    include("test_state.jl")
    include("test_solve.jl")
    include("test_refine.jl")
    include("test_gradient.jl")
    include("test_pipeline.jl")
    include("test_observables.jl")
    include("test_coordinates.jl")
    include("test_matrix_elements.jl")
    include("test_hamiltonian.jl")
    include("test_svm_eigen.jl")
    include("test_hydrogen.jl")
    include("test_harmonic_oscillator.jl")
    include("test_harmonium.jl")
    include("test_spin_orbit.jl")
    include("test_nuclear.jl")
    include("test_muonic.jl")
    include("test_utils.jl")
    include("test_types.jl")
    include("test_variational.jl")
    include("test_operators.jl")

end
