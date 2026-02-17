using Test
using LinearAlgebra
using FewBodyECG

@testset "FewBodyECG" begin

    include("Aqua.jl")
    include("test_sampling.jl")
    include("test_coordinates.jl")
    include("test_matrix_elements.jl")
    include("test_hamiltonian.jl")
<<<<<<< Updated upstream
    include("test_hydrogen.jl")

=======
    include("test_types.jl")
    include("test_utils.jl")
    
>>>>>>> Stashed changes
end
