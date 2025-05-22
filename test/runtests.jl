using Test
using LinearAlgebra
using FewBodyECG

@testset "FewBodyECG" begin

    include("test_sampling.jl")
    include("test_coordinates.jl")

end

    

