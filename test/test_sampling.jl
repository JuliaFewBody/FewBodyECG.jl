using Test
using FewBodyECG.Sampling

@testset "Sampling Module Tests" begin
    
    @testset "Bij Generation" begin
        # Test quasirandom generation
        b1 = 2.0
        bij_quasi = generate_bij(:quasirandom, 3, 4, b1)
        @test length(bij_quasi) == 4
        @test all(0 .<= bij_quasi .<= b1)
        @test bij_quasi == halton(3, 4) * b1
        
        # Test invalid method
        @test_throws ErrorException generate_bij(:invalid, 3, 4, b1)
    end
end