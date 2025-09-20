using Test
using FewBodyECG

@testset "Sampling Module Tests" begin

    @testset "generate_bij function" begin
        b1 = 1.5
        n_terms = 5
        i = 2

        bij_quasi = generate_bij(:quasirandom, i, n_terms, b1)
        @test length(bij_quasi) == n_terms
        @test all(0 .<= bij_quasi .<= b1)
        using QuasiMonteCarlo
        expected_quasi = QuasiMonteCarlo.sample(i + 1, n_terms, HaltonSample())[:, end] * b1
        @test isapprox(bij_quasi, expected_quasi; atol = 1.0e-12)

        bij_rand = generate_bij(:random, i, n_terms, b1)
        @test length(bij_rand) == n_terms
        @test all(0 .<= bij_rand .<= b1)

        @test_throws ErrorException generate_bij(:foo, i, n_terms, b1)
    end

end
