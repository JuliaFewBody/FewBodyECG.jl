using Test
using FewBodyECG
using LinearAlgebra

@testset "Sampling Module Tests" begin

    @testset "generate bij" begin
        b = generate_bij(:quasirandom, 1, 5, 2.0; bmin = 0.5, bmax = 4.0)
        @test length(b) == 5
        @test all(x -> x >= 0.5 && x <= 4.0, b)
    end

    @testset "Similar" begin

        bq1 = generate_bij(:quasirandom, 2, 4, 1.0)
        bq2 = generate_bij(:quasirandom, 2, 4, 1.0)
        @test bq1 ≈ bq2
    end

    @testset "Errors" begin
        @test_throws ErrorException generate_bij(:unsupported, 1, 3, 1.0)
    end

    @testset "Full" begin

        s = generate_shift(:quasirandom, 1, 3, 2.0)
        @test length(s) == 3
        @test all(abs.(s) .<= 2.0 .+ eps())

        s_q1 = generate_shift(:quasirandom, 5, 3, 1.5)
        s_q2 = generate_shift(:quasirandom, 5, 3, 1.5)
        @test s_q1 ≈ s_q2

        @test_throws ErrorException generate_shift(:nope, 1, 2, 1.0)

        bij = [1.0, 2.0]
        w1 = [1.0, 0.0, 0.0]
        w2 = [0.0, 1.0, 0.0]
        A = _generate_A_matrix(bij, [w1, w2])
        expected = Matrix(Diagonal([1.0, 1.0 / 4.0, 0.0]))
        @test A ≈ expected

        @test_throws ArgumentError _generate_A_matrix([1.0, 2.0], [w1])                # length mismatch
        @test_throws ArgumentError _generate_A_matrix([1.0, 2.0], [w1, [1.0, 0.0]])   # differing dimensions

        svec = [0.1, 0.2, 0.3]
        rg = build_rank0([1.0, 2.0], [w1, w2], svec)
        @test isa(rg, FewBodyECG.Rank0Gaussian) || isa(rg, Rank0Gaussian)
    end
end
