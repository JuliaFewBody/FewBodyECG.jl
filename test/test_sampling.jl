using Test
using FewBodyECG
using LinearAlgebra

import FewBodyECG: _generate_A_matrix, build_rank0, generate_bij, generate_shift

@testset "Sampling Module Tests" begin

    @testset "generate bij" begin
        b = generate_bij(:quasirandom, 1, 5, 2.0; bmin = 0.5, bmax = 4.0)
        @test length(b) == 5
        @test all(x -> x >= 0.5 && x <= 4.0, b)
    end

    @testset "widths = :wide is log-uniform over [0.01, 100]·b1" begin
        b = reduce(vcat, (generate_bij(:quasirandom, i, 3, 1.0; widths = :wide) for i in 1:2000))
        @test all(x -> 0.01 <= x <= 100.0, b)
        share(lo, hi) = count(x -> lo <= x < hi, b) / length(b)
        for lo in (0.01, 0.1, 1.0, 10.0)                # each of the 4 decades gets 1/4
            @test share(lo, 10lo) ≈ 0.25 atol = 0.01
        end
        @test generate_bij(:quasirandom, 7, 3, 1.0) == generate_bij(:quasirandom, 7, 3, 1.0; widths = :linear)
        @test_throws "unknown widths = :log; use :linear or :wide" generate_bij(:quasirandom, 1, 3, 1.0; widths = :log)
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
        @test size(s) == (3, 3)   # N×3 supervector shift
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
