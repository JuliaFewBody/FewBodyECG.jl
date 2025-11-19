using Test
using LinearAlgebra
using FewBodyECG

@testset "Rank0Gaussian constructor and validate!" begin
    A = [2.0 0.0; 0.0 3.0]
    s = [1.0, 2.0]
    g = Rank0Gaussian(A, s)

    @test isa(g, Rank0Gaussian)
    @test isa(g.A, Symmetric)
    @test g.s == s

    A_ns = rand(2, 3)
    @test_throws ArgumentError Rank0Gaussian(A_ns, [1.0, 2.0])

    @test_throws ArgumentError Rank0Gaussian(A, [1.0])

    A_indef = [0.0 -1.0; -1.0 0.0]
    g_indef = Rank0Gaussian(A_indef, s)
end

@testset "Rank1Gaussian constructor and validate!" begin
    A = [3.0 0.0; 0.0 4.0]
    a = [0.1, 0.2]
    s = [1.0, 2.0]
    g1 = Rank1Gaussian(A, a, s)

    @test isa(g1, Rank1Gaussian)
    @test isa(g1.A, Symmetric)
    @test g1.a == a
    @test g1.s == s

    @test_throws ArgumentError Rank1Gaussian(rand(2, 3), a, s)

    @test_throws ArgumentError Rank1Gaussian(A, [1.0], s)
    @test_throws ArgumentError Rank1Gaussian(A, a, [1.0])

end
