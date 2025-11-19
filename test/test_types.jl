using Test
using LinearAlgebra
using FewBodyECG
import FewBodyECG: validate!
using FewBodyHamiltonians

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

@testset "Rank2Gaussian constructor and validate!" begin
    A = [4.0 0.0; 0.0 5.0]
    a = [0.1, 0.2]
    b = [0.3, 0.4]
    s = [1.0, 2.0]

    g2 = Rank2Gaussian(A, a, b, s)
    @test isa(g2, Rank2Gaussian)
    @test isa(g2.A, Symmetric)
    @test g2.a == a
    @test g2.b == b
    @test g2.s == s

    @test_throws ArgumentError Rank2Gaussian(rand(2, 3), a, b, s)
    @test_throws ArgumentError Rank2Gaussian(A, [1.0], b, s)
    @test_throws ArgumentError Rank2Gaussian(A, a, [1.0], s)
    @test_throws ArgumentError Rank2Gaussian(A, a, b, [1.0])

    # validate! succeeds for positive-definite A
    @test validate!(g2) === g2

    # validate! throws for indefinite A
    A_indef = [0.0 -1.0; -1.0 0.0]
    g2_indef = Rank2Gaussian(A_indef, a, b, s)
    @test_throws LinearAlgebra.PosDefException validate!(g2_indef)
end

@testset "validate! for Rank0 and Rank1 positive/negative-definite" begin
    A_pd = [2.0 0.0; 0.0 2.0]
    s = [0.0, 0.0]
    g0 = Rank0Gaussian(A_pd, s)
    @test validate!(g0) === g0

    A_indef = [0.0 -1.0; -1.0 0.0]
    g0_indef = Rank0Gaussian(A_indef, s)
    @test_throws LinearAlgebra.PosDefException validate!(g0_indef)

    A1 = [1.0 0.0; 0.0 1.0]
    a = [0.0, 0.0]
    g1 = Rank1Gaussian(A1, a, s)
    @test validate!(g1) === g1

    g1_indef = Rank1Gaussian(A_indef, a, s)
    @test_throws LinearAlgebra.PosDefException validate!(g1_indef)
end

@testset "BasisSet, KineticOperator, CoulombOperator, and ECG composition" begin
    A = [2.0 0.0; 0.0 3.0]
    s1 = [1.0, 0.0]
    s2 = [0.0, 1.0]
    g1 = Rank0Gaussian(A, s1)
    g2 = Rank0Gaussian(A, s2)

    bset = BasisSet([g1, g2])
    @test isa(bset, BasisSet)
    @test length(bset.functions) == 2
    @test bset.functions[1] == g1
    @test bset.functions[2] == g2

    K = [1.0 0.0; 0.0 1.0]
    kop = KineticOperator(K)
    @test kop.K == K
    @test kop isa FewBodyHamiltonians.KineticTerm

    coeff = 2.5
    w = [1.0, -1.0]
    cop = CoulombOperator(coeff, w)
    @test cop.coefficient == coeff
    @test cop.w == w
    @test cop isa FewBodyHamiltonians.PotentialTerm

    ecg = ECG(bset, [kop, cop])
    @test ecg.basis === bset
    @test ecg.operators == [kop, cop]
end
