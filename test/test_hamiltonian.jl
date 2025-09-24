using Test
using LinearAlgebra

using FewBodyECG
import FewBodyECG: _compute_overlap_element, _build_operator_matrix, _compute_matrix_element

@testset "Hamiltonian / overlap helpers" begin

    @testset "compute_overlap_element for Rank0Gaussian" begin
        A = [1.0 0.2; 0.2 1.5]
        B = [0.9 0.1; 0.1 1.2]

        bra = Rank0Gaussian(A)
        ket = Rank0Gaussian(B)

        val = _compute_overlap_element(bra, ket)

        R = inv(A + B)
        n = length(R)
        expected = (π^n / det(A + B))^(3 / 2)

        @test isapprox(val, expected; atol = 1.0e-10)
    end


end

spd(M) = 0.5 * (M + M') + (size(M, 1) == 1 ? 1.0e-12 : 0.0)I


@testset "_compute_overlap_element basic properties" begin
    A = [1.0;;]
    B = [1.0;;]
    bra = Rank0Gaussian(A)
    ket = Rank0Gaussian(B)
    val = _compute_overlap_element(bra, ket)
    @test isapprox(val, (π / 2)^(3 / 2); atol = 1.0e-12)

    @test isapprox(
        _compute_overlap_element(Rank0Gaussian(A), Rank0Gaussian(B)),
        _compute_overlap_element(Rank0Gaussian(B), Rank0Gaussian(A));
        atol = 1.0e-12
    )

    A2 = [1.0 0.2; 0.2 1.5]
    B2 = [0.9 0.1; 0.1 1.2]
    val2 = _compute_overlap_element(Rank0Gaussian(A2), Rank0Gaussian(B2))
    @test val2 > 0
end

@testset "build_overlap_matrix structure & values" begin
    A = spd([1.0 0.2; 0.2 1.5])
    B = spd([0.9 0.1; 0.1 1.2])
    C = spd([1.3 0.0; 0.0 0.8])
    g1 = Rank0Gaussian(A)
    g2 = Rank0Gaussian(B)
    g3 = Rank0Gaussian(C)
    basis = BasisSet([g1, g2, g3])

    S = build_overlap_matrix(basis)

    @test size(S) == (3, 3)
    # Symmetry
    @test S ≈ S'
    @test isapprox(S[1, 1], _compute_overlap_element(g1, g1); atol = 1.0e-12)
    @test isapprox(S[2, 2], _compute_overlap_element(g2, g2); atol = 1.0e-12)
    @test isapprox(S[3, 3], _compute_overlap_element(g3, g3); atol = 1.0e-12)
    @test isapprox(S[1, 2], _compute_overlap_element(g1, g2); atol = 1.0e-12)
    @test isapprox(S[2, 3], _compute_overlap_element(g2, g3); atol = 1.0e-12)
    @test isapprox(S[1, 3], _compute_overlap_element(g1, g3); atol = 1.0e-12)
end

@testset "_build_operator_matrix matches pairwise _compute_matrix_element" begin
    A = spd([1.0 0.2; 0.2 1.5])
    B = spd([0.9 0.1; 0.1 1.2])
    C = spd([1.3 0.0; 0.0 0.8])
    g = [Rank0Gaussian(A), Rank0Gaussian(B), Rank0Gaussian(C)]
    basis = BasisSet(g)

    K = [2.0 0.1; 0.1 2.0]
    kop = KineticOperator(K)
    Hk = _build_operator_matrix(basis, kop)

    @test size(Hk) == (3, 3)
    @test Hk ≈ Hk'
    for i in 1:3, j in 1:3
        @test isapprox(Hk[i, j], _compute_matrix_element(g[i], g[j], kop); atol = 1.0e-10)
    end

    w = [1.0, -1.0]
    cop = CoulombOperator(1.5, w)
    Hc = _build_operator_matrix(basis, cop)

    @test size(Hc) == (3, 3)
    @test Hc ≈ Hc'
    for i in 1:3, j in 1:3
        @test isapprox(Hc[i, j], _compute_matrix_element(g[i], g[j], cop); atol = 1.0e-10)
    end
end

@testset "build_hamiltonian_matrix sums operator matrices" begin
    A = spd([1.0 0.2; 0.2 1.5])
    B = spd([0.9 0.1; 0.1 1.2])
    C = spd([1.3 0.0; 0.0 0.8])
    g = [Rank0Gaussian(A), Rank0Gaussian(B), Rank0Gaussian(C)]
    basis = BasisSet(g)

    kop = KineticOperator([2.0 0.0; 0.0 2.0])
    cop = CoulombOperator(0.75, [1.0, -1.0])

    Hk = _build_operator_matrix(basis, kop)
    Hc = _build_operator_matrix(basis, cop)

    H = build_hamiltonian_matrix(basis, [kop, cop])

    @test size(H) == (3, 3)
    @test H ≈ H'
    @test H ≈ Hk .+ Hc atol = 1.0e-10

    H_only = build_hamiltonian_matrix(basis, [kop])
    @test H_only ≈ Hk atol = 1.0e-12
end


@testset "solve_generalized_eigenproblem" begin
    S = [2.0 0.5; 0.5 1.5]
    H = [1.0 0.2; 0.2 0.8]

    vals, vecs = FewBodyECG.solve_generalized_eigenproblem(H, S)
    @test length(vals) == 2
    @test size(vecs) == (2, 2)
    @test all(isreal, vals)
    @test all(isreal, vecs)

    S_bad = [0.0 0.0; 0.0 0.0]
    H_bad = [1.0 0.0; 0.0 1.0]
    vals_bad, vecs_bad = FewBodyECG.solve_generalized_eigenproblem(H_bad, S_bad)
    @test length(vals_bad) == 2
    @test size(vecs_bad) == (2, 2)
    @test all(isreal, vals_bad)
    @test all(isreal, vecs_bad)
end
