using Test
using LinearAlgebra
using FewBodyECG
using FewBodyECG: jacobi_transform, Λ, build_hamiltonian_matrix, coulomb_weights

@testset "physical terms" begin
    @testset "constructors store keywords and convert" begin
        k = Kinetic([1, 2, 3])
        @test k.masses == [1.0, 2.0, 3.0] && k.masses isa Vector{Float64}
        @test k.units == UnitSystem("Ha", "bohr")
        @test Kinetic([1.0, 1.0]; units = :nuclear).units == UnitSystem("MeV", "fm")
        c = Coulomb(1, 2)
        @test (c.i, c.j, c.coefficient, c.n_charges) == (1, 2, 1.0, 0)
        @test Coulomb(1, 3; coefficient = -2).coefficient === -2.0
        g = Gaussian(1, 2; exponent = 1 // 2)
        @test g.coefficient === 1.0 && g.factors == [(1, 2, 0.5)]
        @test Oscillator(2, 3; coefficient = 0.25).coefficient === 0.25
        f = r -> -exp(-r^2)
        @test Custom(f, 1, 2).f === f
        @test Tensor(1, 2; exponent = 1.0).traceless
    end

    @testset "argument errors" begin
        @test_throws "exponent" Gaussian(1, 2)
        @test_throws "must differ" Coulomb(1, 1)
        @test_throws "must differ" Gaussian(2, 2; exponent = 1.0)
        @test_throws "positive" Gaussian(1, 2; exponent = -1.0)
        @test_throws "positive" Kinetic([1.0, -1.0])
        @test_throws "at least two" Kinetic([1.0])
        @test_throws "symmetric" ManyBodyGaussian([1.0 2.0; 0.0 1.0])
        @test_throws "row sums" ManyBodyGaussian([1.0 0.0; 0.0 1.0])
        J = [1.0 -1.0 0.0; 0.5 0.5 -1.0] ./ [1.0, 3.0]
        G = J' * [0.6 0.1; 0.1 0.6] * J            # symmetric only to rounding
        mb = ManyBodyGaussian(G)
        @test issymmetric(mb.G) && mb.G ≈ G
    end

    @testset "sums" begin
        H = Kinetic([1.0, 1.0, 1.0]) + Coulomb(1, 2) - Coulomb(2, 3)
        @test H isa Hamiltonian && length(H) == 3
        @test H[3].coefficient == -1.0
        V = sum(Gaussian(i, j; exponent = 1.0) for i in 1:3 for j in (i + 1):3)
        @test V isa Hamiltonian && length(V) == 3
        @test length(Kinetic([1.0, 1.0, 1.0]) + V) == 4
        @test length(V + V) == 6
        @test_throws "two Kinetic" (Kinetic([1.0, 1.0]) + Coulomb(1, 2)) + (Kinetic([1.0, 1.0]) + Coulomb(1, 2))
    end

    @testset "scalars" begin
        @test (-178 * Gaussian(1, 2; exponent = 0.639)).coefficient == -178.0
        @test (Coulomb(1, 2) * 3).coefficient == 3.0
        @test (Oscillator(1, 2) / 4).coefficient == 0.25
        @test (-Coulomb(1, 2)).coefficient == -1.0
        V = 2 * (Coulomb(1, 2) + Gaussian(1, 2; exponent = 1.0))
        @test V[1].coefficient == 2.0 && V[2].coefficient == 2.0
        @test_throws "masses" 2 * Kinetic([1.0, 1.0])
        @test_throws "Kinetic" 2 * (Kinetic([1.0, 1.0]) + Coulomb(1, 2))
        @test_throws "real" (1 + 2im) * Coulomb(1, 2)
        @test_throws "real" Coulomb(1, 2) * 2im
        G = [2.0 -1 -1; -1 2 -1; -1 -1 2]
        @test (2 * ManyBodyGaussian(G)).coefficient == 2.0
        @test (2 * Tensor(1, 2; exponent = 1.0, traceless = false)).coefficient == 2.0
        @test !(2 * Tensor(1, 2; exponent = 1.0, traceless = false)).traceless
        @test (-SpinOrbit(1, 2; exponent = 0.3)).coefficient == -1.0
        q = 2 * Coulomb([2, -1, -1])
        @test all(t.n_charges == 3 for t in q)
    end

    @testset "Coulomb(charges)" begin
        H = Coulomb([2, -1, -1])
        @test [(t.i, t.j, t.coefficient) for t in H] == [(1, 2, -2.0), (1, 3, -2.0), (2, 3, 1.0)]
    end

    @testset "products" begin
        a, b = 1.0, 2.0
        p = Gaussian(1, 2; exponent = a) * (3 * Gaussian(1, 3; exponent = b))
        @test p isa Gaussian && p.coefficient == 3.0 && p.factors == [(1, 2, a), (1, 3, b)]
        G = [2.0 -1 -1; -1 2 -1; -1 -1 2]
        m = Gaussian(1, 2; exponent = a) * ManyBodyGaussian(G)
        @test m isa ManyBodyGaussian && m.G ≈ G + a * [1 -1 0; -1 1 0; 0 0 0]
        @test (ManyBodyGaussian(G) * Gaussian(1, 2; exponent = a)).G == m.G
        mm = (2 * ManyBodyGaussian(G)) * (3 * ManyBodyGaussian(G))
        @test mm isa ManyBodyGaussian && mm.coefficient == 6.0 && mm.G == 2G
        @test_throws "3 and 2 particles" ManyBodyGaussian(G) * ManyBodyGaussian([1.0 -1.0; -1.0 1.0])
        @test_throws "Custom" Coulomb(1, 2) * Gaussian(1, 2; exponent = 1.0)
        @test_throws "Custom" Oscillator(1, 2) * Oscillator(1, 3)
    end

    @testset "raw Jacobi-frame operators do not mix" begin
        @test_throws "Jacobi" Kinetic([1.0, 1.0]) + CoulombOperator(1.0, [1.0])
        @test_throws "Jacobi" CoulombOperator(1.0, [1.0]) + Coulomb(1, 2)
        @test_throws "Jacobi" Coulomb(1, 2) - CoulombOperator(1.0, [1.0])
        @test_throws "Jacobi" CoulombOperator(1.0, [1.0]) - Coulomb(1, 2)
    end

    @testset "show" begin
        s = sprint(show, MIME"text/plain"(), Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([2, -1, -1]))
        @test occursin("Hamiltonian (3 particles; Ha, bohr)", s)
        @test occursin("- 2.0 Coulomb(1, 2)", s) && occursin("+ 1.0 Coulomb(2, 3)", s)
        s = sprint(show, MIME"text/plain"(), Kinetic([1.0, 1.0]) + ManyBodyGaussian([1.0 -1.0; -1.0 1.0]))
        @test occursin("+ 1.0 ManyBodyGaussian(G = [1.0 -1.0; -1.0 1.0])", s)
    end
end

@testset "Jacobi transform" begin
    m = [1.0e15, 1.0, 206.768]
    _, U = jacobi_transform(m)
    e(i, j) = (v = zeros(3); v[i] = 1; v[j] = -1; v)
    w(i, j) = U' * e(i, j)
    to_jacobi = FewBodyECG._to_jacobi

    ops, masses, units = to_jacobi(Kinetic(m; units = :nuclear) + Gaussian(1, 2; coefficient = -3, exponent = 0.5) + Oscillator(2, 3; coefficient = 0.25) + Coulomb(1, 3; coefficient = -1) + Custom(r -> -1 / r, 2, 3))
    @test masses == m && units == UnitSystem("MeV", "fm")
    @test ops[1] isa KineticOperator && ops[1].K ≈ Λ(m)
    @test ops[2] isa GaussianOperator && ops[2].coefficient == -3 && ops[2].γ == 0.5 && ops[2].w ≈ w(1, 2)
    @test ops[3] isa OscillatorOperator && ops[3].w ≈ w(2, 3)
    @test ops[4] isa CoulombOperator && ops[4].w ≈ w(1, 3)
    @test ops[5] isa NumericalPotential && ops[5].w ≈ w(2, 3)

    γ = 0.7
    three = Gaussian(1, 2; exponent = γ) * Gaussian(1, 3; exponent = γ) * Gaussian(2, 3; exponent = γ)
    W = to_jacobi(Kinetic(m) + 5 * three)[1][2]
    @test W isa ManyBodyGaussianOperator && W.coefficient == 5
    @test W.W ≈ γ * (w(1, 2) * w(1, 2)' + w(1, 3) * w(1, 3)' + w(2, 3) * w(2, 3)')
    @test to_jacobi(Kinetic(m) + ManyBodyGaussian(γ * (e(1, 2) * e(1, 2)' + e(1, 3) * e(1, 3)' + e(2, 3) * e(2, 3)')))[1][2].W ≈ W.W

    same = to_jacobi(Kinetic(m) + Gaussian(1, 2; exponent = 1.0) * Gaussian(1, 2; exponent = 2.0))[1][2]
    @test same.W ≈ 3.0 * w(1, 2) * w(1, 2)'
    c6 = to_jacobi(Kinetic(m) + 2 * (3 * Custom(r -> r^2, 1, 2)))[1][2]
    @test c6.f(2.0) ≈ 24.0

    @test to_jacobi(Kinetic(m) + Tensor(1, 2; exponent = 1.0))[1][2] isa GaussianTensorOperator
    @test to_jacobi(Kinetic(m) + SpinOrbit(1, 2; exponent = 1.0))[1][2] isa GaussianSpinOrbitOperator

    @test coulomb_weights(Kinetic(m) + Coulomb([1, 1, -1])) ≈ [w(1, 2), w(1, 3), w(2, 3)]

    @testset "Jacobi transform errors" begin
        @test_throws "Kinetic" to_jacobi(Coulomb(1, 2) + Coulomb(1, 3))
        @test_throws "out of range" to_jacobi(Kinetic([1.0, 1.0]) + Coulomb(1, 3))
        @test_throws "k ≈ 1.44 MeV fm" to_jacobi(Kinetic([1.0, 1.0]; units = :nuclear) + Coulomb([1, -1]))
        @test_throws "charges" to_jacobi(Kinetic([1.0, 1.0, 1.0]) + Coulomb([1, -1]))
    end

    @testset "four-body semidefinite product" begin
        m4 = [1.0, 1.0, 1.0, 1.0]
        H = Kinetic(m4) + Gaussian(1, 2; exponent = 0.5) * Gaussian(3, 4; exponent = 0.5)
        Wop = to_jacobi(H)[1][2]
        @test Wop isa ManyBodyGaussianOperator && rank(Wop.W) == 2
        g = Rank0Gaussian(Matrix(1.0I, 3, 3), zeros(3))
        @test isfinite(build_hamiltonian_matrix(BasisSet([g]), H)[1, 1])
        @test_throws "positive semidefinite" ManyBodyGaussianOperator(1.0, [1.0 0.0; 0.0 -1.0])
    end
end

@testset "Hamiltonian collection interface and display" begin
    @test length(Coulomb([1, -1, 1, -1])) == 6
    H = Kinetic([1.0, 1.0, 1.0]) + Coulomb(1, 2) - 2 * Gaussian(1, 3; exponent = 0.5) +
        0.25 * Oscillator(2, 3) + Custom(r -> r, 1, 2) + Tensor(1, 2; exponent = 1.0, traceless = false) +
        SpinOrbit(2, 3; exponent = 0.3) +
        Gaussian(1, 2; exponent = 1.0) * Gaussian(2, 3; exponent = 2.0)
    @test eltype(Hamiltonian) == FewBodyECG.Operator
    @test [typeof(t) for t in H][1:2] == [Kinetic, Coulomb]
    s = sprint(show, MIME"text/plain"(), H)
    for line in (
            "Kinetic(masses = [1.0, 1.0, 1.0])", "+ 1.0 Coulomb(1, 2)",
            "- 2.0 Gaussian(1, 3; exponent = 0.5)", "+ 0.25 Oscillator(2, 3)", "Custom(f, 1, 2)",
            "+ 1.0 Tensor(1, 2; exponent = 1.0, traceless = false)", "+ 1.0 SpinOrbit(2, 3; exponent = 0.3)",
            "+ 1.0 Gaussian(1, 2; exponent = 1.0) * Gaussian(2, 3; exponent = 2.0)",
        )
        @test occursin(line, s)
    end
    @test startswith(sprint(show, MIME"text/plain"(), Coulomb(1, 2) + Coulomb(1, 3)), "Hamiltonian (no kinetic energy):")
    @test sprint(show, H) == "Hamiltonian with 8 terms"
    @test sprint(show, -3 * Coulomb(1, 2)) == "-3.0 * Coulomb(1, 2)"
end

@testset "final-review regressions" begin
    @testset "couplings are keyword-only" begin
        @test_throws MethodError Oscillator(1, 2, 0.5)
        @test_throws MethodError SpinOrbit(1, 2, 0.5, 1.0)
        @test_throws MethodError Coulomb(1, 2, -1.0)
        @test_throws MethodError Gaussian(1.0, 2)
        @test_throws MethodError ManyBodyGaussian(1.0, [1.0 -1.0; -1.0 1.0])
    end

    @testset "Custom keeps its sign and scale" begin
        s = sprint(show, MIME"text/plain"(), Kinetic([1.0, 1.0]) - Custom(r -> r, 1, 2))
        @test occursin("- 1.0 Custom(f, 1, 2)", s)
        @test (-2 * Custom(r -> r, 1, 2)).coefficient == -2.0
    end

    @testset "non-finite couplings are rejected" begin
        @test_throws "finite" Coulomb(1, 2) / 0
        @test_throws "finite" Coulomb(1, 2; coefficient = NaN)
        @test_throws "finite" Gaussian(1, 2; exponent = Inf)
        @test_throws "finite" Inf * Gaussian(1, 2; exponent = 1.0)
        @test_throws "finite" ManyBodyGaussian([Inf -Inf; -Inf Inf])
    end

    @test_throws "at least two charges" Coulomb([1.0])
    @test_throws "G is 3×3, but Kinetic has 2 masses" FewBodyECG._to_jacobi(
        Kinetic([1.0, 1.0]) + ManyBodyGaussian([2.0 -1 -1; -1 2 -1; -1 -1 2])
    )
end
