using Test
using FewBodyECG
using LinearAlgebra

import FewBodyECG: _compute_matrix_element

@testset "Overlap ⟨g′|g⟩" begin

    @testset "Symmetry ⟨g′|g⟩ = ⟨g|g′⟩" begin
        A1 = [1.0 0.0; 0.0 2.0]
        A2 = [1.5 0.0; 0.0 1.5]
        s1 = [0.1, 0.2]
        s2 = [0.3, 0.4]

        g1 = Rank0Gaussian(A1, s1)
        g2 = Rank0Gaussian(A2, s2)

        overlap_12 = _compute_matrix_element(g1, g2)
        overlap_21 = _compute_matrix_element(g2, g1)

        @test overlap_12 ≈ overlap_21 rtol = 1.0e-10
    end


    @testset "With shift vectors" begin
        A = [1.0 0.0; 0.0 1.0]
        s1 = zeros(2, 3)
        s2 = [0.0 0.2 0.1; 0.5 -0.1 0.3]

        g1 = Rank0Gaussian(A, s1)
        g2 = Rank0Gaussian(A, s2)

        overlap = _compute_matrix_element(g1, g2)
        overlap_noshift = _compute_matrix_element(g1, g1)

        @test overlap > 0
        @test overlap > overlap_noshift
        @test isfinite(overlap)
    end

end

@testset "Rank1/Rank2 matrix elements" begin

    @testset "Rank1 overlap rejects shifts" begin
        A = [1.0 0.2; 0.2 1.5]
        s1 = [0.1, -0.3]
        s2 = [-0.2, 0.4]
        p = [0.6, -0.1]
        q = [-0.3, 0.8]

        g1 = Rank1Gaussian(A, p, s1)
        g2 = Rank1Gaussian(A, q, s2)

        @test_throws ArgumentError _compute_matrix_element(g1, g2)
    end

    @testset "Rank1 kinetic/coulomb (zero shifts)" begin
        A = [1.0 0.1; 0.1 1.2]
        s = [0.0, 0.0]
        p = [0.7, -0.2]
        q = [-0.4, 0.3]

        g1 = Rank1Gaussian(A, p, s)
        g2 = Rank1Gaussian(A, q, s)

        K = KineticOperator([0.5 0.0; 0.0 0.6])
        V = CoulombOperator(-1.0, [1.0, 0.0])

        T12 = _compute_matrix_element(g1, g2, K)
        T21 = _compute_matrix_element(g2, g1, K)
        V12 = _compute_matrix_element(g1, g2, V)
        V21 = _compute_matrix_element(g2, g1, V)

        @test isfinite(T12) && isfinite(V12)
        @test isapprox(T12, T21; rtol = 1.0e-10)
        @test isapprox(V12, V21; rtol = 1.0e-10)
    end

    @testset "Rank2 overlap symmetry" begin
        A = [1.1 0.2; 0.2 1.4]
        s = [0.0, 0.0]
        a = [0.5, -0.4]
        b = [-0.2, 0.7]
        c = [0.3, 0.6]
        d = [-0.1, -0.8]

        g1 = Rank2Gaussian(A, a, b, s)
        g2 = Rank2Gaussian(A, c, d, s)

        S12 = _compute_matrix_element(g1, g2)
        S21 = _compute_matrix_element(g2, g1)

        @test isfinite(S12)
        @test isapprox(S12, S21; rtol = 1.0e-12)
    end

    @testset "Rank2 kinetic/coulomb (zero shifts)" begin
        A = [1.0 0.2; 0.2 1.5]
        B = [0.9 0.1; 0.1 1.2]
        s = [0.0, 0.0]
        a = [0.5, -0.4]
        b = [-0.2, 0.7]
        c = [0.3, 0.6]
        d = [-0.1, -0.8]

        g1 = Rank2Gaussian(A, a, b, s)
        g2 = Rank2Gaussian(B, c, d, s)

        K = KineticOperator([0.5 0.0; 0.0 0.6])
        V = CoulombOperator(1.0, [1.0, 0.0])

        T = _compute_matrix_element(g1, g2, K)
        Vval = _compute_matrix_element(g1, g2, V)

        @test isfinite(T)
        @test isfinite(Vval)
    end
end

@testset "Shifted rank-0 Gaussian potentials" begin
    A = [1.2;;]
    B = [0.8;;]
    sA = reshape([0.2, -0.1, 0.3], 1, 3)
    sB = reshape([-0.1, 0.4, 0.2], 1, 3)
    bra, ket = Rank0Gaussian(B, sB), Rank0Gaussian(A, sA)
    w, gamma = [1.0], 0.7

    R = inv(A + B)
    v = sA + sB
    overlap = exp(tr(v' * R * v) / 4) * (π / det(A + B))^(3 / 2)
    Bprime = A + B + gamma * w * w'
    expected_gaussian = exp(tr(v' * inv(Bprime) * v) / 4) * (π / det(Bprime))^(3 / 2)
    q = vec(w' * R * v / 2)
    expected_oscillator = (3 * (w' * R * w)[1] / 2 + dot(q, q)) * overlap

    @test _compute_matrix_element(bra, ket) ≈ overlap
    @test _compute_matrix_element(bra, ket, GaussianPotential(1.0, gamma, w)) ≈ expected_gaussian
    @test _compute_matrix_element(bra, ket, OscillatorPotential(1.0, w)) ≈ expected_oscillator
    @test _compute_matrix_element(bra, ket, ManyBodyGaussianPotential(1.0, [0.4;;])) ≈
        exp(tr(v' * inv(A + B + [0.4;;]) * v) / 4) * (π / det(A + B + [0.4;;]))^(3 / 2)
    @test_throws ArgumentError GaussianPotential(1.0, -gamma, w)

    up_orbital = SpinGaussian(ket, SpinState([up]))
    down_orbital = SpinGaussian(ket, SpinState([down]))
    gaussian = GaussianPotential(1.0, gamma, w)
    @test _compute_matrix_element(up_orbital, up_orbital, gaussian) ≈
        _compute_matrix_element(ket, ket, gaussian)
    @test _compute_matrix_element(up_orbital, down_orbital, gaussian) == 0
end

@testset "Gaussian tensor and spin-orbit potentials" begin
    w, gamma = [1.0], 0.4
    tensor = GaussianTensorPotential(1.0, gamma, w, 1, 2; traceless = true)
    raw_tensor = GaussianTensorPotential(1.0, gamma, w, 1, 2; traceless = false)
    spin_orbit = GaussianSpinOrbitPotential(0.7, gamma, w, 1, 2)

    tensor_orbital = Rank0Gaussian([1.0;;], reshape([0.2, -0.3, 0.1], 1, 3))
    updown = SpinGaussian(tensor_orbital, SpinState([up, down]))
    downup = SpinGaussian(tensor_orbital, SpinState([down, up]))
    tensor_value = _compute_matrix_element(updown, downup, tensor)

    @test tensor_value isa Complex
    @test tensor_value ≈ conj(_compute_matrix_element(downup, updown, tensor))

    Bprime = 2 .* [1.0;;] + gamma * w * w'
    Rprime = inv(Bprime)
    v = 2 .* tensor_orbital.s
    Mprime = exp(tr(v' * Rprime * v) / 4) * (π / det(Bprime))^(3 / 2)
    q = vec(w' * Rprime * v / 2)
    radial_square = Mprime * (3 * (w' * Rprime * w)[1] / 2 + dot(q, q))
    @test tensor_value ≈ _compute_matrix_element(updown, downup, raw_tensor) - radial_square / 6

    bra_orbital = Rank0Gaussian([1.3;;], reshape([-0.2, 0.4, 0.1], 1, 3))
    ket_orbital = Rank0Gaussian([0.8;;], reshape([0.3, -0.1, 0.2], 1, 3))
    bra = SpinGaussian(bra_orbital, SpinState([up, down]))
    ket = SpinGaussian(ket_orbital, SpinState([down, down]))
    spin_orbit_value = _compute_matrix_element(bra, ket, spin_orbit)

    @test !iszero(spin_orbit_value)
    @test spin_orbit_value ≈ conj(_compute_matrix_element(ket, bra, spin_orbit))
end

@testset "Kinetic Energy ⟨g′|K|g⟩" begin

    @testset "Simple 1D kinetic energy" begin
        A = [1.0;;]
        s = [0.0]
        g = Rank0Gaussian(A, s)

        Λ = [0.5;;]
        K = KineticOperator(Λ)

        T = _compute_matrix_element(g, g, K)

        @test isfinite(T)
        @test !isnan(T)
    end

    @testset "Symmetry ⟨g′|K|g⟩ = ⟨g|K|g′⟩" begin
        A1 = [1.0 0.0; 0.0 1.0]
        A2 = [1.5 0.0; 0.0 1.5]
        s = [0.0, 0.0]

        g1 = Rank0Gaussian(A1, s)
        g2 = Rank0Gaussian(A2, s)

        Λ = [0.5 0.0; 0.0 0.5]
        K = KineticOperator(Λ)

        T12 = _compute_matrix_element(g1, g2, K)
        T21 = _compute_matrix_element(g2, g1, K)

        @test T12 ≈ T21 rtol = 1.0e-10
    end

    @testset "Real-valued result" begin
        A1 = [1.0 0.0; 0.0 2.0]
        A2 = [1.5 0.1; 0.1 1.5]
        s1 = [0.1, 0.2]
        s2 = [0.3, 0.4]

        g1 = Rank0Gaussian(A1, s1)
        g2 = Rank0Gaussian(A2, s2)

        Λ = [0.5 0.0; 0.0 0.5]
        K = KineticOperator(Λ)

        T = _compute_matrix_element(g1, g2, K)

        @test isfinite(T)
        @test !isnan(T)
        @test T isa Real
    end

    @testset "Scaling with Λ" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)

        Λ1 = [1.0 0.0; 0.0 1.0]
        Λ2 = [2.0 0.0; 0.0 2.0]

        K1 = KineticOperator(Λ1)
        K2 = KineticOperator(Λ2)

        T1 = _compute_matrix_element(g, g, K1)
        T2 = _compute_matrix_element(g, g, K2)

        @test T2 ≈ 2 * T1 rtol = 1.0e-10
    end
end


@testset "Coulomb Potential ⟨g′|V|g⟩" begin

    @testset "1D Coulomb attractive" begin
        A = [1.0;;]
        s = [0.0]
        g = Rank0Gaussian(A, s)

        w = [1.0]
        V = CoulombOperator(-1.0, w)

        result = _compute_matrix_element(g, g, V)

        @test isfinite(result)
        @test !isnan(result)
    end

    @testset "1D Coulomb repulsive" begin
        A = [1.0;;]
        s = [0.0]
        g = Rank0Gaussian(A, s)

        w = [1.0]
        V = CoulombOperator(1.0, w)

        result = _compute_matrix_element(g, g, V)

        @test isfinite(result)
    end

    @testset "Symmetry ⟨g′|V|g⟩ = ⟨g|V|g′⟩" begin
        A1 = [1.0 0.0; 0.0 1.0]
        A2 = [1.5 0.0; 0.0 1.5]
        s = [0.0, 0.0]

        g1 = Rank0Gaussian(A1, s)
        g2 = Rank0Gaussian(A2, s)

        w = [1.0, 0.0]
        V = CoulombOperator(-1.0, w)

        V12 = _compute_matrix_element(g1, g2, V)
        V21 = _compute_matrix_element(g2, g1, V)

        @test V12 ≈ V21 rtol = 1.0e-10
    end

    @testset "Small q limit (numerical stability)" begin
        A = [1.0 0.0; 0.0 1.0]
        s1 = [0.0, 0.0]
        s2 = [1.0e-13, 1.0e-13]

        g1 = Rank0Gaussian(A, s1)
        g2 = Rank0Gaussian(A, s2)

        w = [1.0, 0.0]
        V = CoulombOperator(-1.0, w)

        result = _compute_matrix_element(g1, g2, V)

        @test isfinite(result)
        @test !isnan(result)
    end

    @testset "Scaling with coefficient" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)

        w = [1.0, 0.0]
        V1 = CoulombOperator(-1.0, w)
        V2 = CoulombOperator(-2.0, w)

        result1 = _compute_matrix_element(g, g, V1)
        result2 = _compute_matrix_element(g, g, V2)

        @test result2 ≈ 2 * result1 rtol = 1.0e-10
    end

    @testset "Different w vectors" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)

        w1 = [1.0, 0.0]
        w2 = [0.0, 1.0]

        V1 = CoulombOperator(-1.0, w1)
        V2 = CoulombOperator(-1.0, w2)

        result1 = _compute_matrix_element(g, g, V1)
        result2 = _compute_matrix_element(g, g, V2)

        @test result1 ≈ result2 rtol = 1.0e-10
    end
end
