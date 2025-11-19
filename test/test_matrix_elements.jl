using Test
using FewBodyECG
using LinearAlgebra

import FewBodyECG: _compute_matrix_element

@testset "Overlap ⟨g′|g⟩" begin

    @testset "Identical Gaussians" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)

        overlap = _compute_matrix_element(g, g)

        expected = (π^2 / det(2 * A))^(3 / 2)
        @test overlap ≈ expected rtol = 1.0e-10
        @test overlap > 0
    end

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

    @testset "No shift vectors" begin
        A1 = [1.0 0.0; 0.0 1.0]
        A2 = [2.0 0.0; 0.0 2.0]
        s = [0.0, 0.0]

        g1 = Rank0Gaussian(A1, s)
        g2 = Rank0Gaussian(A2, s)

        overlap = _compute_matrix_element(g1, g2)

        @test overlap > 0
        @test isfinite(overlap)

        expected = (π^2 / det(A1 + A2))^(3 / 2)
        @test overlap ≈ expected rtol = 1.0e-10
    end

    @testset "With shift vectors" begin
        A = [1.0 0.0; 0.0 1.0]
        s1 = [0.0, 0.0]
        s2 = [0.5, 0.5]

        g1 = Rank0Gaussian(A, s1)
        g2 = Rank0Gaussian(A, s2)

        overlap = _compute_matrix_element(g1, g2)
        overlap_noshift = _compute_matrix_element(g1, g1)

        @test overlap > 0
        @test overlap > overlap_noshift
        @test isfinite(overlap)
    end

    @testset "1D case" begin
        A = [1.0;;]
        s = [0.0]
        g = Rank0Gaussian(A, s)

        overlap = _compute_matrix_element(g, g)

        expected = (π / 2.0)^(3 / 2)
        @test overlap ≈ expected rtol = 1.0e-10
    end

    @testset "Numerical stability" begin
        for scale in [0.1, 1.0, 10.0]
            A = scale * [1.0 0.0; 0.0 1.0]
            s = [0.0, 0.0]
            g = Rank0Gaussian(A, s)

            overlap = _compute_matrix_element(g, g)
            @test isfinite(overlap)
            @test overlap > 0
            @test !isnan(overlap)
        end
    end
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

        @test result < 0
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

        @test result > 0
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

@testset "Error Cases" begin

    @testset "Dimension mismatch" begin
        A1 = [1.0;;]  # 1D
        A2 = [1.0 0.0; 0.0 1.0]  # 2D
        s1 = [0.0]
        s2 = [0.0, 0.0]

        g1 = Rank0Gaussian(A1, s1)
        g2 = Rank0Gaussian(A2, s2)

        @test_throws Exception _compute_matrix_element(g1, g2)
    end
end
