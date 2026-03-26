using Test
using FewBodyECG
using LinearAlgebra
using Random

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

end

@testset "Rank1/Rank2 matrix elements" begin

    @testset "Rank1 overlap supports shifts" begin
        A = [1.0 0.2; 0.2 1.5]
        s1 = [0.1, -0.3]
        s2 = [-0.2, 0.4]
        p = [0.6, -0.1]
        q = [-0.3, 0.8]

        g1 = Rank1Gaussian(A, p, s1)
        g2 = Rank1Gaussian(A, q, s2)

        val = _compute_matrix_element(g1, g2)

        S = A + A
        R = inv(S)
        t = s1 + s2
        Rt = R * t
        n = size(R, 1)
        I0 = exp(0.25 * t' * R * t) * (π^n / det(S))^(3 / 2)
        expected = (0.5 * dot(p, R * q) + 0.25 * dot(p, Rt) * dot(q, Rt)) * I0

        @test isapprox(val, expected; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(val, _compute_matrix_element(g2, g1); rtol = 1.0e-12, atol = 1.0e-12)
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

    @testset "Matrix polarizations are backward compatible with single-column vectors" begin
        A = [1.0 0.2; 0.2 1.5]
        B = [0.9 0.1; 0.1 1.2]
        s = [0.0, 0.0]
        a = [0.5, -0.4]
        b = [-0.2, 0.7]
        c = [0.3, 0.6]
        d = [-0.1, -0.8]
        K = KineticOperator([0.5 0.0; 0.0 0.6])
        V = CoulombOperator(1.0, [1.0, 0.0])

        g1v = Rank2Gaussian(A, a, b, s)
        g2v = Rank2Gaussian(B, c, d, s)
        g1m = Rank2Gaussian(A, reshape(a, :, 1), reshape(b, :, 1), s)
        g2m = Rank2Gaussian(B, reshape(c, :, 1), reshape(d, :, 1), s)

        @test _compute_matrix_element(g1v, g2v) ≈ _compute_matrix_element(g1m, g2m) rtol = 1.0e-12
        @test _compute_matrix_element(g1v, g2v, K) ≈ _compute_matrix_element(g1m, g2m, K) rtol = 1.0e-12
        @test _compute_matrix_element(g1v, g2v, V) ≈ _compute_matrix_element(g1m, g2m, V) rtol = 1.0e-12
    end

    @testset "Incompatible polarization components throw DimensionMismatch" begin
        A = [1.0 0.2; 0.2 1.5]
        s = [0.0, 0.0]

        g1 = Rank1Gaussian(A, [0.5 0.1; -0.4 0.3], s)
        g2 = Rank1Gaussian(A, [0.2; 0.7], s)

        @test_throws DimensionMismatch _compute_matrix_element(g1, g2)
        @test_throws DimensionMismatch _compute_matrix_element(g1, g2, KineticOperator([0.5 0.0; 0.0 0.6]))
        @test_throws DimensionMismatch _compute_matrix_element(g1, g2, CoulombOperator(1.0, [1.0, 0.0]))
    end

    @testset "Rank2 kinetic/coulomb agree with paper formulas for matrix polarizations" begin
        Random.seed!(7)
        s(v, M, w) = tr(transpose(v) * M * w)

        function paper_rank2_kin(B, c, d, A, a, b, K)
            R = inv(A + B)
            n = size(R, 1)
            M0 = (π^n / det(A + B))^(3 / 2)
            M2 = 0.25 * (s(a, R, b) * s(c, R, d) + s(a, R, c) * s(b, R, d) + s(a, R, d) * s(b, R, c)) * M0
            T1 = 6 * tr(B * K * A * R) * M2
            T2 = 0.5 * (s(a, K, c) * s(b, R, d) + s(a, K, d) * s(b, R, c) + s(b, K, c) * s(a, R, d) + s(b, K, d) * s(a, R, c)) * M0

            M = R * B * K * A * R
            T3 = 0.5 * (
                s(a, M, b) * s(c, R, d) + s(a, M, c) * s(b, R, d) + s(a, M, d) * s(b, R, c) +
                    s(b, M, a) * s(c, R, d) + s(b, M, c) * s(a, R, d) + s(b, M, d) * s(a, R, c) +
                    s(c, M, a) * s(b, R, d) + s(c, M, b) * s(a, R, d) + s(c, M, d) * s(a, R, b) +
                    s(d, M, a) * s(b, R, c) + s(d, M, b) * s(a, R, c) + s(d, M, c) * s(a, R, b)
            ) * M0

            RBK = R * B * K
            T4 = -0.5 * (
                s(a, RBK, b) * s(c, R, d) + s(b, RBK, a) * s(c, R, d) +
                    s(c, RBK, a) * s(b, R, d) + s(c, RBK, b) * s(a, R, d) +
                    s(d, RBK, a) * s(b, R, c) + s(d, RBK, b) * s(a, R, c)
            ) * M0

            KAR = K * A * R
            T5 = -0.5 * (
                s(c, KAR, a) * s(b, R, d) + s(c, KAR, b) * s(a, R, d) + s(c, KAR, d) * s(a, R, b) +
                    s(d, KAR, a) * s(b, R, c) + s(d, KAR, b) * s(a, R, c) + s(d, KAR, c) * s(a, R, b)
            ) * M0

            return T1 + T2 + T3 + T4 + T5
        end

        function paper_rank2_coul(B, c, d, A, a, b, w, coef)
            R = inv(A + B)
            n = size(R, 1)
            M0 = (π^n / det(A + B))^(3 / 2)
            β = 1 / dot(w, R * w)
            Rw = R * w
            proj(x, y) = dot(transpose(x) * Rw, transpose(y) * Rw)

            M2 = 0.25 * (s(a, R, b) * s(c, R, d) + s(a, R, c) * s(b, R, d) + s(a, R, d) * s(b, R, c)) * M0
            term1 = 2 * sqrt(β / π) * M2

            term2 = -2 * sqrt(β / π) * β / 3 * 0.25 * (
                proj(a, b) * s(c, R, d) +
                    proj(a, c) * s(b, R, d) +
                    proj(a, d) * s(b, R, c) +
                    proj(b, c) * s(a, R, d) +
                    proj(b, d) * s(a, R, c) +
                    proj(c, d) * s(a, R, b)
            ) * M0

            term3 = 2 * sqrt(β / π) * β^2 / 10 * 0.5 * (
                proj(a, b) * proj(c, d) +
                    proj(a, c) * proj(b, d) +
                    proj(a, d) * proj(b, c)
            ) * M0

            return coef * (term1 + term2 + term3)
        end

        for n in (2, 3), _ in 1:3
            X = randn(n, n)
            Y = randn(n, n)
            Z = randn(n, n)
            A = X' * X + I
            B = Y' * Y + I
            K = Z' * Z
            w = randn(n)

            a = randn(n, 3)
            b = randn(n, 3)
            c = randn(n, 3)
            d = randn(n, 3)
            s0 = zeros(n)

            bra = Rank2Gaussian(A, a, b, s0)
            ket = Rank2Gaussian(B, c, d, s0)
            opK = KineticOperator(K)
            opV = CoulombOperator(-1.3, w)

            gotK = _compute_matrix_element(bra, ket, opK)
            gotV = _compute_matrix_element(bra, ket, opV)
            refK = paper_rank2_kin(B, c, d, A, a, b, K)
            refV = paper_rank2_coul(B, c, d, A, a, b, w, -1.3)

            @test gotK ≈ refK rtol = 1.0e-10 atol = 1.0e-12
            @test gotV ≈ refV rtol = 1.0e-10 atol = 1.0e-12
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

@testset "Gaussian Potential ⟨g′|V|g⟩" begin

    @testset "Analytic check: 1D zero-shift Gaussians" begin
        # For g_i = exp(-α_i x²) and V = exp(-γ x²) with w=[1],
        # the matrix element is exactly (π/(α₁+α₂+γ))^(1/2).
        α1, α2, γ = 1.0, 1.5, 0.8
        g1 = Rank0Gaussian([α1;;], [0.0])
        g2 = Rank0Gaussian([α2;;], [0.0])
        op = GaussianOperator(1.0, γ, [1.0])
        expected = (π / (α1 + α2 + γ))^(3 / 2)
        @test _compute_matrix_element(g1, g2, op) ≈ expected rtol = 1.0e-12
    end

    @testset "γ → 0 limit equals overlap" begin
        # As γ → 0, exp(-γ r²) → 1 so the matrix element → overlap.
        A1 = [1.0 0.0; 0.0 2.0]
        A2 = [1.5 0.1; 0.1 1.5]
        s1 = [0.1, -0.2]
        s2 = [-0.3, 0.4]
        g1 = Rank0Gaussian(A1, s1)
        g2 = Rank0Gaussian(A2, s2)
        w = [1.0, 0.0]
        op_tiny = GaussianOperator(1.0, 1.0e-14, w)
        @test _compute_matrix_element(g1, g2, op_tiny) ≈ _compute_matrix_element(g1, g2) rtol = 1.0e-10
    end

    @testset "Symmetry ⟨g′|V|g⟩ = ⟨g|V|g′⟩" begin
        A1 = [1.0 0.0; 0.0 2.0]
        A2 = [1.5 0.0; 0.0 1.5]
        s1 = [0.1, 0.2]
        s2 = [-0.1, 0.3]
        g1 = Rank0Gaussian(A1, s1)
        g2 = Rank0Gaussian(A2, s2)
        op = GaussianOperator(-1.0, 2.0, [1.0, 0.0])
        @test _compute_matrix_element(g1, g2, op) ≈ _compute_matrix_element(g2, g1, op) rtol = 1.0e-12
    end

    @testset "Scaling with coefficient" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        w = [1.0, 0.0]
        op1 = GaussianOperator(-1.0, 1.0, w)
        op2 = GaussianOperator(-3.0, 1.0, w)
        @test _compute_matrix_element(g, g, op2) ≈ 3 * _compute_matrix_element(g, g, op1) rtol = 1.0e-12
    end

    @testset "Larger γ → smaller matrix element (more localised)" begin
        A = [1.0;;]
        s = [0.0]
        g = Rank0Gaussian(A, s)
        w = [1.0]
        op_narrow = GaussianOperator(1.0, 10.0, w)
        op_wide   = GaussianOperator(1.0, 0.1, w)
        @test _compute_matrix_element(g, g, op_narrow) < _compute_matrix_element(g, g, op_wide)
    end

    @testset "Operators tuple interface" begin
        masses = [1.0e15, 1.0]
        ops = Operators(masses)
        ops += ("Gaussian", 1, 2, -1.0, 2.0)
        @test length(ops) == 1
        @test ops[1] isa GaussianOperator
        @test ops[1].coefficient ≈ -1.0
        @test ops[1].γ ≈ 2.0
    end

    @testset "Operators tuple: γ ≤ 0 throws" begin
        ops = Operators([1.0e15, 1.0])
        @test_throws ArgumentError ops += ("Gaussian", 1, 2, -1.0, 0.0)
        @test_throws ArgumentError ops += ("Gaussian", 1, 2, -1.0, -1.0)
    end
end
