using Test
using LinearAlgebra
using FewBodyECG
using QuadGK: quadgk

@testset "Permutation and contact operators" begin
    @testset "Two-particle exchange selects even and odd spatial states" begin
        basis = BasisSet(
            GaussianBase[
                Rank0Gaussian([0.8;;], [0.0]),
                Rank1Gaussian([0.8;;], [1.0], [0.0]),
                Rank2Gaussian([0.8;;], [1.0], [1.0], [0.0]),
            ]
        )
        P = PermutationOperator([1.0, 1.0], [2, 1])
        S = FewBodyECG.build_overlap_matrix(basis)
        M = FewBodyECG.build_operator_matrix(basis, P)
        @test M ≈ S * Diagonal([1, -1, 1])
        @test (S + M)[2, 2] ≈ 0 atol = 1.0e-14
        @test (S - M)[[1, 3], [1, 3]] ≈ zeros(2, 2) atol = 1.0e-14
    end

    @testset "Permutation cycles transform shifts and polarizations" begin
        masses = [1.0, 1.0, 1.0]
        P = PermutationOperator(masses, [2, 3, 1])
        Pinv = PermutationOperator(masses, [3, 1, 2])
        # Hand-derived rotation for this ordering of the mass-scaled coordinates.
        T = [-0.5 sqrt(3) / 2; -sqrt(3) / 2 -0.5]
        A = [1.1 0.2; 0.2 0.7]
        s = [0.2 -0.1 0.3; 0.4 0.1 -0.2]
        a = [0.8 0.1 0.2; 0.2 -0.3 0.5]
        b = [-0.2 0.4 0.1; 0.7 0.2 -0.3]
        gs = (Rank0Gaussian(A, s), Rank1Gaussian(A, a, s), Rank2Gaussian(A, a, b, s))
        transformed = (
            Rank0Gaussian(T' * A * T, T' * s),
            Rank1Gaussian(T' * A * T, T' * a, T' * s),
            Rank2Gaussian(T' * A * T, T' * a, T' * b, T' * s),
        )
        V = GaussianOperator(-0.7, 0.4, [1.0, -0.3])
        for i in eachindex(gs)
            @test FewBodyECG._compute_matrix_element(gs[i], gs[i], P) ≈
                FewBodyECG._compute_matrix_element(gs[i], transformed[i]) rtol = 2.0e-12
            @test FewBodyECG._compute_matrix_element(gs[i], gs[i], V * P) ≈
                FewBodyECG._compute_matrix_element(gs[i], transformed[i], V) rtol = 2.0e-12
        end
        basis = BasisSet([gs[1], Rank0Gaussian([0.6 -0.1; -0.1 1.3], -2s)])
        M = FewBodyECG.build_operator_matrix(basis, P)
        @test norm(M - M') > 1.0e-4
        @test FewBodyECG.build_operator_matrix(basis, Pinv) ≈ M'
        @test FewBodyECG.build_hamiltonian_matrix(basis, [P]) ≈ M
        MP = FewBodyECG.build_operator_matrix(basis, V * P)
        @test norm(MP - MP') > 1.0e-4
        @test FewBodyECG.build_hamiltonian_matrix(basis, [V * P]) ≈ MP
        identity = PermutationOperator(masses, [1, 2, 3])
        @test FewBodyECG.build_operator_matrix(basis, identity) ≈ FewBodyECG.build_overlap_matrix(basis)
        @test FewBodyECG.build_operator_matrix(basis, P * Pinv) ≈ FewBodyECG.build_overlap_matrix(basis)
    end

    @testset "Point delta evaluates every rank pair with its volume factor" begin
        A, B = [0.7;;], [1.2;;]
        s, t = [0.2 -0.1 0.3], [-0.1 0.3 0.2]
        a, b = [0.4 -0.2 0.7], [-0.3 0.5 0.1]
        bra = (Rank0Gaussian(A, s), Rank1Gaussian(A, a, s), Rank2Gaussian(A, a, b, s))
        ket = (Rank0Gaussian(B, t), Rank1Gaussian(B, b, t), Rank2Gaussian(B, b, a, t))
        r = [0.3, -0.2, 0.4]
        for w in (1.4, -1.4)
            x = r / w
            pref_bra = (1.0, dot(vec(a), x), dot(vec(a), x) * dot(vec(b), x))
            pref_ket = (1.0, dot(vec(b), x), dot(vec(b), x) * dot(vec(a), x))
            base = -0.8 * exp(-1.9 * sum(abs2, x) + dot(vec(s + t), x)) / abs(w)^3
            D = ContactDensityOperator([w], r; coefficient = -0.8)
            for i in eachindex(bra), j in eachindex(ket)
                @test FewBodyECG._compute_matrix_element(bra[i], ket[j], D) ≈
                    base * pref_bra[i] * pref_ket[j] rtol = 2.0e-11 atol = 1.0e-15
            end
        end
    end

    @testset "Contact leaves spectator-coordinate fluctuations" begin
        A = [0.7 0.0; 0.0 0.9]
        s = zeros(2, 3)
        a = [1.0 0.0 0.0; 0.0 0.0 1.0]
        g = Rank0Gaussian(A, s)
        p = Rank1Gaussian(A, a, s)
        d = Rank2Gaussian(A, a, a, s)
        # At x₁ = 0, the polynomial is x₂,z and its square.
        contact = ContactDensityOperator([1.0, 0.0])
        spectator = (π / 1.8)^(3 / 2)
        @test FewBodyECG._compute_matrix_element(g, g, contact) ≈ spectator
        @test FewBodyECG._compute_matrix_element(p, p, contact) ≈ spectator / 3.6
        @test FewBodyECG._compute_matrix_element(g, d, contact) ≈ spectator / 3.6
        @test FewBodyECG._compute_matrix_element(d, d, contact) ≈ 3spectator / 3.6^2
        @test FewBodyECG._compute_matrix_element(g, p, contact) ≈ 0 atol = 1.0e-14
    end

    @testset "Large shifts keep representable contact elements finite" begin
        g = Rank0Gaussian([1.0;;], [40.0])
        p = Rank1Gaussian([1.0;;], [1.0], [40.0])
        at_zero = ContactDensityOperator([1.0])
        at_one = ContactDensityOperator([1.0], [0.0, 0.0, 1.0])
        @test FewBodyECG._compute_matrix_element(g, g, at_zero) ≈ 1
        @test FewBodyECG._compute_matrix_element(p, p, at_zero) ≈ 0 atol = 1.0e-14
        @test FewBodyECG._compute_matrix_element(g, g, at_one) ≈ exp(78) rtol = 1.0e-13
        @test FewBodyECG._compute_matrix_element(p, p, at_one) ≈ exp(78) rtol = 1.0e-13

        # A large source in the constrained coordinate leaves a finite spectator.
        h = Rank0Gaussian([1.0 0.0; 0.0 1.0], [40.0 0.0 0.0; 0.2 0.0 0.0])
        @test FewBodyECG._compute_matrix_element(h, h, ContactDensityOperator([1.0, 0.0])) ≈
            (π / 2)^(3 / 2) * exp(0.02) rtol = 1.0e-13
    end

    @testset "Pair density normalizes to overlap and has contact selection rules" begin
        g = Rank0Gaussian([0.6;;], [0.0])
        p = Rank1Gaussian([0.6;;], [1.0], [0.0])
        D = ContactDensityOperator([1.3])
        S = FewBodyECG._compute_matrix_element(g, g)
        @test FewBodyECG._compute_matrix_element(g, g, D) / S ≈ (1.2 / π)^(3 / 2) / 1.3^3
        @test FewBodyECG._compute_matrix_element(p, p, D) ≈ 0 atol = 1.0e-14
        radial(r) = 4π * r^2 * FewBodyECG._compute_matrix_element(
            g, g, ContactDensityOperator([1.3], [0.0, 0.0, r])
        )
        integral, _ = quadgk(radial, 0.0, Inf; rtol = 1.0e-10)
        @test integral ≈ S rtol = 1.0e-10
        basis = BasisSet(GaussianBase[g, p])
        M = FewBodyECG.build_operator_matrix(basis, D)
        @test M ≈ M'
        @test eigmin(Symmetric(M)) ≥ -1.0e-14
    end

    @testset "Invalid permutations and densities fail clearly" begin
        @test_throws ArgumentError PermutationOperator([1.0], [1])
        @test_throws ArgumentError PermutationOperator([1.0, -1.0], [2, 1])
        @test_throws ArgumentError PermutationOperator([1.0, Inf], [2, 1])
        @test_throws ArgumentError PermutationOperator([1.0, 1.0], [1, 1])
        @test_throws ArgumentError PermutationOperator([1.0, 1.0], [0, 2])
        @test_throws ArgumentError PermutationOperator([1.0, 1.0], [1])
        @test_throws ArgumentError ContactDensityOperator([0.0])
        @test_throws ArgumentError ContactDensityOperator([NaN])
        @test_throws ArgumentError ContactDensityOperator([1.0], [0.0])
        @test_throws ArgumentError ContactDensityOperator([1.0], [0.0, Inf, 0.0])
        @test_throws ArgumentError ContactDensityOperator([1.0]; coefficient = Inf)
        g = Rank0Gaussian([1.0;;], [0.0])
        @test_throws DimensionMismatch FewBodyECG._compute_matrix_element(g, g, ContactDensityOperator([1.0, 0.0]))
        @test_throws DimensionMismatch FewBodyECG._compute_matrix_element(g, g, PermutationOperator(ones(3), [2, 1, 3]))
        @test_throws ArgumentError FewBodyECG.build_operator_matrix(BasisSet(Rank0Gaussian[]), ContactDensityOperator([1.0]))
        @test FewBodyECG._compute_matrix_element(g, g, ContactDensityOperator(view([1.0], :))) ≈ 1
    end
end
