using Test, LinearAlgebra, FewBodyECG
using FewBodyECG: up, down, build_overlap_matrix, build_operator_matrix, build_hamiltonian_matrix

@testset "Spin and isospin exchange" begin
    @testset "Exact two-nucleon exchange matrices and channels" begin
        p, n = FewBodyECG.proton, FewBodyECG.neutron
        spins = [SpinState([up, up]), SpinState([up, down]), SpinState([down, up]), SpinState([down, down])]
        isos = [IsospinState([p, p]), IsospinState([p, n]), IsospinState([n, p]), IsospinState([n, n])]
        g = Rank0Gaussian([0.7;;], [0.2])
        basis = BasisSet([SpinGaussian(g, s; isospin = t) for s in spins for t in isos])
        S0 = (π / 1.4)^(3 / 2) * exp(0.4^2 / (4 * 1.4))
        swap = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
        identity = Matrix{Float64}(I, 4, 4)
        Ps, Pt = SpinExchangeOperator(1, 2), IsospinExchangeOperator(1, 2)
        Ms, Mt = build_operator_matrix(basis, Ps), build_operator_matrix(basis, Pt)
        S = build_overlap_matrix(basis)
        @test S ≈ S0 * Matrix{Float64}(I, 16, 16)
        @test Ms ≈ S0 * kron(swap, identity)
        @test Mt ≈ S0 * kron(identity, swap)
        @test build_operator_matrix(basis, Ps * Pt) ≈ S0 * kron(swap, swap)
        @test build_operator_matrix(basis, Ps * Ps) ≈ S
        @test build_operator_matrix(basis, Pt * Pt) ≈ S

        V = GaussianOperator(-2.0, 0.4, [1.0])
        V0 = -2 * (π / 1.8)^(3 / 2) * exp(0.4^2 / (4 * 1.8))
        @test build_operator_matrix(basis, V * Ps) ≈ V0 * kron(swap, identity)
        @test build_operator_matrix(basis, V * Pt) ≈ V0 * kron(identity, swap)
        @test build_hamiltonian_matrix(basis, [V * Ps * Pt]) ≈ V0 * kron(swap, swap)
        singlet, triplet = [0, 1, -1, 0] / sqrt(2), [0, 1, 1, 0] / sqrt(2)
        for (cs, es) in ((singlet, -1), (triplet, 1)), (ct, et) in ((singlet, -1), (triplet, 1))
            c = kron(cs, ct)
            @test dot(c, Ms * c) / dot(c, S * c) ≈ es
            @test dot(c, Mt * c) / dot(c, S * c) ≈ et
            @test dot(c, (2Ms - S) * c) / dot(c, S * c) ≈ 2es - 1
            @test dot(c, (2Mt - S) * c) / dot(c, S * c) ≈ 2et - 1
        end
    end

    @testset "Pauli sign and radial/spatial exchange at every orbital rank" begin
        p, n = FewBodyECG.proton, FewBodyECG.neutron
        gs = (
            Rank0Gaussian([0.8;;], [0.0]), Rank1Gaussian([0.8;;], [1.0], [0.0]),
            Rank2Gaussian([0.8;;], [1 0 0], [0 1 0], [0.0]),
        )
        spin_states = [SpinState([up, down]), SpinState([down, up])]
        iso_states = [IsospinState([p, n]), IsospinState([n, p])]
        Ps, Pt = SpinExchangeOperator(1, 2), IsospinExchangeOperator(1, 2)
        Pr = PermutationOperator(ones(2), [2, 1])
        V = GaussianOperator(1.7, 0.3, [1.0])
        singlet, triplet = [1, -1] / sqrt(2), [1, 1] / sqrt(2)
        for (g, parity) in zip(gs, (1, -1, 1))
            basis = BasisSet([SpinGaussian(g, s; isospin = t) for s in spin_states for t in iso_states])
            S = build_overlap_matrix(basis)
            M = build_operator_matrix(basis, Pr * Ps * Pt)
            for (s, es) in ((singlet, -1), (triplet, 1)), (t, et) in ((singlet, -1), (triplet, 1))
                c = kron(s, t)
                @test dot(c, M * c) / dot(c, S * c) ≈ parity * es * et
            end
            # All factors act on the same pair and commute in this central term.
            @test build_operator_matrix(basis, V * Ps * Pt * Pr) ≈
                build_operator_matrix(basis, V * Pr * Pt * Ps)
        end
        # Shifted cross-rank orbitals use the same orbital element as the raw VPr.
        rngs = (
            Rank0Gaussian([0.7;;], [0.2]), Rank1Gaussian([0.9;;], [1.0], [-0.3]),
            Rank2Gaussian([1.1;;], [1.0], [1.0], [0.1]),
        )
        for bra in rngs, ket in rngs
            b = SpinGaussian(bra, SpinState([up, down]); isospin = IsospinState([p, n]))
            k = SpinGaussian(ket, SpinState([down, up]); isospin = IsospinState([n, p]))
            @test FewBodyECG._compute_matrix_element(b, k, V * Ps * Pt * Pr) ≈
                FewBodyECG._compute_matrix_element(bra, ket, V * Pr)
        end
    end

    @testset "Noncommuting three-particle exchanges preserve order" begin
        bits = [
            (up, up, up), (up, up, down), (up, down, up), (up, down, down),
            (down, up, up), (down, up, down), (down, down, up), (down, down, down),
        ]
        g = Rank0Gaussian([0.7 0.1; 0.1 1.2], zeros(2, 3))
        basis = BasisSet([SpinGaussian(g, SpinState(collect(s))) for s in bits])
        P12, P23 = SpinExchangeOperator(1, 2), SpinExchangeOperator(2, 3)
        M = build_operator_matrix(basis, P12 * P23)
        S0 = (π^2 / (4 * (0.7 * 1.2 - 0.1^2)))^(3 / 2)
        expected = [br == (kt[3], kt[1], kt[2]) for br in bits, kt in bits]
        @test M ≈ S0 * expected
        @test !isapprox(M, M')
        @test build_operator_matrix(basis, P23 * P12) ≈ M'
        @test build_hamiltonian_matrix(basis, [P12 * P23]) ≈ M
        @test_throws ArgumentError build_operator_matrix(basis, SpinExchangeOperator(1, 4))
        p, n = FewBodyECG.proton, FewBodyECG.neutron
        ibasis = BasisSet(
            [
                SpinGaussian(
                    g, SpinState([up, up, up]);
                    isospin = IsospinState([s == up ? p : n for s in state])
                ) for state in bits
            ]
        )
        T12, T23 = IsospinExchangeOperator(1, 2), IsospinExchangeOperator(2, 3)
        @test build_operator_matrix(ibasis, T12 * T23) ≈ S0 * expected
        @test build_operator_matrix(ibasis, T23 * T12) ≈ S0 * expected'
        @test_throws ArgumentError build_operator_matrix(ibasis, IsospinExchangeOperator(1, 4))
    end

    @testset "Isospin is a spectator for existing spin operators" begin
        p, n = FewBodyECG.proton, FewBodyECG.neutron
        s = SpinState([up, up])
        g = Rank0Gaussian([0.8;;], [0.3 0.2 -0.1])
        h = Rank0Gaussian([1.1;;], [-0.2 0.1 0.4])
        b, k = SpinGaussian(g, s), SpinGaussian(h, s)
        bp = SpinGaussian(g, s; isospin = IsospinState([p, n]))
        kp = SpinGaussian(h, s; isospin = IsospinState([p, n]))
        kn = SpinGaussian(h, s; isospin = IsospinState([n, p]))
        ops = (
            GaussianOperator(1.0, 0.3, [1.0]), GaussianTensorOperator(1.0, 0.3, [1.0], 1, 2),
            GaussianSpinOrbitOperator(1.0, 0.3, [1.0], 1, 2),
        )
        @test FewBodyECG._compute_matrix_element(bp, kn) ≈ 0
        for op in ops
            @test FewBodyECG._compute_matrix_element(bp, kp, op) ≈ FewBodyECG._compute_matrix_element(b, k, op)
            @test FewBodyECG._compute_matrix_element(bp, kn, op) ≈ 0
            @test_throws ArgumentError FewBodyECG._compute_matrix_element(b, kp, op)
        end
        polarized = SpinGaussian(Rank1Gaussian([0.8;;], [1.0], [0.2]), s; isospin = IsospinState([p, n]))
        for op in ops[2:3]
            @test FewBodyECG._compute_matrix_element(polarized, kp, op) ≈
                conj(FewBodyECG._compute_matrix_element(kp, polarized, op))
            @test isfinite(FewBodyECG._compute_matrix_element(polarized, polarized, op))
        end
        d = Rank2Gaussian([0.8;;], [1 0 0], [0 1 0], [0.0])
        ds = SpinGaussian(d, s; isospin = IsospinState([p, n]))
        # For xy and ↑↑ the traceless tensor is (z²-r²/3)/4. Wick moments
        # under exp(-(1.6+0.3)r²) give -variance³/3 times the Gaussian integral.
        variance = 1 / (2 * 1.9)
        @test FewBodyECG._compute_matrix_element(ds, ds, ops[2]) ≈
            -(π / 1.9)^(3 / 2) * variance^3 / 3
        dxx = SpinGaussian(
            Rank2Gaussian([1.1;;], [1 0 0], [1 0 0], [0.0]), s;
            isospin = IsospinState([p, n])
        )
        complex_element = FewBodyECG._compute_matrix_element(ds, dxx, ops[3])
        # Lz[x² exp(-1.1r²)] = 2im xy exp(-1.1r²), total spin Sz = 1.
        @test complex_element ≈ 2im * (π / 2.2)^(3 / 2) / (2 * 2.2)^2
        @test abs(imag(complex_element)) > 1.0e-10
        @test real(complex_element) ≈ 0 atol = 1.0e-14
        @test complex_element ≈ conj(FewBodyECG._compute_matrix_element(dxx, ds, ops[3]))
        for op in ops[2:3]
            @test FewBodyECG._compute_matrix_element(ds, kp, op) ≈
                conj(FewBodyECG._compute_matrix_element(kp, ds, op)) atol = 1.0e-14
            @test FewBodyECG._compute_matrix_element(ds, polarized, op) ≈
                conj(FewBodyECG._compute_matrix_element(polarized, ds, op)) atol = 1.0e-14
        end
    end

    @testset "Spin-orbit kernel agrees with direct angular-momentum action" begin
        spin = SpinState([up, up])
        a, b = [0.2 0.1 -0.3], [-0.4 0.5 0.2]
        bra = SpinGaussian(Rank0Gaussian([0.8;;], a), spin)
        ket = SpinGaussian(Rank0Gaussian([1.1;;], b), spin)
        C = 2.2
        M = (π / C)^(3 / 2) * exp(sum(abs2, a + b) / (4C))
        expected = -im * (a[1] * b[2] - a[2] * b[1]) * M / (2C)
        @test FewBodyECG._compute_matrix_element(
            bra, ket,
            GaussianSpinOrbitOperator(1.0, 0.3, [1.0], 1, 2)
        ) ≈ expected

        px = SpinGaussian(Rank1Gaussian([0.8;;], [1 0 0], [0.0]), spin)
        py = SpinGaussian(Rank1Gaussian([1.1;;], [0 1 0], [0.0]), spin)
        # Lz[y exp(-1.1r²)] = -im x exp(-1.1r²). Angular momentum is
        # invariant under rescaling the pair coordinate and its conjugate momentum.
        for scale in (1.0, sqrt(2), -2.0)
            op = GaussianSpinOrbitOperator(1.0, 0.3 / scale^2, [scale], 1, 2)
            @test FewBodyECG._compute_matrix_element(px, py, op) ≈
                -im * (π / C)^(3 / 2) / (2C)
        end

        # Rotate one relative coordinate along w; the orthogonal coordinate
        # contributes only a spectator Gaussian integral.
        w = [1.0, 2.0]
        u = w / norm(w)
        spin3 = SpinState([up, up, up])
        bx = SpinGaussian(Rank1Gaussian(0.8 * Matrix{Float64}(I, 2, 2), u * [1 0 0], zeros(2, 3)), spin3)
        ky = SpinGaussian(Rank1Gaussian(1.1 * Matrix{Float64}(I, 2, 2), u * [0 1 0], zeros(2, 3)), spin3)
        op = GaussianSpinOrbitOperator(1.0, 0.3 / dot(w, w), w, 1, 2)
        @test FewBodyECG._compute_matrix_element(bx, ky, op) ≈
            -im * (π^2 / (C * 1.9))^(3 / 2) / (2C)
        @test_throws ArgumentError GaussianSpinOrbitOperator(1.0, 0.3, [0.0], 1, 2)
    end

    @testset "Validation and backward compatibility" begin
        p, n = FewBodyECG.proton, FewBodyECG.neutron
        @test_throws ArgumentError IsospinProjection(0)
        g = Rank0Gaussian([1.0;;], [0.0])
        s = SpinState([up, down])
        old = SpinGaussian(g, s)
        @test FewBodyECG._compute_matrix_element(old, old) ≈ (π / 2)^(3 / 2)
        @test_throws ArgumentError SpinExchangeOperator(0, 1)
        @test_throws ArgumentError IsospinExchangeOperator(1, 1)
        @test_throws ArgumentError SpinGaussian(g, SpinState([up]))
        @test_throws ArgumentError SpinGaussian(g, s; isospin = IsospinState([p]))
        @test_throws ArgumentError FewBodyECG._compute_matrix_element(old, old, IsospinExchangeOperator(1, 2))
        for op in (SpinExchangeOperator(1, 2), IsospinExchangeOperator(1, 2))
            @test_throws ArgumentError FewBodyECG._compute_matrix_element(g, g, op)
            @test_throws ArgumentError FewBodyECG._compute_matrix_element(g, g, GaussianOperator(1.0, 0.2, [1.0]) * op)
        end
        tagged = SpinGaussian(g, s; isospin = IsospinState([p, n]))
        @test_throws ArgumentError FewBodyECG._compute_matrix_element(old, tagged)
        from_view = SpinGaussian(g, s; isospin = IsospinState(view([p, n], :)))
        @test FewBodyECG._compute_matrix_element(tagged, from_view) ≈ (π / 2)^(3 / 2)
    end
end
