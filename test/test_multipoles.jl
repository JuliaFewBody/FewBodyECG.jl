module MultipoleTests

using Test, LinearAlgebra, FewBodyECG
using QuadGK: quadgk
using OffsetArrays: OffsetVector, OffsetArray
import ForwardDiff
using FewBodyECG: build_operator_matrix, build_overlap_matrix, build_hamiltonian_matrix,
    solve_generalized_eigenproblem
import FewBodyECG: _compute_matrix_element

@testset "Electric multipoles and transitions" begin
    @testset "Rectangular s-p transitions" begin
        s = BasisSet([Rank0Gaussian([a;;], zeros(1)) for a in (0.5, 1.0)])
        p = BasisSet([Rank1Gaussian([b;;], [1.0], zeros(1)) for b in (0.4, 0.8, 1.3)])
        D = DipoleOperator([1.0])
        M = build_operator_matrix(s, p, D)
        # Direct ∫z² exp(-(a+b)r²)d³r; catches swapped axes or missing rank factors.
        expected = [π^(3 / 2) / (2 * (a + b)^(5 / 2)) for a in (0.5, 1.0), b in (0.4, 0.8, 1.3)]
        @test size(M) == (2, 3)
        @test M ≈ expected rtol = 1.0e-13
        @test build_operator_matrix(p, s, D) ≈ M'
        @test build_operator_matrix(s, D) == build_operator_matrix(s, s, D)
        @test iszero(build_operator_matrix(s, s, D))
        empty_basis = BasisSet(Rank0Gaussian[])
        @test_throws ArgumentError build_operator_matrix(empty_basis, p, D)
        @test_throws ArgumentError build_operator_matrix(s, empty_basis, D)

        # Complex spin-orbit matrices must retain every rectangular entry.
        spin = SpinState([FewBodyECG.up, FewBodyECG.up])
        x = SpinGaussian(Rank1Gaussian([0.7;;], [1.0 0 0], zeros(1)), spin)
        y = SpinGaussian(Rank1Gaussian([0.9;;], [0.0 1 0], zeros(1)), spin)
        b = BasisSet([x, y])
        so = GaussianSpinOrbitOperator(1.0, 0.4, [1.0], 1, 2)
        block = build_operator_matrix(BasisSet([x]), b, so)
        @test size(block) == (1, 2)
        @test !iszero(imag(block[1, 2]))
        @test block ≈ build_operator_matrix(b, so)[1:1, :]
        @test build_operator_matrix(b, BasisSet([x]), so) ≈ block'
    end

    @testset "Shifted moments at every rank against quadrature" begin
        a, b = 0.7, 1.1
        s, t = [0.2 -0.1 0.4], [-0.1 0.3 -0.15]
        orbital(A, shift, rank) = rank == 0 ? Rank0Gaussian([A;;], shift) :
            rank == 1 ? Rank1Gaussian([A;;], [1.0], shift) :
            Rank2Gaussian([A;;], [1.0], [1.0], shift)
        # Independent 1D integration, including the polynomial basis factors.
        moment(axis, power) = first(quadgk(x -> x^power * exp(-(a + b) * x^2 + (s[axis] + t[axis]) * x), -Inf, Inf; rtol = 1.0e-12))
        for rb in 0:2, rk in 0:2
            g, h = orbital(a, s, rb), orbital(b, t, rk)
            base = [0, 0, rb + rk]
            integral(powers) = prod(moment(axis, powers[axis]) for axis in 1:3)
            for axis in 1:3
                powers = copy(base)
                powers[axis] += 1
                @test _compute_matrix_element(g, h, DipoleOperator([0.7]; component = axis, coefficient = -1.3)) ≈
                    -0.91 * integral(powers) rtol = 1.0e-10 atol = 1.0e-12
            end
            for (i, j) in ((1, 1), (3, 3), (1, 3))
                powers = copy(base)
                powers[i] += 1
                powers[j] += 1
                reference = 3 * integral(powers)
                if i == j
                    for axis in 1:3
                        powers = copy(base)
                        powers[axis] += 2
                        reference -= integral(powers)
                    end
                end
                @test _compute_matrix_element(g, h, QuadrupoleOperator([0.8;;]; components = (i, j), coefficient = -0.5)) ≈
                    -0.4 * reference rtol = 1.0e-10 atol = 1.0e-12
            end
        end
    end

    @testset "Physical COM and tensor conventions" begin
        # For masses (2,1), charges (2,-1): D=(4/3)r12, and G_Q=-1/3
        # in x=√(2/3)r12. A charged system exposes a missing COM subtraction.
        g = Rank0Gaussian([0.5;;], [0.0 0.0 0.4])
        norm = _compute_matrix_element(g, g)
        @test _compute_matrix_element(g, g, DipoleOperator([2.0, 1.0], [2.0, -1.0])) / norm ≈
            (4 / sqrt(6)) * 0.4
        @test iszero(_compute_matrix_element(g, g, DipoleOperator([2.0, 1.0], [2.0, 1.0])))
        pz = Rank1Gaussian([0.5;;], [1.0], zeros(1))
        normp = _compute_matrix_element(pz, pz)
        Q = QuadrupoleOperator([2.0, 1.0], [2.0, -1.0])
        @test _compute_matrix_element(pz, pz, Q) / normp ≈ -2 / 3
        @test sum(_compute_matrix_element(pz, pz, QuadrupoleOperator([2.0, 1.0], [2.0, -1.0]; components = (axis, axis))) for axis in 1:3) ≈ 0 atol = 1.0e-13
        @test _compute_matrix_element(Rank0Gaussian([0.5;;], zeros(1)), pz, Q) ≈ 0 atol = 1.0e-13

        # Two Jacobi coordinates: off-diagonal G entries contribute twice.
        h = Rank0Gaussian([1.0 0.2; 0.2 0.8], [0.3 -0.2 0.4; -0.1 0.5 0.2])
        G = [0.7 -0.3; -0.3 0.2]
        μ = [
            0.17105263157894737 -0.17105263157894737 0.1842105263157895;
            -0.10526315789473684 0.35526315789473684 0.07894736842105263
        ]
        # μ=(2A)⁻¹s, calculated independently for this fixture.
        @test _compute_matrix_element(h, h, QuadrupoleOperator(G; components = (1, 2))) / _compute_matrix_element(h, h) ≈
            3 * dot(μ[:, 1], G * μ[:, 2])
    end

    @testset "Validation, generic inputs, and outer differentiation" begin
        @test_throws ArgumentError DipoleOperator(Float64[])
        @test_throws ArgumentError DipoleOperator([NaN])
        @test_throws ArgumentError DipoleOperator([1.0]; component = 0)
        @test_throws ArgumentError DipoleOperator([1.0]; coefficient = Inf)
        @test_throws ArgumentError QuadrupoleOperator([1.0 2.0])
        @test_throws ArgumentError QuadrupoleOperator([1.0 0.2; 0.0 1.0])
        @test_throws ArgumentError QuadrupoleOperator([Inf;;])
        @test_throws ArgumentError QuadrupoleOperator([1.0;;]; components = (1, 4))
        @test_throws ArgumentError DipoleOperator([1.0, -1.0], [1.0, -1.0])
        @test_throws DimensionMismatch DipoleOperator([1.0, 2.0], [1.0])
        @test_throws ArgumentError QuadrupoleOperator([1.0, 2.0], [1.0, NaN])
        g = Rank0Gaussian([1.0;;], zeros(1))
        @test_throws DimensionMismatch _compute_matrix_element(g, g, DipoleOperator([1.0, 2.0]))
        @test_throws DimensionMismatch _compute_matrix_element(g, g, QuadrupoleOperator(Matrix{Float64}(I, 2, 2)))
        w = [0.4, -0.7]
        h = Rank0Gaussian(Matrix{Float64}(I, 2, 2), [0.1 0.0 0.2; 0.3 0.0 -0.4])
        @test _compute_matrix_element(h, h, DipoleOperator(view(w, :))) ≈ _compute_matrix_element(h, h, DipoleOperator(w))
        G = [0.7 0.1; 0.1 -0.3]
        @test _compute_matrix_element(h, h, QuadrupoleOperator(view(G, :, :))) ≈ _compute_matrix_element(h, h, QuadrupoleOperator(G))
        masses, charges = [2.0, 1.0], [2.0, -1.0]
        @test _compute_matrix_element(g, g, DipoleOperator(view(masses, :), view(charges, :))) ≈ _compute_matrix_element(g, g, DipoleOperator(masses, charges))
        @test_throws ArgumentError DipoleOperator(OffsetVector(w, -1))
        @test_throws ArgumentError QuadrupoleOperator(OffsetArray([1.0;;], 0:0, 0:0))
        spin = SpinState([FewBodyECG.up, FewBodyECG.down])
        gp = SpinGaussian(g, spin)
        pp = SpinGaussian(Rank1Gaussian([1.0;;], [1.0], zeros(1)), spin)
        @test _compute_matrix_element(gp, pp, DipoleOperator([1.0])) ≈ π^(3 / 2) / (2 * 2^(5 / 2))

        # An outer width derivative must compose with the polarized shift tags.
        f(a) = _compute_matrix_element(Rank0Gaussian([a;;], zeros(1)), Rank1Gaussian([0.8;;], [1.0], zeros(1)), DipoleOperator([1.0]))
        @test ForwardDiff.derivative(f, 0.5) ≈ -5π^(3 / 2) / (4 * 1.3^(7 / 2)) rtol = 1.0e-12
        @test ForwardDiff.derivative(a -> ForwardDiff.derivative(f, a), 0.5) ≈ 35π^(3 / 2) / (8 * 1.3^(9 / 2)) rtol = 1.0e-12
        q(a) = _compute_matrix_element(
            Rank2Gaussian([a;;], [1.0], [1.0], zeros(1)),
            Rank2Gaussian([0.8;;], [1.0], [1.0], zeros(1)), QuadrupoleOperator([1.0;;])
        )
        @test ForwardDiff.derivative(q, 0.5) ≈ -13.5π^(3 / 2) / 1.3^(11 / 2) rtol = 1.0e-12
    end

    @testset "Hydrogen spectroscopy and parity" begin
        masses, charges = [1.0e15, 1.0], [1.0, -1.0]
        μ = prod(masses) / sum(masses)
        H = Kinetic(masses) + Coulomb(charges)
        widths_s = sort(exp10.(range(-3.5, 4, length = 28)); by = a -> abs(log10(a) + 0.5))
        widths_p = sort(exp10.(range(-3.5, 2.5, length = 26)); by = a -> abs(log10(a) + 1.25))
        function level(basis)
            S = build_overlap_matrix(basis)
            D = Diagonal(1 ./ sqrt.(diag(S)))
            @test cond(D * S * D) < 1.0e12
            E, c = solve_generalized_eigenproblem(D * build_hamiltonian_matrix(basis, H) * D, D * S * D)
            return (; energy = E[1], c = D * c[:, 1])
        end
        basis_s = BasisSet([Rank0Gaussian([a;;], zeros(1)) for a in widths_s])
        basis_p = BasisSet([Rank1Gaussian([a;;], [1.0], zeros(1)) for a in widths_p])
        s, p = level(basis_s), level(basis_p)
        @test s.energy ≈ -μ / 2 atol = 1.0e-6
        @test p.energy ≈ -μ / 8 atol = 1.0e-7
        Dz = build_operator_matrix(basis_s, basis_p, DipoleOperator(masses, charges))
        d = dot(s.c, Dz * p.c)
        @test abs(d) ≈ 128sqrt(2) / (243μ) rtol = 1.0e-5
        ψs, ψp = Wavefunction(basis_s, s.c), Wavefunction(basis_p, p.c)
        radial_dipole = 4π / 3 * μ^(3 / 2) * first(quadgk(r -> r^3 * ψs([sqrt(μ) * r]) * ψp([sqrt(μ) * r]), 0, Inf; rtol = 1.0e-10))
        @test radial_dipole ≈ d rtol = 1.0e-10
        @test 2 * (p.energy - s.energy) * abs2(d) ≈ 8192 / (19683μ) rtol = 2.0e-5
        @test iszero(build_operator_matrix(basis_s, basis_s, DipoleOperator(masses, charges)))
        @test iszero(build_operator_matrix(basis_s, basis_p, DipoleOperator(masses, charges; component = 1)))
        @test iszero(build_operator_matrix(basis_s, basis_p, QuadrupoleOperator(masses, charges)))
        Q = build_operator_matrix(basis_p, QuadrupoleOperator(masses, charges))
        charge_factor = (masses[2] - masses[1]) / sum(masses)
        @test dot(p.c, Q * p.c) ≈ 24charge_factor / μ^2 rtol = 1.0e-5
    end
end

end
