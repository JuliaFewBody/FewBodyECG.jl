module RelativisticTests

using Test, LinearAlgebra, FewBodyECG
import ForwardDiff
using QuadGK: quadgk
using FewBodyECG: _compute_matrix_element, build_operator_matrix, build_overlap_matrix

axial_gaussian(k, a, s) = k == 0 ? Rank0Gaussian([a;;], [s]) :
    k == 1 ? Rank1Gaussian([a;;], [1.0], [s]) : Rank2Gaussian([a;;], [1.0], [1.0], [s])

# Independent spatial integration of polynomial-Gaussian derivatives in
# spherical coordinates, rather than the production Gaussian-moment formula.
function spatial_integral(k, l, a, b, sa, sb; orbit = false)
    function radial(r)
        value, _ = quadgk(-1.0, 1.0; rtol = 1.0e-10, atol = 1.0e-12) do t
            z = r * t
            P, Q = z^k, z^l
            dP = k == 0 ? 0.0 : k * z^(k - 1)
            dQ = l == 0 ? 0.0 : l * z^(l - 1)
            if orbit
                Lz, Lr = dP + sa * P, -2a * P
                Rz, Rr = -dQ - sb * Q, 2b * Q
                polynomial = r * Lz * Rz * (1 + t^2) +
                    2r^2 * t * (Lz * Rr + Lr * Rz) + 2r^3 * Lr * Rr
            else
                ddP = k < 2 ? 0.0 : k * (k - 1) * z^(k - 2)
                ddQ = l < 2 ? 0.0 : l * (l - 1) * z^(l - 2)
                lapP = ddP + 2dP * (sa - 2a * z) + P * (sa^2 - 4a * sa * z + 4a^2 * r^2 - 6a)
                lapQ = ddQ + 2dQ * (sb - 2b * z) + Q * (sb^2 - 4b * sb * z + 4b^2 * r^2 - 6b)
                polynomial = r^2 * lapP * lapQ
            end
            return polynomial * exp(-(a + b) * r^2 + (sa + sb) * z)
        end
        return 2π * value
    end
    return first(quadgk(radial, 0.0, Inf; rtol = 1.0e-9, atol = 1.0e-11))
end

@testset "Relativistic matrix elements" begin
    @testset "Fourth momentum and individual-particle masses" begin
        a = 0.7
        op = MomentumFourthOperator([1.0])
        for (g, expected) in (
                (Rank0Gaussian([a;;], zeros(1)), 7.35),
                (Rank1Gaussian([a;;], [1.0], zeros(1)), 17.15),
                (Rank2Gaussian([a;;], [1.0 0 0], [0.0 1 0], zeros(1)), 30.87),
            )
            @test _compute_matrix_element(g, g, op) / _compute_matrix_element(g, g) ≈ expected
        end
        g = Rank0Gaussian([0.4;;], zeros(1))
        mv = MassVelocityOperator([2.0, 3.0])
        @test _compute_matrix_element(g, g, mv) / _compute_matrix_element(g, g) ≈ -0.07
        three = Rank0Gaussian(0.4Matrix{Float64}(I, 2, 2), zeros(2))
        @test _compute_matrix_element(three, three, MassVelocityOperator(ones(3))) / _compute_matrix_element(three, three) ≈ -0.4
        narrow, broad = Rank0Gaussian([1.0e12;;], zeros(1)), Rank0Gaussian([1.0e-4;;], zeros(1))
        @test _compute_matrix_element(narrow, broad, op) / _compute_matrix_element(narrow, broad) ≈ 6.0e-7 rtol = 1.0e-12
    end

    @testset "Hydrogen 1s from the Gaussian transform of exp(-r)" begin
        # exp(-r) = ∫exp(-y/2-exp(-y)/4) exp(-exp(y)r²)dy/(2√π).
        ys = range(-6.0, 26.0; length = 151)
        basis = BasisSet([Rank0Gaussian([exp(y);;], zeros(1)) for y in ys])
        c = [step(ys) * exp(-y / 2 - exp(-y) / 4) / (2sqrt(π)) for y in ys]
        norm = dot(c, build_overlap_matrix(basis) * c)
        expectation(op) = dot(c, build_operator_matrix(basis, op) * c) / norm
        masses = [1.0e15, 1.0]
        _, U = FewBodyECG.jacobi_transform(masses)
        mv = expectation(MassVelocityOperator(masses))
        darwin = expectation(ContactDensityOperator(U[1, :] - U[2, :]; coefficient = π / 2))
        @test mv ≈ -0.625 atol = 5.0e-5
        @test darwin ≈ 0.5 atol = 5.0e-5
        @test mv + darwin ≈ -0.125 atol = 5.0e-5
    end

    @testset "Orbit tensor and Coulomb derivatives" begin
        g = Rank0Gaussian([1.0 0.5; 0.5 1.0], zeros(2))
        op = OrbitOrbitOperator([1.0, -1.0], [1.0, 0.0], [0.0, 1.0])
        @test _compute_matrix_element(g, g, op) / _compute_matrix_element(g, g) ≈ 1.5957691216057308
        uncoupled = Rank0Gaussian(Matrix{Float64}(I, 2, 2), zeros(2))
        @test _compute_matrix_element(uncoupled, uncoupled, op) ≈ 0 atol = 1.0e-13
        h = Rank0Gaussian([0.4;;], zeros(1))
        physical = OrbitOrbitOperator([2.0, 3.0], [1.0, -1.0])
        @test _compute_matrix_element(h, h, physical) / _compute_matrix_element(h, h) ≈ -0.17689300452932374
        small = OrbitOrbitOperator([2.0, 3.0], Float32[1.0e-25, -1.0e-25])
        @test _compute_matrix_element(h, h, small) / _compute_matrix_element(h, h) ≈ -1.7689300452932374e-51 rtol = 1.0e-6
        @test_throws ArgumentError OrbitOrbitOperator(ones(2), [1.0e308, -1.0e308])
    end

    @testset "Shifted and polarized elements against spatial quadrature" begin
        for (sa, sb) in ((0.0, 0.0), (0.6, -0.2), (2.2, 1.0), (4.0, 3.0))
            for k in 0:2, l in 0:2
                bra, ket = axial_gaussian(k, 0.7, sa), axial_gaussian(l, 1.1, sb)
                p4 = _compute_matrix_element(bra, ket, MomentumFourthOperator([1.0]))
                oo = _compute_matrix_element(bra, ket, OrbitOrbitOperator([1.0], [1.0], [-1.0]))
                @test p4 ≈ spatial_integral(k, l, 0.7, 1.1, sa, sb) rtol = 2.0e-9 atol = 2.0e-10
                @test oo ≈ spatial_integral(k, l, 0.7, 1.1, sa, sb; orbit = true) rtol = 2.0e-9 atol = 2.0e-10
            end
        end
    end

    @testset "Frames, Hermiticity, parameter derivatives, and wrappers" begin
        masses, charges = ones(3), [2.0, -1.0, -1.0]
        A = [0.8 0.2; 0.2 1.3]
        s = [0.2 -0.1 0.5; 0.1 0.3 -0.2]
        gs = [Rank0Gaussian(A, s), Rank0Gaussian([1.1 -0.1; -0.1 0.7], -s)]
        symmetry = PermutationSymmetry(masses, [[1, 3, 2]])
        for op in (MassVelocityOperator(masses), OrbitOrbitOperator(masses, charges))
            M = build_operator_matrix(BasisSet(gs), op)
            @test M ≈ M' atol = 1.0e-12
            projected = BasisSet([ProjectedGaussian(g, symmetry) for g in gs])
            @test all(isfinite, build_operator_matrix(projected, op))
            spin = SpinState([FewBodyECG.up, FewBodyECG.down, FewBodyECG.up])
            @test build_operator_matrix(BasisSet([SpinGaussian(g, spin) for g in gs]), op) ≈ M
        end
        g = Rank2Gaussian([0.7;;], [1.0 0 0], [0.0 1 0], zeros(1))
        p4(t) = begin
            f = Rank2Gaussian([t;;], g.a, g.b, g.s)
            _compute_matrix_element(f, f, MomentumFourthOperator([1.0])) / _compute_matrix_element(f, f)
        end
        @test ForwardDiff.derivative(p4, 0.7) ≈ 88.2
        oo(t) = begin
            f = Rank1Gaussian(t * A, [0.3, -0.7], zeros(2))
            _compute_matrix_element(f, f, OrbitOrbitOperator([1.0, -1.0], [1.0, 0.0], [0.0, 1.0])) / _compute_matrix_element(f, f)
        end
        @test ForwardDiff.derivative(oo, 1.0) ≈ 1.5oo(1.0) rtol = 1.0e-11
        g0 = first(gs)
        coupling(c) = _compute_matrix_element(g0, g0, OrbitOrbitOperator(masses, charges; coefficient = c))
        @test ForwardDiff.derivative(coupling, 0.0) ≈ coupling(1.0)
        @test _compute_matrix_element(g0, g0, MomentumFourthOperator(zeros(2))) == 0
        @test_throws ArgumentError MassVelocityOperator([1.0])
        @test_throws ArgumentError MassVelocityOperator([0.0, 1.0])
        @test_throws ArgumentError MomentumFourthOperator([NaN])
        @test_throws ArgumentError MomentumFourthOperator([1.0]; coefficient = Inf)
        @test_throws ArgumentError OrbitOrbitOperator(zeros(2), ones(2), ones(2))
        @test_throws DimensionMismatch OrbitOrbitOperator(ones(2), ones(1), ones(2))
        @test_throws DimensionMismatch OrbitOrbitOperator(ones(3), ones(2))
        @test_throws ArgumentError OrbitOrbitOperator(ones(2), [NaN, 1.0])
        @test_throws DimensionMismatch _compute_matrix_element(g0, g0, MomentumFourthOperator([1.0]))
        @test_throws DimensionMismatch _compute_matrix_element(g0, g0, OrbitOrbitOperator([1.0], [1.0], [1.0]))
        @test MomentumFourthOperator(view([1.0, 2.0], 1:1)).u == [1.0]
    end
end

end
