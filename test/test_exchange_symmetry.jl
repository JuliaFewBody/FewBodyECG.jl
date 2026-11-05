module ExchangeSymmetryTests

using Test, LinearAlgebra, FewBodyECG
import ForwardDiff
using FewBodyECG: build_operator_matrix, build_overlap_matrix, build_hamiltonian_matrix
import FewBodyECG: _compute_matrix_element

@testset "Exchange symmetry" begin
    @testset "Group character and pointwise projection" begin
        masses = ones(3)
        symmetry = PermutationSymmetry(masses, [[1, 3, 2]])
        @test length(symmetry.permutations) == 2
        @test symmetry == PermutationSymmetry(masses, [[1, 2, 3], [1, 3, 2]])
        equivalent = PermutationSymmetry(masses, [[1, 2, 3], [1, 3, 2]])
        @test hash(symmetry) == hash(equivalent)
        @test get(Dict(symmetry => 7), equivalent, 0) == 7
        A = [0.8 0.2; 0.2 1.3]
        s = [0.1 -0.2 0.3; -0.1 0.0 0.2]
        a = [0.4, -0.7]
        P = PermutationOperator(masses, [1, 3, 2])
        x = [0.3 -0.4 0.6; 0.1 0.2 -0.3]
        for g in (Rank0Gaussian(A, s), Rank1Gaussian(A, a, s), Rank2Gaussian(A, a, -a, s))
            projected = ProjectedGaussian(g, symmetry)
            ψ = Wavefunction(BasisSet([projected]), [1.0])
            raw = Wavefunction(BasisSet([g]), [1.0])
            @test ψ(x) ≈ (raw(x) + raw(P.T * x)) / 2
            @test ψ(P.T * x) ≈ ψ(x)
            expected = (_compute_matrix_element(g, g) + _compute_matrix_element(g, g, P)) / 2
            @test _compute_matrix_element(projected, projected) ≈ expected
            @test _compute_matrix_element(projected, projected, P) ≈ expected
            r, density = radial_profile(ψ; npoints = 10, normalize = false)
            @test length(r) == length(density) == 10
        end
        anti = PermutationSymmetry(masses, [[2, 1, 3], [1, 3, 2]]; signs = [-1, -1])
        @test length(anti.permutations) == 6
        @test sum(anti.signs) == 0
        @test_throws ArgumentError PermutationSymmetry(masses, [[2, 3, 1]]; signs = [-1])
        @test_throws ArgumentError PermutationSymmetry(masses, [[2, 1, 3], [1, 3, 2]]; signs = [1, -1])
        @test_throws ArgumentError PermutationSymmetry([1.0, 2.0], [[2, 1]])
        @test_throws ArgumentError PermutationSymmetry(ones(2), [[1, 1]])
        @test_throws DimensionMismatch PermutationSymmetry(masses, [[1, 3, 2]]; signs = [1, 1])
        @test_throws ArgumentError PermutationSymmetry(masses, [[1, 3, 2]]; signs = [0])
        @test_throws DimensionMismatch ProjectedGaussian(Rank0Gaussian([1.0;;], zeros(1)), symmetry)
        raw = Rank0Gaussian(A, s)
        @test_throws MethodError ProjectedGaussian(raw, symmetry, [raw, raw])
    end

    @testset "Both projectors in operators and cross-sector transitions" begin
        masses = ones(3)
        even = PermutationSymmetry(masses, [[1, 3, 2]])
        odd = PermutationSymmetry(masses, [[1, 3, 2]]; signs = [-1])
        g = Rank0Gaussian([0.8 0.2; 0.2 1.3], [0.1 -0.2 0.3; -0.1 0.0 0.2])
        h = Rank1Gaussian([1.1 -0.1; -0.1 0.7], [0.5, 0.8], zeros(2))
        Pg, Ph = ProjectedGaussian(g, even), ProjectedGaussian(h, odd)
        @test _compute_matrix_element(Pg, Ph) ≈ 0 atol = 1.0e-13
        P = PermutationOperator(masses, [1, 3, 2])
        transformed(g::Rank0Gaussian) = Rank0Gaussian(P.T' * g.A * P.T, P.T' * g.s)
        transformed(g::Rank1Gaussian) = Rank1Gaussian(P.T' * g.A * P.T, P.T' * g.a, P.T' * g.s)
        for op in (DipoleOperator([0.7, -0.4]), ContactDensityOperator([0.3, 1.0]), GaussianOperator(1.2, 0.6, [1.0, 0.2]))
            reference = sum(sign * _compute_matrix_element(a, b, op) for a in (g, transformed(g)) for (b, sign) in ((h, 1), (transformed(h), -1))) / 4
            @test _compute_matrix_element(Pg, Ph, op) ≈ reference atol = 1.0e-12
            @test build_operator_matrix(BasisSet([Pg]), BasisSet([Ph]), op)[1, 1] ≈ reference atol = 1.0e-12
        end
        basis = BasisSet([Pg, ProjectedGaussian(Rank0Gaussian([1.2 -0.1; -0.1 0.6], zeros(2)), even)])
        noninvariant = Operator[KineticOperator(masses), GaussianOperator(-0.7, 0.4, [1.0, 0.2])]
        @test !FewBodyECG._hamiltonian_invariant(noninvariant, even)
        @test build_hamiltonian_matrix(basis, noninvariant) ≈ sum(build_operator_matrix(basis, op) for op in noninvariant)
        H = Kinetic(masses) + Coulomb([1.0, -1.0, -1.0])
        ops = first(FewBodyECG._to_jacobi(H))
        @test FewBodyECG._hamiltonian_invariant(ops, even)
        @test build_hamiltonian_matrix(basis, H) ≈ sum(build_operator_matrix(basis, op) for op in ops) rtol = 1.0e-12
    end

    @testset "Projected stochastic selection, observables, and warm starts" begin
        masses = ones(3)
        H = Kinetic(masses) + Coulomb([1.0, -1.0, -1.0])
        symmetry = PermutationSymmetry(masses, [[1, 3, 2]])
        p = PermutationOperator(masses, [1, 3, 2])
        sol = solve(H, SVM(12; candidates = 15); symmetry)
        @test all(g -> g isa ProjectedGaussian, sol.basis)
        @test -0.262005071 < energy(sol) < -0.25
        c = sol.coefficients[:, 1]
        S = build_overlap_matrix(sol.basis)
        @test dot(c, S * c) ≈ 1 atol = 1.0e-10
        @test dot(c, build_operator_matrix(sol.basis, p) * c) ≈ 1 atol = 1.0e-10
        ψ = wavefunction(sol)
        x = [0.1 -0.2 0.4; 0.3 0.2 -0.1]
        @test ψ(x) ≈ ψ(p.T * x)
        st = FewBodyECG._solution_basis_state(sol, sol.operators)
        @test st.S ≈ S rtol = 1.0e-11
        @test st.H ≈ build_hamiltonian_matrix(sol.basis, H) rtol = 1.0e-11
        grown = solve(H, SVM(2; candidates = 5); init = sol)
        @test all(g -> g isa ProjectedGaussian, grown.basis)
        @test energy(grown) ≤ energy(sol) + 1.0e-10
        refined = solve(H, Refine(1; candidates = 3); init = sol)
        @test energy(refined) ≤ energy(sol) + 1.0e-10
        @test length(refined.basis) == length(sol.basis)
        @test_throws ArgumentError solve(H, SVM(1); init = sol, symmetry = PermutationSymmetry(masses, [[1, 3, 2]]; signs = [-1]))
        @test_throws ArgumentError solve(H, GVM(3); symmetry)
        @test_throws ArgumentError solve(H, GVM(); init = sol)
        @test_throws ArgumentError solve(Kinetic([2.0, 2.0, 2.0]) + Coulomb([1.0, -1.0, -1.0]), SVM(1); symmetry)
    end

    @testset "Projected Rank1 sector and null candidates" begin
        masses = ones(2)
        H = Kinetic(masses) + Coulomb([1.0, -1.0])
        odd = PermutationSymmetry(masses, [[2, 1]]; signs = [-1])
        sol = solve(H, SVM(8; rank = 1, candidates = 15); symmetry = odd)
        @test energy(sol) ≈ -0.0625 atol = 0.0002
        @test all(g -> g.orbital isa Rank1Gaussian, sol.basis)
        refined = solve(H, Refine(1; candidates = 2); init = sol)
        @test energy(refined) ≤ energy(sol) + 1.0e-10
        @test all(g -> g.orbital isa Rank1Gaussian && g.symmetry == odd, refined.basis)
        pipe = solve(H, SVM(3; rank = 1, scale = 2.0, candidates = 5) → Refine(1; scale = 2.0, candidates = 2); symmetry = odd)
        @test length(pipe.stages) == 2
        @test all(g -> g.symmetry == odd, pipe.basis)
        @test_throws ArgumentError solve(H, SVM(2; scale = 1.0); symmetry = odd)
        even = PermutationSymmetry(masses, [[2, 1]])
        p = PermutationOperator(masses, [2, 1])
        null = ProjectedGaussian(Rank0Gaussian([0.5;;], zeros(1)), odd)
        @test abs(_compute_matrix_element(null, null)) < 1.0e-14
        @test _compute_matrix_element(ProjectedGaussian(Rank0Gaussian([0.5;;], zeros(1)), even), null, p) ≈ 0 atol = 1.0e-14
    end

    @testset "Ps2 charge-exchange sectors" begin
        masses, charges = ones(4), [1.0, -1.0, 1.0, -1.0]
        generators = [[3, 2, 1, 4], [1, 4, 3, 2], [2, 1, 4, 3]]
        even = PermutationSymmetry(masses, generators)
        odd = PermutationSymmetry(masses, reverse(generators); signs = [-1, 1, 1])
        @test length(even.permutations) == length(odd.permutations) == 8
        A = [0.8 0.1 -0.2; 0.1 1.1 0.3; -0.2 0.3 0.9]
        shift = [0.1 0.0 0.2; -0.2 0.1 0.4; 0.3 -0.1 0.2]
        g = ProjectedGaussian(Rank0Gaussian(A, shift), even)
        h = ProjectedGaussian(Rank1Gaussian(A, [0.3, -0.2, 0.6], shift), odd)
        x = [0.3 0.1 0.4; 0.1 -0.3 0.2; -0.2 0.1 0.5]
        ψg, ψh = Wavefunction(BasisSet([g]), [1.0]), Wavefunction(BasisSet([h]), [1.0])
        for (permutation, sign) in zip(generators, [1, 1, -1])
            T = PermutationOperator(masses, permutation).T
            @test ψg(T * x) ≈ ψg(x)
            @test ψh(T * x) ≈ sign * ψh(x)
        end
        D = DipoleOperator(masses, charges)
        @test _compute_matrix_element(g, g, D) ≈ 0 atol = 1.0e-13
        @test abs(_compute_matrix_element(g, h, D)) > 1.0e-6
        H = Kinetic(masses) + Coulomb(charges)
        @test FewBodyECG._hamiltonian_invariant(first(FewBodyECG._to_jacobi(H)), even)
    end

    @testset "Projection shortcuts preserve scale, scalar types, and perturbations" begin
        symmetry = PermutationSymmetry(ones(3), [[1, 3, 2]])
        A = [0.3 0.1; 0.1 2.0]
        g = ProjectedGaussian(Rank0Gaussian(A, zeros(2)), symmetry)
        basis = BasisSet([g])
        S = _compute_matrix_element(g, g)
        for op in (CoulombOperator(1.0f0, Float32[1, 0.2]), GaussianOperator(-1.0f0, 0.3f0, Float32[1, 0.2]))
            @test build_hamiltonian_matrix(basis, [op]) ≈ build_operator_matrix(basis, op) rtol = 1.0e-12
        end

        broad = ProjectedGaussian(Rank0Gaussian(1.0e-16 * A, zeros(2)), symmetry)
        V = GaussianOperator(-1.0, 1.0e-16, [1.0, 0.2])
        @test !FewBodyECG._hamiltonian_invariant([V], symmetry)
        normalized = build_hamiltonian_matrix(BasisSet([broad]), [V])[1, 1] / _compute_matrix_element(broad, broad)
        @test normalized ≈ -0.36044555671 atol = 1.0e-10
        tiny = CoulombOperator(1.0, 1.0e-8 * [1.0, 0.2])
        @test !FewBodyECG._hamiltonian_invariant([tiny], symmetry)
        @test build_hamiltonian_matrix(basis, [tiny])[1, 1] / S ≈ 1.1544895356910754e8 rtol = 1.0e-10

        w = [1.0, 0.2]
        v = symmetry.transforms[2]' * w
        operators(t) = Operator[CoulombOperator(1 + t, w .+ zero(t)), CoulombOperator(one(t), v .+ zero(t))]
        @test FewBodyECG._hamiltonian_invariant(operators(0.0), symmetry)
        @test !FewBodyECG._hamiltonian_invariant(operators(ForwardDiff.Dual(0.0, 1.0)), symmetry)
        perturbation(t) = build_hamiltonian_matrix(basis, operators(t))[1, 1] / S
        @test ForwardDiff.derivative(perturbation, 0.0) ≈ 1.1544895356910754 rtol = 1.0e-10

        # Basis derivatives still use an invariant ordinary Hamiltonian safely.
        function width(t)
            f = ProjectedGaussian(Rank0Gaussian(t * A, zeros(2)), symmetry)
            return build_hamiltonian_matrix(BasisSet([f]), operators(0.0))[1, 1] / _compute_matrix_element(f, f)
        end
        @test ForwardDiff.derivative(width, 1.0) ≈ 1.1544895356910754 rtol = 1.0e-10
    end
end

end
