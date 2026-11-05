module PolarizedSVMTests

using Test, LinearAlgebra, FewBodyECG
using FewBodyECG: BasisState, _draw_candidate!, _candidate_columns, commit!,
    rebuild_without, _solution_basis_state, build_operator_matrix,
    build_overlap_matrix, build_hamiltonian_matrix

@testset "Adaptive Rank1 SVM" begin
    masses = [1.0e15, 1.0]
    H = Kinetic(masses) + Coulomb([1.0, -1.0])

    @testset "Automatic 2p widths and requested basis sizes" begin
        p = solve(H, SVM(12; rank = 1, candidates = 25, indep_tol = 1.0e-8))
        @test length(p.basis) == 12
        @test all(g -> g isa Rank1Gaussian, p.basis)
        @test p.E₀ ≈ -0.125 atol = 2.0e-5
        @test FewBodyECG._rms_pair_max(p, masses) ≈ sqrt(30) rtol = 0.005
        @test any(n -> occursin("rms pair distance 5.48", n), p.convergence.notes)
        @test all(diff(energy_history(p)) .≤ 1.0e-10)
        @test p.draws == 12 * 25
        @test dot(p.coefficients[:, 1], build_overlap_matrix(p.basis) * p.coefficients[:, 1]) ≈ 1 atol = 1.0e-10
        for g in p.basis
            @test iszero(g.s)
            @test g.a ≈ [0.0 0.0 1.0]
        end
        larger = solve(H, SVM(15; rank = 1, candidates = 25, indep_tol = 1.0e-8))
        @test length(larger.basis) == 15
        @test larger.E₀ ≤ p.E₀ + 1.0e-10
        @test all(larger.basis[i].A == p.basis[i].A && larger.basis[i].a == p.basis[i].a for i in eachindex(p.basis.functions))

        s = solve(H, SVM(12; candidates = 25, indep_tol = 1.0e-8))
        D = build_operator_matrix(s.basis, p.basis, DipoleOperator(masses, [1.0, -1.0]))
        d = abs(dot(s.coefficients[:, 1], D * p.coefficients[:, 1]))
        @test d ≈ 128sqrt(2) / 243 rtol = 0.003
        @test 2 * (p.E₀ - s.E₀) * d^2 ≈ 8192 / 19683 rtol = 0.006
    end

    @testset "Warm starts, refinement, and pipelines" begin
        first_stage = solve(H, SVM(6; rank = 1, candidates = 15, scale = 3.0, indep_tol = 1.0e-8))
        grown = solve(H, SVM(3; candidates = 15, scale = 3.0, indep_tol = 1.0e-8); init = first_stage)
        direct = solve(H, SVM(9; rank = 1, candidates = 15, scale = 3.0, indep_tol = 1.0e-8))
        @test length(grown.basis) == 9
        @test grown.E₀ ≈ direct.E₀ atol = 1.0e-10
        @test grown.draws == direct.draws == 9 * 15
        refined = solve(H, Refine(1; candidates = 5, indep_tol = 1.0e-8); init = grown)
        @test all(g -> g isa Rank1Gaussian, refined.basis)
        @test length(refined.basis) == length(grown.basis)
        @test refined.E₀ ≤ grown.E₀ + 1.0e-10
        @test refined.draws == grown.draws + 5 * length(grown.basis)
        pipe = solve(H, SVM(6; rank = 1, candidates = 15, scale = 3.0) → Refine(1; candidates = 5, scale = 3.0))
        @test all(g -> g isa Rank1Gaussian, pipe.basis)
        @test length(pipe.stages) == 2
        @test_throws ArgumentError solve(H, SVM(1; rank = 0); init = grown)
        @test_throws ArgumentError solve(H, SVM(1; rank = 1); init = solve(H, SVM(2; scale = 1.0)))
        @test_throws ArgumentError solve(H, SVM(2; rank = 2))
        mixed = BasisSet(GaussianBase[Rank0Gaussian([1.0;;], zeros(1)), Rank1Gaussian([1.0;;], [1.0], zeros(1))])
        @test_throws ArgumentError FewBodyECG.BasisState(mixed.functions, first(FewBodyECG._to_jacobi(H)))
    end

    @testset "Many-body global vectors and cached matrices" begin
        terms = Operator[KineticOperator([0.5 0; 0 0.5]), OscillatorOperator(0.5, [1.0, 0.0]), OscillatorOperator(0.5, [0.0, 1.0])]
        weights = [[1.0, 0.0], [0.0, 1.0]]
        st = BasisState(Rank1Gaussian)
        polarizations = Vector{Float64}[]
        for _ in 1:5
            g = _draw_candidate!(st, 1.0, FewBodyECG.HaltonSample(), :wide, weights, 2)
            @test iszero(g.a[:, 1:2])
            @test norm(g.a[:, 3]) ≈ 1
            push!(polarizations, g.a[:, 3])
            cols = _candidate_columns(g, st.basis, terms)
            commit!(st, g, cols)
        end
        @test polarizations[1] != polarizations[2]
        basis = BasisSet(st.basis)
        @test st.S ≈ build_overlap_matrix(basis)
        @test st.H ≈ build_hamiltonian_matrix(basis, terms)
        rebuilt = BasisState(copy(st.basis), terms)
        @test rebuilt.S ≈ st.S
        removed = rebuild_without(st, 2)
        @test all(g -> g isa Rank1Gaussian, removed.basis)
        @test removed.S ≈ st.S[[1, 3, 4, 5], [1, 3, 4, 5]]
    end
end

end
