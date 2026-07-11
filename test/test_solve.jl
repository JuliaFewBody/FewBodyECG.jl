using Test
using LinearAlgebra
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "solve dispatch + SVM" begin
    sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0))
    @test sol isa Solution
    @test sol.E₀ ≈ -0.5 atol = 5.0e-2              # hydrogen anchor
    @test sol.E₀ > -0.5 - 1.0e-6                    # variational bound
    @test length(sol.stages) == 1
    @test sol.stages[1].method isa SVM
    @test all(diff(energies(sol)) .<= 1.0e-9)       # monotone selection
    @test sol.convergence.criterion in (:saturation, :max_steps)
    @test FewBodyECG.SATURATION_CAVEAT in sol.convergence.notes
    @test size(sol.coefficients, 2) == length(sol.E)
    # coefficients are S-orthonormal
    S = build_overlap_matrix(sol.basis)
    @test sol.coefficients' * S * sol.coefficients ≈ I atol = 1.0e-6

    # accept-first strategy
    sol1 = solve(ops, SVM(basis = 15, candidates = 1, scale = 1.0))
    @test sol1.E₀ < -0.4

    # default method + raw-terms entry + :auto scale
    @test solve(ops).E₀ < -0.4
    @test solve(ops.terms, SVM(basis = 10, candidates = 5, scale = 1.0)) isa Solution
    @test_throws ArgumentError solve(ops.terms, SVM(basis = 5))   # :auto needs masses

    # deterministic (Halton)
    @test solve(ops, SVM(basis = 15, candidates = 10, scale = 1.0)).E₀ ==
        solve(ops, SVM(basis = 15, candidates = 10, scale = 1.0)).E₀

    # excited state targeting
    sol2 = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0); state = 2)
    @test sol2.state == 2 && sol2.E₀ == sol2.E[2] && sol2.E₀ > sol2.E[1]

    # warm start grows an existing basis (unshifted candidates may occasionally
    # be linearly dependent and skipped — the report documents this honestly)
    small = solve(ops, SVM(basis = 5, candidates = 10, scale = 1.0))
    bigger = solve(ops, SVM(basis = 10, candidates = 10, scale = 1.0); init = small)
    @test 5 < length(bigger.basis.functions) <= 15
    @test bigger.E₀ <= small.E₀ + 1.0e-12

    # early stop: an impossible independence floor rejects every candidate,
    # leaving the warm-start basis intact with an honest :early_stop report
    stuck = solve(
        ops, SVM(basis = 5, candidates = 5, scale = 1.0, indep_tol = 1.0);
        init = small
    )
    @test stuck.convergence.criterion == :early_stop
    @test !converged(stuck)
    @test length(stuck.basis.functions) == length(small.basis.functions)
    @test any(occursin("no admissible candidate", n) for n in stuck.convergence.notes)
end
