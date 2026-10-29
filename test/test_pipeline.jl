using Test
using FewBodyECG
using OptimKit: LBFGS

_pipeline_lbfgs(maxiter) = LBFGS(; maxiter, gradtol = 1.0e-6, verbosity = 0, ls_verbosity = 0)

ops = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])

@testset "Pipelines" begin
    p = SVM(basis = 12, candidates = 10, scale = 1.0) →
        Refine(sweeps = 1, candidates = 15, scale = 1.0) →
        GVM(optimizer = _pipeline_lbfgs(200))
    sol = solve(ops, p)
    @test length(sol.stages) == 3
    @test sol.stages[1].method isa SVM
    @test sol.stages[3].method isa GVM
    # monotone: each stage's final energy ≤ the previous stage's
    finals = [last(s.history) for s in sol.stages]
    @test all(diff(finals) .<= 1.0e-10)
    @test sol.convergence === sol.stages[end].report
    @test occursin("→", sprint(show, MIME"text/plain"(), sol))
    # pipeline respects an outer init
    pre = solve(ops, SVM(basis = 6, candidates = 10, scale = 1.0))
    sol2 = solve(
        ops, SVM(basis = 6, candidates = 10, scale = 1.0) →
            GVM(optimizer = _pipeline_lbfgs(100)); init = pre
    )
    @test length(sol2.basis.functions) == 12
end

@testset "Warm starts continue the quasi-random sequence" begin
    svm = solve(ops, SVM(basis = 6, candidates = 4, scale = 1.0))
    @test svm.draws == 24                          # each growth step draws `candidates` points
    k = length(svm.basis.functions)
    @test FewBodyECG._solution_basis_state(svm, svm.operators).draw == svm.draws

    ref = solve(ops, Refine(1; candidates = 3, scale = 1.0); init = svm)
    @test ref.draws == svm.draws + 3k              # one sweep draws `candidates` per slot
    pipe = solve(ops, SVM(basis = 6, candidates = 4, scale = 1.0) → Refine(1; candidates = 3, scale = 1.0))
    @test pipe.draws == ref.draws
    @test pipe.E₀ == ref.E₀                        # a pipeline stage is the same warm start

    dyn = solve(
        ops, DynamicGVM(basis = k + 2, candidates = 2, scale = 1.0, optimizer = _pipeline_lbfgs(20));
        init = svm
    )
    @test dyn.draws == svm.draws + 2 * 2

    @test solve(ops, GVM(4; scale = 1.0, optimizer = _pipeline_lbfgs(20))).draws == 4
    @test solve(ops, GVM(optimizer = _pipeline_lbfgs(20)); init = svm).draws == svm.draws
end

@testset "widths = :wide" begin
    # With scale four times too large, wide widths still reach the compact
    # hydrogen ground state; linear widths mostly miss it.
    errs = map((:linear, :wide)) do widths
        solve(ops, SVM(; basis = 20, candidates = 20, scale = 4.0, widths)).E₀ + 0.5
    end
    @test errs[2] < errs[1] / 10

    sol = solve(
        ops, SVM(basis = 6, candidates = 4, scale = 1.0, widths = :wide) →
            Refine(1; candidates = 3, scale = 1.0, widths = :wide)
    )
    @test sol.E₀ < -0.49
    @test solve(ops, DynamicGVM(basis = 3, candidates = 2, scale = 1.0, widths = :wide, optimizer = _pipeline_lbfgs(20))).E₀ < -0.4
    @test solve(ops, GVM(3; scale = 1.0, widths = :wide, optimizer = _pipeline_lbfgs(20))).E₀ < -0.4
    @test_throws "omit `widths`" solve(ops, GVM(widths = :wide); init = sol)
    @test_throws "unknown widths = :log" solve(ops, SVM(basis = 2, scale = 1.0, widths = :log))
end
