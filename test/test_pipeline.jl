using Test
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "Pipelines" begin
    p = SVM(basis = 12, candidates = 10, scale = 1.0) →
        Refine(sweeps = 1, candidates = 15, scale = 1.0) →
        Variational(basis = 12, maxiter = 200)
    sol = solve(ops, p)
    @test length(sol.stages) == 3
    @test sol.stages[1].method isa SVM
    @test sol.stages[3].method isa Variational
    # monotone: each stage's final energy ≤ the previous stage's
    finals = [last(s.energies) for s in sol.stages]
    @test all(diff(finals) .<= 1.0e-10)
    @test sol.convergence === sol.stages[end].report
    @test occursin("→", sprint(show, MIME"text/plain"(), sol))
    # pipeline respects an outer init
    pre = solve(ops, SVM(basis = 6, candidates = 10, scale = 1.0))
    sol2 = solve(
        ops, SVM(basis = 6, candidates = 10, scale = 1.0) →
            Variational(basis = 12, maxiter = 100); init = pre
    )
    @test length(sol2.basis.functions) == 12
end
