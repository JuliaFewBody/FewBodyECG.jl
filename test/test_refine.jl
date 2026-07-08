using Test
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "Refine" begin
    # Deliberately poor starting basis (wrong scale), then refine at scale 1.
    poor = solve(ops, SVM(basis = 10, candidates = 5, scale = 4.0))
    ref = solve(ops, Refine(sweeps = 2, candidates = 25, scale = 1.0); init = poor)
    @test ref isa Solution
    @test ref.E₀ <= poor.E₀ + 1.0e-12               # never raises the energy
    @test ref.E₀ < poor.E₀ - 1.0e-3                 # actually improves a bad basis
    @test length(ref.basis.functions) == length(poor.basis.functions)
    @test ref.stages[end].method isa Refine
    @test length(energies(ref, length(ref.stages))) == 2    # one entry per sweep

    # standalone Refine without a basis is a user error
    @test_throws ArgumentError solve(ops, Refine(1))
end

@testset "Refine after a zero-addition stage" begin
    poor = solve(ops, SVM(basis = 10, candidates = 5, scale = 4.0))
    stuck = solve(ops, SVM(basis = 5, candidates = 5, scale = 1.0, indep_tol = 1.0); init = poor)
    @test isempty(energies(stuck))
    ref = solve(ops, Refine(sweeps = 1, candidates = 5, scale = 1.0); init = stuck)
    @test ref isa Solution
    @test ref.E₀ <= stuck.E₀ + 1.0e-12
    @test ref.convergence.criterion in (:saturation, :max_steps)
end
