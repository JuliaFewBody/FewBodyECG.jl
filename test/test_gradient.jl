using Test
using LinearAlgebra
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "Variational and GrowVariational" begin
    sol = solve(ops, Variational(basis = 8, scale = 1.0, maxiter = 300))
    @test sol.E₀ ≈ -0.5 atol = 1.0e-2
    @test sol.E₀ > -0.5 - 1.0e-6
    @test sol.convergence.criterion in (:stationarity, :max_steps)
    @test sol.convergence.gradnorm isa Float64
    @test sol.convergence.window == 0
    @test !isempty(energies(sol))

    # warm start from a stochastic run must not be worse than the start
    svm = solve(ops, SVM(basis = 8, candidates = 10, scale = 1.0))
    ref = solve(ops, Variational(basis = 8, maxiter = 200); init = svm)
    @test ref.E₀ <= svm.E₀ + 1.0e-10
    @test length(ref.basis.functions) == 8

    # init size mismatch is a clear user error
    @test_throws ArgumentError solve(ops, Variational(basis = 5); init = svm)

    g = solve(ops, GrowVariational(basis = 5, candidates = 5, scale = 1.0))
    @test g.E₀ < -0.45
    @test length(energies(g)) == length(g.basis.functions)
end

@testset "GrowVariational init sizing" begin
    seed = solve(ops, SVM(basis = 4, candidates = 10, scale = 1.0))
    @test_throws ArgumentError solve(
        ops, GrowVariational(basis = 4, scale = 1.0); init = seed
    )
    @test_throws ArgumentError solve(
        ops, GrowVariational(basis = 3, scale = 1.0); init = seed
    )
    g = solve(ops, GrowVariational(basis = 6, candidates = 5, scale = 1.0); init = seed)
    @test length(g.basis.functions) == 6
    @test g.E₀ <= seed.E₀ + 1.0e-10
    @test !isnan(something(g.convergence.gradnorm, NaN))
end

@testset "engine cold-start shift_init" begin
    terms = ops.terms
    n_dim = 1
    # legacy path: zeros
    basis_z, _, _ = FewBodyECG._variational_engine(
        terms, 1, nothing, 1.0, 0, 1.0e-6, false; shift_init = :zeros
    )
    @test all(iszero, first(basis_z.functions).s)
    # new-API path: qmc (generally nonzero)
    basis_q, _, _ = FewBodyECG._variational_engine(
        terms, 1, nothing, 1.0, 0, 1.0e-6, false; shift_init = :qmc
    )
    @test !all(iszero, first(basis_q.functions).s)
end
