using Test
using LinearAlgebra
using ForwardDiff
using FewBodyECG
import FewBodyECG: _compute_matrix_element

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "GVM and DynamicGVM" begin
    sol = solve(ops, GVM(basis = 8, scale = 1.0, maxiter = 300))
    @test sol.E₀ ≈ -0.5 atol = 1.0e-2
    @test sol.E₀ > -0.5 - 1.0e-6
    @test sol.convergence.criterion in (:stationarity, :max_steps)
    @test sol.convergence.gradnorm isa Float64
    @test sol.convergence.window == 0
    @test !isempty(energies(sol))

    # warm start from a stochastic run must not be worse than the start
    svm = solve(ops, SVM(basis = 8, candidates = 10, scale = 1.0))
    ref = solve(ops, GVM(maxiter = 200); init = svm)
    @test ref.E₀ <= svm.E₀ + 1.0e-10
    @test length(ref.basis.functions) == 8
    @test ref.stages[1].method isa GVM

    # an explicit warm-start size is allowed only when it matches
    matched = solve(ops, GVM(basis = 8, maxiter = 10); init = svm)
    @test length(matched.basis.functions) == 8

    # init size mismatch is a clear user error
    @test_throws ArgumentError solve(ops, GVM(basis = 5); init = svm)
    # scale has no meaning when every Gaussian comes from init
    @test_throws ArgumentError solve(ops, GVM(scale = 1.0); init = svm)
    # a cold joint optimization cannot infer its basis size
    @test_throws ArgumentError solve(ops, GVM(maxiter = 1))

    g = solve(ops, DynamicGVM(basis = 5, candidates = 5, scale = 1.0))
    @test g.E₀ < -0.45
    @test length(energies(g)) == length(g.basis.functions)
end

@testset "DynamicGVM init sizing" begin
    seed = solve(ops, SVM(basis = 4, candidates = 10, scale = 1.0))
    @test_throws ArgumentError solve(
        ops, DynamicGVM(basis = 4, scale = 1.0); init = seed
    )
    @test_throws ArgumentError solve(
        ops, DynamicGVM(basis = 3, scale = 1.0); init = seed
    )
    g = solve(ops, DynamicGVM(basis = 6, candidates = 5, scale = 1.0); init = seed)
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

@testset "NumericalPotential ForwardDiff compatibility" begin
    numerical = NumericalPotential(r -> exp(-2r^2), [1.0])
    analytic = GaussianOperator(1.0, 2.0, [1.0])

    numerical_element(θ) = begin
        g = Rank0Gaussian([exp(θ);;], [0.0])
        _compute_matrix_element(g, g, numerical)
    end
    analytic_element(θ) = begin
        g = Rank0Gaussian([exp(θ);;], [0.0])
        _compute_matrix_element(g, g, analytic)
    end

    θ = 0.1
    @test numerical_element(θ) ≈ analytic_element(θ) rtol = 1.0e-8
    numerical_gradient = ForwardDiff.derivative(numerical_element, θ)
    analytic_gradient = ForwardDiff.derivative(analytic_element, θ)
    @test isfinite(numerical_gradient)
    @test numerical_gradient ≈ analytic_gradient rtol = 1.0e-6
end
