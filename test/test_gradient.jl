using Test
using LinearAlgebra
using ForwardDiff
using FewBodyECG
using OptimKit: ConjugateGradient, GradientDescent, LBFGS
import FewBodyECG: _compute_matrix_element

quiet_lbfgs(maxiter; gradtol = 1.0e-6) =
    LBFGS(; maxiter, gradtol, verbosity = 0, ls_verbosity = 0)

struct PrimalOnlyOperator <: Operator end
_compute_matrix_element(::Rank0Gaussian{Float64}, ::Rank0Gaussian{Float64}, ::PrimalOnlyOperator) = 0.0

struct NonfiniteGradientOperator <: Operator end
function _compute_matrix_element(
        ::Rank0Gaussian{T}, ::Rank0Gaussian{T}, ::NonfiniteGradientOperator
    ) where {T <: ForwardDiff.Dual}
    return zero(T) / zero(T)
end
_compute_matrix_element(::Rank0Gaussian{Float64}, ::Rank0Gaussian{Float64}, ::NonfiniteGradientOperator) = 0.0

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

@testset "gradient solvers accept OptimKit algorithms" begin
    for optimizer in (
            GradientDescent(; maxiter = 0, gradtol = 1.0e-6, verbosity = 0),
            ConjugateGradient(; maxiter = 0, gradtol = 1.0e-6, verbosity = 0),
            LBFGS(; maxiter = 0, gradtol = 1.0e-6, verbosity = 0),
        )
        result = try
            solve(ops, GVM(basis = 1, scale = 1.0, optimizer = optimizer))
        catch e
            e
        end
        @test result isa Solution
        result isa Solution && @test result.stages[1].method.optimizer === optimizer
    end

    optimizer = GradientDescent(; maxiter = 0, gradtol = 1.0e-6, verbosity = 0)
    result = try
        solve(
            ops,
            DynamicGVM(basis = 1, candidates = 1, scale = 1.0, optimizer = optimizer),
        )
    catch e
        e
    end
    @test result isa Solution
    result isa Solution && @test result.stages[1].method.optimizer === optimizer
end

@testset "gradient solver verbosity" begin
    optimizer = GradientDescent(
        ; maxiter = 1, gradtol = 0.0, verbosity = 0, ls_verbosity = 0
    )
    @test_logs solve(
        ops, GVM(basis = 1, scale = 1.0, optimizer = optimizer); verbose = false
    )
    @test_logs (:info, r"GVM: iteration 1") solve(
        ops, GVM(basis = 1, scale = 1.0, optimizer = optimizer); verbose = true
    )
    @test_logs (
        :info, r"DynamicGVM: step 1/1, iteration 1",
    ) (:info, r"DynamicGVM: completed step 1/1") solve(
        ops,
        DynamicGVM(basis = 1, candidates = 1, scale = 1.0, optimizer = optimizer);
        verbose = true,
    )

    talkative = LBFGS(; maxiter = 0, gradtol = Inf, verbosity = 2, ls_verbosity = 0)
    @test_logs (
        :info, r"LBFGS: initializing",
    ) (:info, r"LBFGS: converged") solve(
        ops, GVM(basis = 1, scale = 1.0, optimizer = talkative); verbose = false
    )
end

@testset "GVM history records accepted optimizer iterations" begin
    optimizer = GradientDescent(
        ; maxiter = 1, gradtol = 0.0, verbosity = 0, ls_verbosity = 0
    )
    sol = solve(ops, GVM(basis = 1, scale = 1.0, optimizer = optimizer))
    @test length(energies(sol)) == 2 # initial point and one accepted iteration
end

@testset "GVM and DynamicGVM" begin
    sol = solve(ops, GVM(basis = 8, scale = 1.0, optimizer = quiet_lbfgs(300)))
    @test sol.E₀ ≈ -0.5 atol = 1.0e-2
    @test sol.E₀ > -0.5 - 1.0e-6
    @test sol.convergence.criterion in (:stationarity, :max_steps)
    @test sol.convergence.gradnorm isa Float64
    @test sol.convergence.window == 0
    @test !isempty(energies(sol))

    # warm start from a stochastic run must not be worse than the start
    svm = solve(ops, SVM(basis = 8, candidates = 10, scale = 1.0))
    ref = solve(ops, GVM(optimizer = quiet_lbfgs(200)); init = svm)
    @test ref.E₀ <= svm.E₀ + 1.0e-10
    @test length(ref.basis.functions) == 8
    @test ref.stages[1].method isa GVM

    # an explicit warm-start size is allowed only when it matches
    matched = solve(ops, GVM(basis = 8, optimizer = quiet_lbfgs(10)); init = svm)
    @test length(matched.basis.functions) == 8

    # init size mismatch is a clear user error
    @test_throws ArgumentError solve(ops, GVM(basis = 5); init = svm)
    # scale has no meaning when every Gaussian comes from init
    @test_throws ArgumentError solve(ops, GVM(scale = 1.0); init = svm)
    # a cold joint optimization cannot infer its basis size
    @test_throws ArgumentError solve(ops, GVM(optimizer = quiet_lbfgs(1)))

    err = try
        solve(ops, GVM(basis = 2, scale = 1.0, optimizer = quiet_lbfgs(1)); state = 5)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("requested state 5", sprint(showerror, err))

    g = solve(ops, DynamicGVM(basis = 5, candidates = 5, scale = 1.0))
    @test g.E₀ < -0.45
    @test length(energies(g)) == length(g.basis.functions)
end

@testset "gradient solvers surface automatic-differentiation failures" begin
    primal_only_ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
    primal_only_ops += "Kinetic"
    primal_only_ops += "Coulomb"
    primal_only_ops += PrimalOnlyOperator()

    for alg in (
            GVM(basis = 2, scale = 1.0, optimizer = quiet_lbfgs(1)),
            DynamicGVM(basis = 2, candidates = 1, scale = 1.0, optimizer = quiet_lbfgs(1)),
        )
        err = try
            solve(primal_only_ops, alg)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("automatic differentiation failed", sprint(showerror, err))
    end
end

@testset "gradient solvers reject non-finite automatic derivatives" begin
    nonfinite_ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
    nonfinite_ops += "Kinetic"
    nonfinite_ops += "Coulomb"
    nonfinite_ops += NonfiniteGradientOperator()

    for alg in (
            GVM(basis = 2, scale = 1.0, optimizer = quiet_lbfgs(1)),
            DynamicGVM(basis = 2, candidates = 1, scale = 1.0, optimizer = quiet_lbfgs(1)),
        )
        err = try
            solve(nonfinite_ops, alg)
            nothing
        catch e
            e
        end
        @test err isa DomainError
        @test occursin("non-finite gradient", sprint(showerror, err))
    end
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

@testset "DynamicGVM early stop when every candidate fails" begin
    # An absurdly large `scale` pushes every quasi-random candidate's width
    # into a regime where its matrix elements against `terms` are numerically
    # degenerate, so every one of the 5 candidates throws inside the
    # per-candidate try/catch and the "all candidates failed" branch fires at
    # step 1, leaving the solver with no basis to return.
    local caught
    @test_logs (:warn, r"Sequential search stopped at step 1: all 5 candidates failed") match_mode = :any begin
        try
            solve(
                ops,
                DynamicGVM(
                    basis = 5, candidates = 5, scale = 1.0e105,
                    optimizer = quiet_lbfgs(50),
                );
                verbose = true,
            )
        catch e
            caught = e
        end
    end
    @test caught isa ErrorException
    @test occursin("Sequential selection produced no basis functions", caught.msg)
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
