using Test
using FewBodyECG
using FewBodyECG: StageResult, SATURATION_CAVEAT

function _dummy_solution(; converged = true)
    g = Rank0Gaussian([1.0;;], [0.0])
    rep = ConvergenceReport(
        converged, :saturation, 3.2e-5, 1.0e-4, 20, nothing, 1.0e3,
        [SATURATION_CAVEAT]
    )
    st = StageResult(SVM(2), [-0.3, -0.42], rep)
    return Solution(
        [-0.42, 1.7], BasisSet([g, g]), [1.0 0.0; 0.0 1.0],
        FewBodyECG.Operator[], 1, [st, st], rep
    )
end

@testset "Solution and ConvergenceReport" begin
    sol = _dummy_solution()
    @test sol.E₀ ≈ -0.42
    @test sol.E == [-0.42, 1.7]
    @test converged(sol)
    @test !converged(_dummy_solution(converged = false))
    @test energies(sol) == [-0.3, -0.42, -0.3, -0.42]
    @test energies(sol, 2) == [-0.3, -0.42]
    @test :E₀ in propertynames(sol)

    out = sprint(show, MIME"text/plain"(), sol)
    @test occursin("E₀", out)
    @test occursin("-0.42", out) || occursin("−0.42", out)
    @test occursin("variational upper bound", out)
    @test occursin("saturation", out)
    @test occursin("SVM(2)", out)

    rout = sprint(show, MIME"text/plain"(), sol.convergence)
    @test occursin("saturated", rout) && occursin("1.0e-4", rout)
end
