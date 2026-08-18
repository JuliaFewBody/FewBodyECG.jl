using Test
using FewBodyECG

@testset "Solver method hierarchy and pipelines" begin
    @test SVM() isa StochasticMethod
    @test Refine() isa StochasticMethod
    @test GVM() isa GradientMethod
    @test DynamicGVM() isa GradientMethod
    @test SVM() isa SolverMethod
    @test SVM().basis == 50 && SVM().candidates == 25
    @test SVM(120).basis == 120                    # positional convenience
    @test SVM(120; candidates = 40).candidates == 40
    @test Refine(3).sweeps == 3
    @test GVM().basis === nothing
    @test GVM().scale === nothing
    @test GVM(30).basis == 30
    @test DynamicGVM().basis == 15

    p = SVM(120) → Refine(2) → GVM()
    @test p isa Pipeline
    @test p isa SolverMethod
    @test length(p.stages) == 3
    @test p.stages[1] isa SVM && p.stages[3] isa GVM
    @test (SVM() → (Refine() → GVM())).stages |> length == 3

    @test sprint(show, SVM(120)) == "SVM(120)"
    @test occursin("→", sprint(show, p))

    @test FewBodyECG._resolve_scale(2.0, [1.0, 1.0]) == 2.0
    @test FewBodyECG._resolve_scale(:auto, [1.0e15, 1.0]) ≈ 1.0
    @test_throws ArgumentError FewBodyECG._resolve_scale(:auto, nothing)
end
