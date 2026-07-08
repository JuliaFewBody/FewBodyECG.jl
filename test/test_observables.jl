using Test
using RecipesBase
using FewBodyECG

# test-only: lets recipes resolve attributes without a Plots backend (mirrors RecipesBase's own test suite)
RecipesBase.is_key_supported(::Symbol) = true

ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"
sol = solve(ops, SVM(basis = 15, candidates = 15, scale = 1.0))

@testset "Wavefunction" begin
    ψ = wavefunction(sol)
    @test ψ isa Wavefunction
    @test isfinite(ψ([0.5]))
    # matches the explicit linear combination
    c = sol.coefficients[:, 1]
    fns = sol.basis.functions
    ref = sum(
        c[i] * exp(-([0.5]' * fns[i].A * [0.5]) + fns[i].s' * [0.5])
            for i in eachindex(fns)
    )
    @test ψ([0.5]) ≈ ref rtol = 1.0e-12
    # Rank1 evaluation: (aᵀr)·exp(−rᵀAr)
    g1 = Rank1Gaussian([1.0;;], [1.0], [0.0])
    ψ1 = Wavefunction(BasisSet([g1]), [1.0])
    @test ψ1([0.7]) ≈ 0.7 * exp(-0.49) rtol = 1.0e-12
    # Rank2 evaluation: (aᵀr)(bᵀr)·exp(−rᵀAr)
    g2 = Rank2Gaussian([1.0;;], [1.0], [1.0], [0.0])
    ψ2 = Wavefunction(BasisSet([g2]), [1.0])
    @test ψ2([0.7]) ≈ 0.7 * 0.7 * exp(-0.49) rtol = 1.0e-12
end

@testset "Recipes" begin
    # convergence recipe
    plots = RecipesBase.apply_recipe(Dict{Symbol, Any}(), sol)
    @test !isempty(plots)
    # with reference energy
    plots2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(), sol, -0.5)
    @test length(plots2) ≥ 2
    # wavefunction recipe
    ψ = wavefunction(sol)
    wplots = RecipesBase.apply_recipe(Dict{Symbol, Any}(), ψ)
    @test !isempty(wplots)
end
