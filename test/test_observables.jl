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
        c[i] * exp(-([0.5]' * fns[i].A * [0.5]) + (@view parent(fns[i].s)[:, 3])' * [0.5])
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

@testset "convergence and radial_profile utilities" begin
    steps, history = convergence(sol)
    @test steps == 1:length(history)
    @test history == energies(sol)

    r, density = radial_profile(wavefunction(sol); rmax = 4, npoints = 200)
    @test first(r) ≥ 0
    @test all(≥(0), density)
    @test isapprox(
        sum((density[i] + density[i + 1]) * (r[i + 1] - r[i]) / 2 for i in 1:(length(r) - 1)),
        1; atol = 1.0e-8
    )

    # unnormalized profile is the bare r²|ψ|²
    r2, d2 = radial_profile(wavefunction(sol); rmax = 4, npoints = 200, normalize = false)
    @test all(≥(0), d2)

    # rank-1 and rank-2 wavefunction profiles are supported and half-line normalized
    g1 = Rank1Gaussian([1.0;;], [1.0], [0.0])
    ψ1 = Wavefunction(BasisSet([g1]), [1.0])
    r1, dens1 = radial_profile(ψ1; rmax = 6, npoints = 300)
    @test first(r1) ≥ 0 && all(≥(0), dens1)
    @test isapprox(
        sum((dens1[i] + dens1[i + 1]) * (r1[i + 1] - r1[i]) / 2 for i in 1:(length(r1) - 1)),
        1; atol = 1.0e-8
    )

    g2 = Rank2Gaussian([1.0;;], [1.0], [1.0], [0.0])
    ψ2 = Wavefunction(BasisSet([g2]), [1.0])
    r2b, dens2 = radial_profile(ψ2; rmax = 6, npoints = 300)
    @test all(≥(0), dens2)
    @test isapprox(
        sum((dens2[i] + dens2[i + 1]) * (r2b[i + 1] - r2b[i]) / 2 for i in 1:(length(r2b) - 1)),
        1; atol = 1.0e-8
    )

    @test_throws ArgumentError radial_profile(wavefunction(sol); coord = 5)
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
