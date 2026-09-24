using Test
using LinearAlgebra
using RecipesBase
using FewBodyECG

# test-only: lets recipes resolve attributes without a Plots backend (mirrors
# RecipesBase's own test suite); `KEY_SUPPORTED[] = false` simulates a backend
# that supports no attributes.
const KEY_SUPPORTED = Ref(true)
RecipesBase.is_key_supported(::Symbol) = KEY_SUPPORTED[]

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

    # Cartesian N×3 positions: (a·r)(b·r)·exp(−tr(rᵀAr) + tr(sᵀr)) with a·r = tr(aᵀr)
    a = [1.0 0.0 0.0]
    b = [0.0 1.0 0.0]
    ψxy = Wavefunction(BasisSet([Rank2Gaussian([1.0;;], a, b, [0.0])]), [1.0])
    @test ψxy([0.7]) == 0                                      # d_xy vanishes on the z axis
    @test ψxy([0.3 0.5 -0.2]) ≈ 0.3 * 0.5 * exp(-(0.09 + 0.25 + 0.04)) rtol = 1.0e-12

    A = [1.0 0.2; 0.2 1.5]
    p = [0.4 -0.1 0.3; 0.2 0.5 -0.6]
    s = [0.1 0.0 -0.2; 0.3 -0.1 0.2]
    r = [0.3 -0.4 0.5; -0.2 0.6 0.1]
    ψ1s = Wavefunction(BasisSet([Rank1Gaussian(A, p, s)]), [1.0])
    @test ψ1s(r) ≈ tr(p' * r) * exp(-tr(r' * A * r) + tr(s' * r)) rtol = 1.0e-12

    # a length-N vector places every coordinate on the z axis
    v = [0.3, -0.7]
    for g in (
            Rank0Gaussian(A, s), Rank1Gaussian(A, p, s),
            Rank2Gaussian(A, p, [0.2, -0.4], s),
        )
        ψg = Wavefunction(BasisSet([g]), [1.0])
        @test ψg(v) ≈ ψg([0 0 v[1]; 0 0 v[2]]) rtol = 1.0e-12
    end
    @test_throws "r must be a length-2 vector (z component) or a 2×3 matrix" ψ1s([0.1 0.2; 0.3 0.4])
end

@testset "convergence and radial_profile utilities" begin
    steps, history = convergence(sol)
    @test steps == 1:length(history)
    @test history == energy_history(sol)

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

    # polarized functions need a direction off their nodal plane
    ψxy = Wavefunction(BasisSet([Rank2Gaussian([1.0;;], [1.0 0.0 0.0], [0.0 1.0 0.0], [0.0])]), [1.0])
    _, dz = radial_profile(ψxy; rmax = 6, npoints = 300)
    @test all(iszero, dz)
    rd, dd = radial_profile(ψxy; direction = (1, 1, 0), rmax = 6, npoints = 300)
    @test isapprox(sum((dd[i] + dd[i + 1]) * (rd[i + 1] - rd[i]) / 2 for i in 1:(length(rd) - 1)), 1; atol = 1.0e-8)
    _, dd_raw = radial_profile(ψxy; direction = (1, 1, 0), rmax = 6, npoints = 300, normalize = false)
    @test dd_raw ≈ rd .^ 2 .* (rd .^ 2 ./ 2 .* exp.(-rd .^ 2)) .^ 2 rtol = 1.0e-12
    @test radial_profile(ψxy; direction = (2, 2, 0), normalize = false)[2] ≈
        radial_profile(ψxy; direction = (1, 1, 0), normalize = false)[2]
    @test_throws "direction must have 3 Cartesian components" radial_profile(ψxy; direction = (1, 0))
    @test_throws "direction must be nonzero" radial_profile(ψxy; direction = (0, 0, 0))
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
    wplots_dir = RecipesBase.apply_recipe(Dict{Symbol, Any}(:direction => (1, 0, 0)), ψ)
    @test !isempty(wplots_dir)
    # recipe keywords are consumed, not forwarded to a backend that lacks them
    KEY_SUPPORTED[] = false
    try
        attrs = Dict{Symbol, Any}(:coord => 1, :direction => (1, 0, 0))
        series = RecipesBase.apply_recipe(attrs, ψ)
        @test only(series).args[2] ≈ radial_profile(ψ; direction = (1, 0, 0))[2]
        @test !haskey(attrs, :coord) && !haskey(attrs, :direction)
    finally
        KEY_SUPPORTED[] = true
    end
end
