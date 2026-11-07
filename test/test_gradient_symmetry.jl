module GradientSymmetryTests

using Test, LinearAlgebra, Random, ForwardDiff, FewBodyECG
using OptimKit: LBFGS
using FewBodyECG: _GradientLayout, _gradient_layout, _encode_basis, _decode_basis, _n_params,
    ProjectedGaussian, PermutationSymmetry

he_sym = PermutationSymmetry([1.0e15, 1.0, 1.0], [[1, 3, 2]])
A = [1.3 0.2; 0.2 0.7]
r0 = Rank0Gaussian(A, zeros(2, 3))
r1 = Rank1Gaussian(A, [0.6, 0.8], zeros(2, 3))

@testset "gradient parameter layouts" begin
    @testset "layout round trips" begin
        for (rank, sym, g) in (
                (0, nothing, r0), (1, nothing, r1),
                (0, he_sym, ProjectedGaussian(r0, he_sym)),
                (1, he_sym, ProjectedGaussian(r1, he_sym)),
            )
            L = _gradient_layout(rank, sym, 2)
            @test L isa _GradientLayout
            θ = _encode_basis(L, BasisSet([g, g]))
            @test length(θ) == 2 * _n_params(L)
            h = _decode_basis(L, θ, 2).functions[2]
            @test typeof(h) == typeof(g)
            o, p = sym === nothing ? (h, g) : (h.orbital, g.orbital)
            @test o.A ≈ p.A
            rank == 1 && @test o.a ≈ p.a
        end
        @test _n_params(_gradient_layout(0, nothing, 2)) == 9
        @test _n_params(_gradient_layout(1, nothing, 2)) == 5
    end

    @testset "Rank1 polarization is normalized on decode" begin
        L = _gradient_layout(1, nothing, 2)
        θ = _encode_basis(L, BasisSet([r1]))
        θ[(end - 1):end] .*= 3
        @test _decode_basis(L, θ, 1).functions[1].a ≈ [0 0 0.6; 0 0 0.8]
    end

    @testset "unrepresentable initial bases" begin
        init(fs) = (; basis = BasisSet(fs))
        @test_throws "Rank1 functions must be unshifted" _gradient_layout(
            nothing, nothing, 2; init = init([Rank1Gaussian(A, [0.6, 0.8], [0.0 0 0.1; 0 0 0])])
        )
        @test_throws "polarized along z" _gradient_layout(
            nothing, nothing, 2; init = init([Rank1Gaussian(A, [1.0 0 0; 0 0 1], zeros(2, 3))])
        )
        @test_throws "uniformly Rank0Gaussian or Rank1Gaussian" _gradient_layout(
            nothing, nothing, 2; init = init([r0, r1])
        )
        @test_throws "rank=1 does not match the rank-0 initial basis" _gradient_layout(
            1, nothing, 2; init = init([r0])
        )
        @test_throws "gradient methods support rank 0 or 1" _gradient_layout(2, nothing, 2)
    end
end

quiet(maxiter) = LBFGS(; maxiter, gradtol = 1.0e-6, verbosity = 0, ls_verbosity = 0)

He = Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([2.0, -1.0, -1.0])
antisym = PermutationSymmetry([1.0e15, 1.0, 1.0], [[1, 3, 2]]; signs = [-1])
Hh = Kinetic([1.0e15, 1.0]) + Coulomb([1.0, -1.0])
well = Kinetic([1.0, 1.0]) + -2.0 * Gaussian(1, 2; exponent = 1.0)

function gradient_pair(sol, layout; seed = 1)
    terms = getfield(sol, :operators)
    n = length(sol.basis)
    θ = _encode_basis(layout, sol.basis) .+ 0.05 .* randn(Xoshiro(seed), n * _n_params(layout))
    cfg = FewBodyECG._gradient_config(θ)
    bad = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    energy(x) = first(FewBodyECG._energy_gradient(x, layout, n, terms, cfg, 1, 1.0e-10, bad))
    _, g_ad = FewBodyECG._energy_gradient(θ, layout, n, terms, cfg, 1, 1.0e-10, bad)
    h = 1.0e-5
    g_fd = map(eachindex(θ)) do k
        e = zeros(length(θ)); e[k] = h
        (energy(θ + e) - energy(θ - e)) / 2h
    end
    return g_ad, g_fd
end

@testset "gradient methods with symmetry and rank" begin
    @testset "Hellmann–Feynman gradient matches finite differences" begin
        for (H, rank, sym, svm) in (
                (Hh, 1, nothing, SVM(6; rank = 1, candidates = 25, indep_tol = 1.0e-8)),
                (He, 0, antisym, SVM(6)),
                (well, 1, nothing, SVM(4; rank = 1, scale = 1.0)),
                (He, 1, antisym, SVM(6; rank = 1)),
            )
            sol = solve(H, svm; symmetry = sym)
            n_dim = FewBodyECG._gaussian_dimension(first(sol.basis.functions))
            g_ad, g_fd = gradient_pair(sol, _gradient_layout(rank, sym, n_dim))
            @test all(isfinite, g_ad)
            @test maximum(abs, g_ad - g_fd) ≤ 1.0e-5 * max(1, maximum(abs, g_fd))
        end
    end

    @testset "GVM on symmetric and polarized bases" begin
        p2 = solve(Hh, SVM(8; rank = 1, candidates = 25, indep_tol = 1.0e-8) → GVM(; optimizer = quiet(40)))
        @test all(g -> g isa Rank1Gaussian && iszero(g.s), p2.basis)
        @test -0.125 - 1.0e-8 ≤ p2.E₀ < p2.stages[1].history[end]

        t = solve(He, SVM(12) → GVM(; optimizer = quiet(40)); symmetry = antisym)
        @test all(g -> g isa ProjectedGaussian, t.basis)
        @test -2.175229378 - 1.0e-6 ≤ t.E₀ < t.stages[1].history[end]

        tp = solve(He, SVM(12; rank = 1) → GVM(; optimizer = quiet(40)); symmetry = antisym)
        @test all(g -> g isa ProjectedGaussian && g.orbital isa Rank1Gaussian, tp.basis)
        @test -2.133164191 - 1.0e-6 ≤ tp.E₀ < tp.stages[1].history[end]

        pw = solve(well, SVM(4; rank = 1, scale = 1.0) → GVM(; optimizer = quiet(20)))
        @test all(g -> g isa Rank1Gaussian, pw.basis)
        @test pw.E₀ < pw.stages[1].history[end]

        cold = solve(He, GVM(10; rank = 1, optimizer = quiet(30)); symmetry = antisym)
        @test all(g -> g isa ProjectedGaussian && g.orbital isa Rank1Gaussian, cold.basis)
        @test cold.E₀ ≥ -2.133164191 - 1.0e-6
        @test cold.draws ≥ 10

        s2 = solve(He, SVM(15) → GVM(; optimizer = quiet(30)); symmetry = antisym, state = 2)
        @test s2.E₀ == s2.E[2]
        @test s2.E₀ ≥ -2.068689067 - 1.0e-6
    end

    @testset "pipeline round trip keeps rank and symmetry" begin
        r = solve(He, SVM(10; rank = 1) → GVM(; optimizer = quiet(20)) → Refine(1); symmetry = antisym)
        @test all(g -> g isa ProjectedGaussian && g.orbital isa Rank1Gaussian, r.basis)
        @test r.E₀ ≤ r.stages[2].history[end] + 1.0e-10
    end

    @testset "DynamicGVM in a symmetry sector" begin
        d = solve(He, DynamicGVM(6; rank = 1, candidates = 8, optimizer = quiet(15)); symmetry = antisym)
        @test length(d.basis) == 6
        @test all(g -> g isa ProjectedGaussian && g.orbital isa Rank1Gaussian, d.basis)
        @test d.E₀ ≥ -2.133164191 - 1.0e-6
        @test all(diff(energy_history(d)) .≤ 1.0e-10)
        r = solve(He, Refine(1); init = d)
        @test r.draws > d.draws
        @test r.E₀ ≤ d.E₀ + 1.0e-10
    end

    @testset "empty symmetry sector" begin
        two = Kinetic([1.0, 1.0]) + -2.0 * Gaussian(1, 2; exponent = 1.0)
        odd = PermutationSymmetry([1.0, 1.0], [[2, 1]]; signs = [-1])
        @test_throws "no admissible projected candidates" solve(two, GVM(4; scale = 1.0); symmetry = odd)
    end
end

end # module
