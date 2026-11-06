module ProjectedGradientTests

using Test, LinearAlgebra, FewBodyECG
import ForwardDiff
using OptimKit: LBFGS
using FewBodyECG: build_hamiltonian_matrix, build_overlap_matrix, build_operator_matrix

optimizer(n) = LBFGS(; maxiter = n, gradtol = 1.0e-7, verbosity = 0, ls_verbosity = 0)

function finite_energy(θ, basis, layout, terms)
    decoded = FewBodyECG._decode_basis(θ, length(basis), 2, layout)
    S = build_overlap_matrix(decoded)
    D = Diagonal(1 ./ sqrt.(diag(S)))
    return first(eigen(Symmetric(D * build_hamiltonian_matrix(decoded, terms) * D), Symmetric(D * S * D)).values)
end

@testset "Projected gradient encoding and derivatives" begin
    masses = ones(3)
    terms = first(FewBodyECG._to_jacobi(Kinetic(masses) + Coulomb([2.0, -1.0, -1.0])))
    for rank in (0, 1), sign in (-1, 1)
        symmetry = PermutationSymmetry(masses, [[1, 3, 2]]; signs = [sign])
        orbital(A, a) = rank == 0 ? Rank0Gaussian(A, zeros(2)) : Rank1Gaussian(A, a, zeros(2))
        basis = BasisSet(
            [
                ProjectedGaussian(orbital([0.8 0.2; 0.2 1.3], [0.4, -0.7]), symmetry),
                ProjectedGaussian(orbital([0.5 -0.15; -0.15 0.9], [0.8, 0.3]), symmetry),
            ]
        )
        layout = FewBodyECG._gradient_layout(basis)
        θ = FewBodyECG._encode_basis(basis, layout)
        decoded = FewBodyECG._decode_basis(θ, 2, 2, layout)
        @test all(g -> g.symmetry == symmetry && iszero(g.orbital.s), decoded)
        @test decoded[1].orbital.A ≈ basis[1].orbital.A
        rank == 1 && @test norm(decoded[1].orbital.a) ≈ 1
        cfg = ForwardDiff.GradientConfig(FewBodyECG._energy_gradient, θ, ForwardDiff.Chunk(min(12, length(θ))))
        bad = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        energy, gradient = FewBodyECG._energy_gradient(θ, 2, 2, terms, cfg, 1, bad, layout)
        @test energy ≈ finite_energy(θ, basis, layout, terms)
        h = 1.0e-5
        differences = map(eachindex(θ)) do i
            step = zeros(length(θ))
            step[i] = h
            (finite_energy(θ + step, basis, layout, terms) - finite_energy(θ - step, basis, layout, terms)) / (2h)
        end
        @test gradient ≈ differences rtol = 2.0e-5 atol = 2.0e-7
        @test norm(gradient) > 1.0e-3
    end
end

@testset "Gradient bases reject null directions" begin
    odd = PermutationSymmetry(ones(2), [[2, 1]]; signs = [-1])
    null = ProjectedGaussian(Rank0Gaussian([0.5;;], zeros(1)), odd)
    @test_throws ArgumentError FewBodyECG._gradient_layout(BasisSet([null]))
    g = Rank1Gaussian([0.5;;], [1.0], zeros(1))
    basis = BasisSet([g, g])
    layout = FewBodyECG._gradient_layout(basis)
    θ = FewBodyECG._encode_basis(basis, layout)
    terms = first(FewBodyECG._to_jacobi(Kinetic(ones(2)) + 0.5Oscillator(1, 2)))
    cfg = ForwardDiff.GradientConfig(FewBodyECG._energy_gradient, θ, ForwardDiff.Chunk(min(12, length(θ))))
    bad = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    @test first(FewBodyECG._energy_gradient(θ, 2, 1, terms, cfg, 1, bad, layout)) == Inf
    θ[2:4] .= 0
    @test first(FewBodyECG._energy_gradient(θ, 2, 1, terms, cfg, 1, bad, layout)) == Inf
    @test_throws ArgumentError FewBodyECG._gradient_layout(BasisSet(GaussianBase[Rank0Gaussian([0.5;;], zeros(1)), g]))
    @test_throws ArgumentError FewBodyECG._gradient_layout(BasisSet(GaussianBase[ProjectedGaussian(g, odd), g]))
    @test_throws ArgumentError FewBodyECG._gradient_layout(BasisSet([Rank2Gaussian([0.5;;], [1.0], [1.0], zeros(1))]))
end

@testset "Extreme trial points have energy Inf" begin
    terms = first(FewBodyECG._to_jacobi(Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([2.0, -1.0, -1.0])))
    layout = FewBodyECG._GradientLayout(0, nothing)
    θ₀ = [0.0, 0.1, 0.2, zeros(6)...]
    cfg = ForwardDiff.GradientConfig(FewBodyECG._energy_gradient, [θ₀; θ₀], ForwardDiff.Chunk(12))
    bad = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    # Widths that overflow, underflow to a singular matrix, or are NaN.
    for log_width in (800.0, -800.0, NaN)
        θ = [[log_width; θ₀[2:end]]; θ₀ .+ 0.3]
        @test first(FewBodyECG._energy_gradient(θ, 2, 2, terms, cfg, 1, bad, layout)) == Inf
    end
    @test isfinite(first(FewBodyECG._energy_gradient([θ₀; θ₀ .+ 0.3], 2, 2, terms, cfg, 1, bad, layout)))
    H = Kinetic([1.0e15, 1.0]) + Coulomb([1.0, -1.0])
    @test_throws "requested state 3" solve(H, GVM(2; scale = 1.0, optimizer = optimizer(0)); state = 3)
end

@testset "GVM optimizes S/P sectors" begin
    masses = [1.0e15, 1.0]
    H = Kinetic(masses) + Coulomb([1.0, -1.0])
    p0 = solve(H, SVM(5; rank = 1, candidates = 12))
    p = solve(H, GVM(; optimizer = optimizer(100)); init = p0)
    @test -0.125000001 < energy(p) < -0.12499
    @test energy(p) < energy(p0) - 1.0e-6
    @test all(g -> g isa Rank1Gaussian && iszero(g.s) && norm(g.a) ≈ 1, p.basis)
    @test p.draws == p0.draws
    @test p.coefficients' * build_overlap_matrix(p.basis) * p.coefficients ≈ I atol = 1.0e-7
    @test_throws ArgumentError solve(H, GVM(; rank = 0); init = p0)
    @test_throws ArgumentError solve(H, GVM(2; rank = 2))
    cold = solve(H, GVM(5; rank = 1, optimizer = optimizer(100)))
    @test -0.125000001 < energy(cold) < -0.1249
    @test any(note -> occursin("pilot", note), cold.convergence.notes)

    masses = [7294.2618241, 1.0, 1.0]
    H = Kinetic(masses) + Coulomb([2.0, -1.0, -1.0])
    for rank in (0, 1), sign in (-1, 1)
        symmetry = PermutationSymmetry(masses, [[1, 3, 2]]; signs = [sign])
        seed = solve(H, SVM(6; rank, candidates = 15); symmetry)
        sol = solve(H, GVM(; optimizer = optimizer(40)); init = seed)
        @test energy(sol) < energy(seed) - 1.0e-5
        lower, upper = rank == 0 ? (sign == 1 ? (-2.904, -2.89) : (-2.176, -2.15)) :
            (sign == 1 ? (-2.124, -2.1) : (-2.134, -2.11))
        @test lower < energy(sol) < upper
        @test all(g -> g.symmetry == symmetry && iszero(g.orbital.s), sol.basis)
        rank == 1 && @test all(g -> iszero(g.orbital.a[:, 1:2]), sol.basis)
        c = sol.coefficients[:, 1]
        P = build_operator_matrix(sol.basis, PermutationOperator(masses, [1, 3, 2]))
        @test dot(c, P * c) ≈ sign atol = 1.0e-8
        @test dot(c, build_overlap_matrix(sol.basis) * c) ≈ 1 atol = 1.0e-9
    end
end

@testset "Projected gradient cold starts, growth and pipelines" begin
    masses = ones(2)
    H = Kinetic(masses) + Coulomb([1.0, -1.0])
    odd = PermutationSymmetry(masses, [[2, 1]]; signs = [-1])
    cold = solve(H, GVM(3; rank = 1, optimizer = optimizer(20)); symmetry = odd)
    @test -0.062500001 < energy(cold) < -0.06
    grown = solve(H, DynamicGVM(5; candidates = 3, optimizer = optimizer(20)); init = cold)
    @test length(grown.basis) == 5
    @test grown.draws == cold.draws + 6
    @test energy(grown) ≤ energy(cold) + 1.0e-9
    @test all(g -> g.symmetry == odd && g.orbital isa Rank1Gaussian && iszero(g.orbital.s), grown.basis)
    pipe = solve(H, SVM(4; rank = 1, candidates = 10) → GVM(; optimizer = optimizer(20)); symmetry = odd)
    @test length(pipe.stages) == 2
    @test energy(pipe) ≤ last(pipe.stages[1].history) + 1.0e-9
    @test_throws ArgumentError solve(H, GVM(2; scale = 1.0, optimizer = optimizer(0)); symmetry = odd)
    @test_throws ArgumentError solve(H, DynamicGVM(2; rank = 2))
end

@testset "Rank1 gradient solvers compose with shift differentiation" begin
    H = Kinetic(ones(2)) + 0.5Oscillator(1, 2)
    odd = PermutationSymmetry(ones(2), [[2, 1]]; signs = [-1])
    sol = solve(H, DynamicGVM(1; rank = 1, candidates = 3, scale = 1.0, optimizer = optimizer(30)); symmetry = odd)
    @test energy(sol) ≈ 5 / sqrt(2) atol = 1.0e-7
    @test iszero(only(sol.basis).orbital.s)
    numerical = Kinetic(ones(2)) + Custom(r -> 0.5r^2, 1, 2)
    sol = solve(numerical, GVM(1; rank = 1, scale = 1.0, optimizer = optimizer(30)); symmetry = odd)
    @test energy(sol) ≈ 5 / sqrt(2) atol = 1.0e-7

    g = Rank1Gaussian([0.7;;], [0.8 -0.3 0.2], [0.2 0.1 -0.4])
    basis = BasisSet([g])
    layout = FewBodyECG._gradient_layout(basis)
    θ = FewBodyECG._encode_basis(basis, layout)
    decoded = only(FewBodyECG._decode_basis(θ, 1, 1, layout))
    @test decoded.s ≈ g.s
    @test decoded.a ≈ g.a / norm(g.a)
    terms = first(FewBodyECG._to_jacobi(H))
    cfg = ForwardDiff.GradientConfig(FewBodyECG._energy_gradient, θ, ForwardDiff.Chunk(min(12, length(θ))))
    bad = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    value, gradient = FewBodyECG._energy_gradient(θ, 1, 1, terms, cfg, 1, bad, layout)
    function finite(θ)
        decoded = FewBodyECG._decode_basis(θ, 1, 1, layout)
        return build_hamiltonian_matrix(decoded, terms)[1, 1] / build_overlap_matrix(decoded)[1, 1]
    end
    h = 1.0e-5
    differences = map(eachindex(θ)) do i
        step = zeros(length(θ))
        step[i] = h
        (finite(θ + step) - finite(θ - step)) / (2h)
    end
    @test value ≈ finite(θ)
    @test gradient ≈ differences rtol = 1.0e-6 atol = 1.0e-8
end

@testset "Cold GVM selects an admissible starting basis" begin
    H = Kinetic([1.0e15, 1.0]) + Coulomb([1.0, -1.0])
    @test_throws ArgumentError solve(H, GVM(0; scale = 1.0, optimizer = optimizer(0)))
    for n in (16, 24), rank in (0, 1)
        sol = solve(H, GVM(n; rank, scale = 1.0, optimizer = optimizer(0)))
        @test length(sol.basis) == n
        @test sol.draws ≥ n
        @test isfinite(energy(sol))
        @test -(rank == 0 ? 0.50001 : 0.12501) < energy(sol) < 0
    end
    widened = solve(H, GVM(32; scale = 1.0, optimizer = optimizer(0)))
    @test length(widened.basis) == 32
    @test any(note -> occursin("widths = :wide", note), widened.convergence.notes)
end

end
