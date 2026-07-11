using Test
using LinearAlgebra
using FewBodyECG

# Model spin-orbit doublet: two spin-½ particles in a p-wave-like manifold of
# shifted Gaussians bound by a central well.  A GaussianSpinOrbitOperator makes
# the Hamiltonian complex Hermitian and splits the degenerate upper doublet
# linearly in the coupling strength.  Exercises the complex generalized
# eigensolver end-to-end.
@testset "Spin-orbit doublet (complex Hermitian)" begin
    ops = Operators([1.0, 1.0])
    ops += "Kinetic"
    ops += ("Gaussian", 1, 2, -4.0, 0.4)
    terms = ops.terms
    w = only(op.w for op in terms if op isa GaussianOperator)

    shift(dir) = (v = zeros(1, 3); v[1, dir] = 0.8; v)
    basis = BasisSet(
        [
            SpinGaussian(Rank0Gaussian([0.4;;], shift(d)), SpinState([up, up])) for d in 1:3
        ]
    )

    H₀ = build_hamiltonian_matrix(basis, terms)
    Hₛₒ = build_hamiltonian_matrix(
        basis, FewBodyECG.Operator[GaussianSpinOrbitOperator(1.0, 0.4, w, 1, 2)]
    )
    S = build_overlap_matrix(basis)

    # spin-orbit-free block is real; spin-orbit block is genuinely complex Hermitian
    @test eltype(H₀) <: Real
    @test eltype(Hₛₒ) <: Complex
    @test Hₛₒ ≈ Hₛₒ'
    @test !isapprox(Hₛₒ, real.(Hₛₒ))          # actually complex, not a real matrix in disguise

    gap(λ) = begin
        E, C = solve_generalized_eigenproblem(H₀ .+ λ .* Hₛₒ, S)
        @test maximum(abs, imag.(E)) < 1.0e-10      # eigenvalues are real
        @test eltype(C) <: Complex                   # eigenvectors stay complex
        Es = sort(real.(E))
        Es[3] - Es[2]
    end

    # degenerate at λ = 0, splits, and the gap is linear in λ
    @test gap(0.0) < 1.0e-8
    g1 = gap(1.0)
    g2 = gap(2.0)
    @test g1 > 1.0e-3
    @test g2 ≈ 2 * g1 rtol = 1.0e-3               # linear splitting

    # nested-basis enrichment converges the ground state monotonically
    αs = exp10.(range(-0.8, 0.6, length = 8))
    sg(α, d) = SpinGaussian(Rank0Gaussian([α;;], shift(d)), SpinState([up, up]))
    ground = map(1:length(αs)) do k
        b = BasisSet([sg(α, d) for α in αs[1:k] for d in 1:3])
        Hk = build_hamiltonian_matrix(b, terms) .+
            build_hamiltonian_matrix(b, FewBodyECG.Operator[GaussianSpinOrbitOperator(1.0, 0.4, w, 1, 2)])
        minimum(real.(solve_generalized_eigenproblem(Hk, build_overlap_matrix(b))[1]))
    end
    @test all(diff(ground) .≤ 1.0e-9)             # monotone non-increasing (variational)
    @test ground[end] < ground[1]                  # actually improves
end
