using Test, LinearAlgebra, Random
import ForwardDiff
using SpecialFunctions: erf
using FewBodyECG
using FewBodyECG: build_hamiltonian_matrix, build_overlap_matrix, solve_generalized_eigenproblem,
    Λ, jacobi_transform
import FewBodyECG: _compute_matrix_element, _erf_ratio_from_square, _shifted_radial_kernel,
    _polarized_element

d1(f, x) = ForwardDiff.derivative(f, x)
d2(f, x) = ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x)
# Named callables: every distinct NumericalPotential callable compiles separately.
inv_r(r) = -1 / r
gauss07(r) = exp(-0.7 * r^2)

const RNG_SEED = 20261104
random_spd(rng, n) = (X = randn(rng, n, n); X' * X + I)
function random_gaussian(rng, rank, n; shift)
    A, s = random_spd(rng, n), shift .* randn(rng, n, 3)
    rank == 0 && return Rank0Gaussian(A, s)
    rank == 1 && return Rank1Gaussian(A, randn(rng, n, 3), s)
    return Rank2Gaussian(A, randn(rng, n, 3), randn(rng, n, 3), s)
end
with_exponent(g::Rank0Gaussian, A) = Rank0Gaussian(A, g.s)
with_exponent(g::Rank1Gaussian, A) = Rank1Gaussian(A, g.a, g.s)
with_exponent(g::Rank2Gaussian, A) = Rank2Gaussian(A, g.a, g.b, g.s)
rank0_element(op) = (g, h) -> _compute_matrix_element(g, h, op)

@testset "Coulomb kernel survives repeated differentiation" begin
    ref_erf(x) = erf(sqrt(x)) / sqrt(x)
    for x in (1.0e-12, 1.0e-6, 0.3, 0.999999, 1.0, 1.000001, 2.5, 30.0)
        xb = big(x)
        @test _erf_ratio_from_square(x) ≈ ref_erf(xb) rtol = 1.0e-14
        @test d1(_erf_ratio_from_square, x) ≈ d1(ref_erf, xb) rtol = 1.0e-13
        @test d2(_erf_ratio_from_square, x) ≈ d2(ref_erf, xb) rtol = 1.0e-12
    end
    @test _erf_ratio_from_square(0.0) ≈ 2 / sqrt(π) rtol = 1.0e-15
    @test d1(_erf_ratio_from_square, 0.0) ≈ -2 / (3 * sqrt(π)) rtol = 1.0e-15
    @test d2(_erf_ratio_from_square, 0.0) ≈ 2 / (5 * sqrt(π)) rtol = 1.0e-15
    # Fourth derivatives are needed for Rank2×Rank2 elements near the switch.
    d4(f, x) = d2(y -> d2(f, y), x)
    @test d4(_erf_ratio_from_square, 0.999999) ≈
        d4(ref_erf, big(0.999999)) rtol = 1.0e-14
end

@testset "Shifted radial kernel: derivatives in μ² and no overflow" begin
    ref_kernel(r, β, μ2) = exp(-β * (r^2 + μ2)) * sinh(2β * r * sqrt(μ2)) / (2β * r * sqrt(μ2))
    # (0.5, 1.0, ·) puts x² = 4β²r²μ² on either side of the series switch at 1;
    # (30.0, 1.0, 784.0) has sinh(x) beyond Float64 range with a representable product.
    for (r, β, μ2) in (
            (0.7, 1.3, 1.0e-12), (0.7, 1.3, 1.0e-6), (0.7, 1.3, 0.2),
            (0.5, 1.0, 0.999999), (0.5, 1.0, 1.000001), (2.0, 0.8, 4.0), (30.0, 1.0, 784.0),
        )
        k(m) = _shifted_radial_kernel(r, β, m)
        kb(m) = ref_kernel(big(r), big(β), m)
        @test k(μ2) ≈ kb(big(μ2)) rtol = 1.0e-13
        @test d1(k, μ2) ≈ d1(kb, big(μ2)) rtol = 1.0e-12
        @test d2(k, μ2) ≈ d2(kb, big(μ2)) rtol = 1.0e-11
    end
    @test _shifted_radial_kernel(0.7, 1.3, 0.0) ≈ exp(-1.3 * 0.49) rtol = 1.0e-15
    @test d1(m -> _shifted_radial_kernel(0.7, 1.3, m), 0.0) ≈
        exp(-1.3 * 0.49) * (4 * 1.3^2 * 0.49 / 6 - 1.3) rtol = 1.0e-14
end

@testset "Shifted NumericalPotential matches the Gaussian closed form at large shifts" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.8 -0.1; -0.1 1.1]
    w = [1.0, -0.5]
    for scale in (0.3, 1.0, 3.0, 6.0)
        s = scale .* [0.3 -0.2 0.5; 0.1 0.4 -0.3]
        g, h = Rank0Gaussian(A, s), Rank0Gaussian(B, s)
        @test _compute_matrix_element(g, h, NumericalPotential(gauss07, w; rtol = 1.0e-12)) ≈
            _compute_matrix_element(g, h, GaussianOperator(1.0, 0.7, w)) rtol = 1.0e-9
    end
end

@testset "Rank0 second shift derivatives at and near zero shift" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.8 -0.1; -0.1 1.1]
    p = [0.3 -0.2 0.5; 0.1 0.4 -0.3]
    w = [1.0, -0.5]
    element(op) = t -> _compute_matrix_element(Rank0Gaussian(A, t .* p), Rank0Gaussian(B, zeros(2, 3)), op)

    f = element(CoulombOperator(-1.0, w))
    h = 1.0e-3
    @test d2(f, 0.0) ≈ (f(h) - 2 * f(0.0) + f(-h)) / h^2 rtol = 1.0e-5

    num = element(NumericalPotential(gauss07, w; rtol = 1.0e-12))
    gau = element(GaussianOperator(1.0, 0.7, w))
    for t in (0.0, 1.0e-3)
        @test d2(num, t) ≈ d2(gau, t) rtol = 1.0e-8
    end
end

@testset "Generic overlap reproduces the closed-form Rank1/Rank2 overlaps" begin
    rng = Xoshiro(RNG_SEED)
    for n in (1, 2, 3), rank in (1, 2), _ in 1:3
        bra = random_gaussian(rng, rank, n; shift = 0.3)
        ket = random_gaussian(rng, rank, n; shift = 0.3)
        @test _polarized_element(_compute_matrix_element, bra, ket) ≈
            _compute_matrix_element(bra, ket) rtol = 1.0e-10 atol = 1.0e-12
    end
end

@testset "Cross-rank overlaps match Gaussian moment formulas" begin
    rng = Xoshiro(RNG_SEED + 1)
    for _ in 1:3
        A, B = random_spd(rng, 2), random_spd(rng, 2)
        s, t = 0.3 .* randn(rng, 2, 3), 0.3 .* randn(rng, 2, 3)
        a, c, d = randn(rng, 2, 3), randn(rng, 2, 3), randn(rng, 2, 3)
        R = inv(A + B)
        μ = R * (s + t) / 2
        cov(x, y) = tr(x' * R * y) / 2
        I0 = _compute_matrix_element(Rank0Gaussian(A, s), Rank0Gaussian(B, t))
        @test _compute_matrix_element(Rank0Gaussian(A, s), Rank1Gaussian(B, c, t)) ≈ dot(c, μ) * I0 rtol = 1.0e-12
        @test _compute_matrix_element(Rank1Gaussian(A, a, s), Rank0Gaussian(B, t)) ≈ dot(a, μ) * I0 rtol = 1.0e-12
        @test _compute_matrix_element(Rank0Gaussian(A, s), Rank2Gaussian(B, c, d, t)) ≈
            (cov(c, d) + dot(c, μ) * dot(d, μ)) * I0 rtol = 1.0e-12
        @test _compute_matrix_element(Rank1Gaussian(A, a, s), Rank2Gaussian(B, c, d, t)) ≈
            (
            dot(a, μ) * dot(c, μ) * dot(d, μ) + dot(a, μ) * cov(c, d) +
                dot(c, μ) * cov(a, d) + dot(d, μ) * cov(a, c)
        ) * I0 rtol = 1.0e-12
    end
end

@testset "Complex-valued Rank0 elements keep both parts" begin
    rng = Xoshiro(RNG_SEED + 8)
    bra, ket = random_gaussian(rng, 1, 2; shift = 0.3), random_gaussian(rng, 2, 2; shift = 0.3)
    complex_overlap(g, h) = (2.0 - 3.0im) * _compute_matrix_element(g, h)
    @test _polarized_element(complex_overlap, bra, ket) ≈
        (2.0 - 3.0im) * _compute_matrix_element(bra, ket) rtol = 1.0e-12
end

@testset "Integer-typed Gaussians in cross-rank overlaps" begin
    gi, hi = Rank2Gaussian([2;;], [1], [1], [0]), Rank1Gaussian([1;;], [1], [1])
    gf, hf = Rank2Gaussian([2.0;;], [1.0], [1.0], [0.0]), Rank1Gaussian([1.0;;], [1.0], [1.0])
    @test _compute_matrix_element(gi, hi) ≈ _compute_matrix_element(gf, hf) rtol = 1.0e-12
    @test _compute_matrix_element(hi, gi) ≈ _compute_matrix_element(hf, gf) rtol = 1.0e-12
end

@testset "Operator derivatives compose with polarized shift derivatives" begin
    bra = Rank2Gaussian([0.7;;], [1.0 0 0], [0.0 1 0], [0.0])
    ket = Rank1Gaussian([0.8;;], [1.0], [0.2])
    r = [0.3, -0.4, 0.5]
    contact(c) = _compute_matrix_element(bra, ket, ContactDensityOperator([1.0], r; coefficient = c))
    expected = prod(r) * exp(-1.5 * sum(abs2, r) + 0.2r[3])
    @test ForwardDiff.derivative(contact, 0.0) ≈ expected rtol = 1.0e-12
end

@testset "Generic operators reproduce the closed-form Rank1/Rank2 kinetic and Coulomb elements" begin
    rng = Xoshiro(RNG_SEED + 2)
    for n in (1, 2, 3), rank in (1, 2), _ in 1:3
        bra = random_gaussian(rng, rank, n; shift = 0.0)
        ket = random_gaussian(rng, rank, n; shift = 0.0)
        Z = randn(rng, n, n)
        for op in (KineticOperator(Z' * Z), CoulombOperator(-1.3, randn(rng, n)))
            @test _polarized_element(rank0_element(op), bra, ket) ≈
                _compute_matrix_element(bra, ket, op) rtol = 1.0e-10 atol = 1.0e-12
        end
    end
end

const RANK_PAIRS = [(rb, rk) for rb in 0:2 for rk in 0:2 if (rb, rk) != (0, 0)]

@testset "Gaussian-type potentials equal overlaps with a widened exponent" begin
    rng = Xoshiro(RNG_SEED + 3)
    for (rb, rk) in RANK_PAIRS
        bra, ket = random_gaussian(rng, rb, 2; shift = 0.3), random_gaussian(rng, rk, 2; shift = 0.3)
        w, W = randn(rng, 2), 0.2 .* random_spd(rng, 2)
        @test _compute_matrix_element(bra, ket, GaussianOperator(-2.0, 0.7, w)) ≈
            -2.0 * _compute_matrix_element(with_exponent(bra, bra.A + 0.7 * (w * w')), ket) rtol = 1.0e-12
        @test _compute_matrix_element(bra, ket, ManyBodyGaussianOperator(1.5, W)) ≈
            1.5 * _compute_matrix_element(with_exponent(bra, bra.A + W), ket) rtol = 1.0e-12
    end
end

# Subset with one pair per nesting depth that involves a polarized ket.
const DEPTH_PAIRS = ((0, 1), (1, 1), (1, 2), (2, 2))

@testset "Oscillator equals minus the exponent derivative of the widened overlap" begin
    rng = Xoshiro(RNG_SEED + 4)
    h = 1.0e-3
    for (rb, rk) in DEPTH_PAIRS
        bra, ket = random_gaussian(rng, rb, 2; shift = 0.3), random_gaussian(rng, rk, 2; shift = 0.3)
        w = randn(rng, 2)
        widened(γ) = _compute_matrix_element(with_exponent(bra, bra.A + γ * (w * w')), ket)
        # Fourth-order central difference: truncation error O(h⁴).
        slope = (-widened(2h) + 8 * widened(h) - 8 * widened(-h) + widened(-2h)) / (12h)
        @test _compute_matrix_element(bra, ket, OscillatorOperator(0.4, w)) ≈ -0.4 * slope rtol = 1.0e-9
    end
end

@testset "Coulomb agrees with a numerical 1/r potential" begin
    rng = Xoshiro(RNG_SEED + 5)
    for (rb, rk) in DEPTH_PAIRS, scale in (0.0, 1.0e-3, 0.3, 3.0)
        bra, ket = random_gaussian(rng, rb, 2; shift = scale), random_gaussian(rng, rk, 2; shift = scale)
        w = randn(rng, 2)
        @test _compute_matrix_element(bra, ket, CoulombOperator(-1.0, w)) ≈
            _compute_matrix_element(bra, ket, NumericalPotential(inv_r, w; rtol = 1.0e-12)) rtol = 1.0e-7
    end
end

@testset "Hermiticity of the operators without an exact identity below" begin
    rng = Xoshiro(RNG_SEED + 6)
    for (rb, rk) in ((0, 2), (1, 2))
        g, h = random_gaussian(rng, rb, 2; shift = 0.3), random_gaussian(rng, rk, 2; shift = 0.3)
        w, Z = randn(rng, 2), randn(rng, 2, 2)
        for (op, tol) in (
                (KineticOperator(Z' * Z), 1.0e-10), (CoulombOperator(-1.0, w), 1.0e-10),
                (NumericalPotential(inv_r, w; rtol = 1.0e-12), 1.0e-8),
            )
            @test _compute_matrix_element(g, h, op) ≈ _compute_matrix_element(h, g, op) rtol = tol
        end
    end
end

@testset "Spin operators on plain or polarized Gaussians fail clearly" begin
    g0, g1 = Rank0Gaussian([1.0;;], [0.0]), Rank1Gaussian([1.0;;], [1.0], [0.0])
    tensor = GaussianTensorOperator(1.0, 0.5, [1.0], 1, 2)
    spin_orbit = GaussianSpinOrbitOperator(1.0, 0.5, [1.0], 1, 2)
    for op in (tensor, spin_orbit), (bra, ket) in ((g0, g0), (g1, g1), (g0, g1))
        @test_throws "couples to particle spins; build the basis from SpinGaussian functions" _compute_matrix_element(bra, ket, op)
    end
end

@testset "Integer-typed Gaussians with Coulomb" begin
    gi, hi = Rank2Gaussian([2;;], [1], [1], [0]), Rank1Gaussian([1;;], [1], [1])
    gf, hf = Rank2Gaussian([2.0;;], [1.0], [1.0], [0.0]), Rank1Gaussian([1.0;;], [1.0], [1.0])
    op = CoulombOperator(-1.0, [1.0])
    @test _compute_matrix_element(gi, hi, op) ≈ _compute_matrix_element(gf, hf, op) rtol = 1.0e-12
end

@testset "Harmonic-oscillator eigenfunctions: ⟨g|H|ψ⟩ = E⟨g|ψ⟩ for shifted g of every rank" begin
    masses = [1.0, 2.0, 3.0]
    _, U = jacobi_transform(masses)
    K = Matrix(Λ(masses))
    pair(i, j) = U' * [k == i ? 1.0 : k == j ? -1.0 : 0.0 for k in 1:3]
    couplings = ((1, 2, 0.3), (1, 3, 0.5), (2, 3, 0.7))
    ops = FewBodyECG.Operator[KineticOperator(masses); [OscillatorOperator(c, pair(i, j)) for (i, j, c) in couplings]]
    Q = sum(c * pair(i, j) * pair(i, j)' for (i, j, c) in couplings)
    # Ground state exp(−Σ_c x_cᵀA₀x_c) with 4A₀KA₀ = Q, energy 3 tr(M^{1/2});
    # excitations (uₖ·x_c)·ground with ωₖ = 2λₖ, where M^{1/2} vₖ = λₖ vₖ and uₖ = K^{-1/2} vₖ.
    Kh = sqrt(Symmetric(K))
    Mh = sqrt(Symmetric(Kh * Q * Kh))
    A0 = Kh \ Mh / Kh / 2
    A0 = (A0 + A0') / 2
    E0 = 3 * tr(Mh)
    λ, V = eigen(Symmetric(Matrix(Mh)))
    u, ω = [Kh \ V[:, k] for k in 1:2], 2 .* λ
    ψ0 = Rank0Gaussian(A0, zeros(2, 3))
    ψ1 = Rank1Gaussian(A0, u[1] * [1.0 0.0 0.0], zeros(2, 3))
    ψ2 = Rank2Gaussian(A0, u[1] * [1.0 0.0 0.0], u[2] * [0.0 1.0 0.0], zeros(2, 3))
    Hel(g, h) = sum(_compute_matrix_element(g, h, op) for op in ops)
    Sel(g, h) = _compute_matrix_element(g, h)

    rng = Xoshiro(RNG_SEED + 7)
    for (ψ, E) in ((ψ0, E0), (ψ1, E0 + ω[1]), (ψ2, E0 + ω[1] + ω[2]))
        @test Hel(ψ, ψ) / Sel(ψ, ψ) ≈ E rtol = 1.0e-12
        for rank in 0:2, _ in 1:3
            g = random_gaussian(rng, rank, 2; shift = 0.4)
            scale = sqrt(Sel(g, g) * Sel(ψ, ψ))
            @test Hel(g, ψ) ≈ E * Sel(g, ψ) atol = 1.0e-10 * scale
            @test Hel(ψ, g) ≈ E * Sel(ψ, g) atol = 1.0e-10 * scale
        end
    end

    # Hylleraas–Undheim: a basis containing ψ0 and the lowest excitation ψ1
    # reproduces the two lowest exact levels, whatever else it contains.
    fns = GaussianBase[ψ0, ψ1]
    for rank in 0:2, _ in 1:2
        push!(fns, random_gaussian(rng, rank, 2; shift = 0.4))
    end
    for order in (fns, reverse(fns))
        basis = BasisSet(order)
        E, _ = solve_generalized_eigenproblem(build_hamiltonian_matrix(basis, ops), build_overlap_matrix(basis))
        @test E[1:2] ≈ [E0, E0 + ω[1]] rtol = 1.0e-8
    end
end

@testset "One-coordinate oscillator spectra from polarized bases" begin
    masses = [1.0e15, 1.0]
    _, U = jacobi_transform(masses)
    w = U' * [1.0, -1.0]
    ops = FewBodyECG.Operator[KineticOperator(masses), OscillatorOperator(0.5, w)]
    αs = exp10.(range(-1.0, 1.0, length = 16))
    W = w * w'
    p_basis = BasisSet([Rank1Gaussian(α .* W, w * [0.0 0.0 1.0], zeros(1, 3)) for α in αs])
    d_basis = BasisSet([Rank2Gaussian(α .* W, w * [1.0 0.0 0.0], w * [0.0 1.0 0.0], zeros(1, 3)) for α in αs])
    for (basis, levels) in ((p_basis, (2.5, 4.5)), (d_basis, (3.5, 5.5)))
        S, H = build_overlap_matrix(basis), build_hamiltonian_matrix(basis, ops)
        # Normalizing leaves the eigenvalues unchanged and keeps cond(S) below
        # the 1e12 threshold at which solve_generalized_eigenproblem adds
        # max(diag S)·1e-10 to S; the unnormalized norms here span several
        # orders of magnitude.
        D = Diagonal(1 ./ sqrt.(diag(S)))
        E, _ = solve_generalized_eigenproblem(D * H * D, D * S * D)
        @test E[1] ≈ levels[1] atol = 1.0e-6
        @test E[2] ≈ levels[2] atol = 1.0e-4
    end
end
