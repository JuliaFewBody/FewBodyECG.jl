using Test
using FewBodyECG
using LinearAlgebra
using QuasiMonteCarlo

import FewBodyECG: _generate_A_matrix, _compute_overlap_element

# Helper: solve generalized eigenvalue problem via symmetric inverse square root of S
function _solve_gep(H, S)
    λs, Us = eigen(S)
    keep = λs .> 1.0e-10
    S⁻¹₂ = Us[:, keep] * Diagonal(1 ./ sqrt.(λs[keep])) * Us[:, keep]'
    H̃ = Symmetric(S⁻¹₂ * H * S⁻¹₂)
    return minimum(eigen(H̃).values)
end

# Common setup for hydrogen atom (2-body: infinite mass nucleus + electron)
masses = [1.0e15, 1.0]
Λmat = Λ(masses)
K_transformed = Λmat
J, U = _jacobi_transform(masses)
w_raw = [U' * [1, -1]]
coeffs = [-1.0]

ops = FewBodyECG.Operator[
    KineticOperator(K_transformed);
    (CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw))...
]

s_zero = [0.0]

@testset "Hydrogen s-wave (Rank0) convergence" begin
    n_basis = 20
    b1 = 1.5
    basis_fns = GaussianBase[]
    E_best = Inf

    for i in 1:n_basis
        bij = generate_bij(:quasirandom, i, length(w_raw), b1; qmc_sampler = SobolSample())
        A = _generate_A_matrix(bij, w_raw)
        push!(basis_fns, Rank0Gaussian(A, s_zero))

        basis = BasisSet(basis_fns)
        H = build_hamiltonian_matrix(basis, ops)
        S = build_overlap_matrix(basis)
        E = _solve_gep(H, S)
        E_best = min(E_best, E)
    end

    E_exact = -0.5
    @test abs(E_best - E_exact) < 0.01
end

@testset "Hydrogen p-wave (Rank1) convergence" begin
    # p-wave (2p state) needs wider Gaussians than s-wave, so use manual α values
    a_vec = [1.0]
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    basis_fns = [Rank1Gaussian([α;;], a_vec, s_zero) for α in alphas]

    basis = BasisSet(basis_fns)
    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)
    E_best = _solve_gep(H, S)

    E_exact = -0.125  # -1/8 Hartree (2p state)
    @test abs(E_best - E_exact) < 0.01
end

@testset "Rank2 Hamiltonian matrix is symmetric" begin
    # Rank2 with scalar polarizations gives r²·exp(-αr²) (mixed s+d wave).
    # Pure d-wave requires 3D polarization vectors (a·b=0), which is beyond scalar code.
    # Here we just test that the Hamiltonian and overlap matrices are well-formed.
    a_vec = [1.0]
    b_vec = [1.0]
    alphas = [0.1, 0.5, 1.0, 3.0]
    basis_fns = [Rank2Gaussian([α;;], a_vec, b_vec, s_zero) for α in alphas]

    basis = BasisSet(basis_fns)
    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)

    @test H ≈ H'
    @test S ≈ S'
    @test all(diag(S) .> 0)
end

@testset "Rank1 overlap is correct" begin
    A = [1.0;;]
    B = [2.0;;]
    a = [1.0]
    b = [1.0]

    bra = Rank1Gaussian(A, a, s_zero)
    ket = Rank1Gaussian(B, b, s_zero)
    val = _compute_overlap_element(bra, ket)

    R = inv(A + B)
    M0 = (π / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0
    @test isapprox(val, M1; atol = 1.0e-12)
end

@testset "Rank2 overlap is correct" begin
    A = [1.0;;]
    B = [2.0;;]
    a = [1.0]
    b_vec = [1.0]
    c = [1.0]
    d = [1.0]

    bra = Rank2Gaussian(A, a, b_vec, s_zero)
    ket = Rank2Gaussian(B, c, d, s_zero)
    val = _compute_overlap_element(bra, ket)

    R = inv(A + B)
    M0 = (π / det(A + B))^(3 / 2)
    M2 = 0.25 * (
        dot(a, R * b_vec) * dot(c, R * d) +
            dot(a, R * c) * dot(b_vec, R * d) +
            dot(a, R * d) * dot(b_vec, R * c)
    ) * M0
    @test isapprox(val, M2; atol = 1.0e-12)
end

@testset "Matrix element symmetry (bra/ket swap)" begin
    A = [1.5;;]
    B = [2.3;;]
    a = [1.0]
    b = [1.0]
    K_op = KineticOperator(K_transformed)
    C_op = CoulombOperator(-1.0, w_raw[1])

    # Rank1 symmetry
    bra1 = Rank1Gaussian(A, a, s_zero)
    ket1 = Rank1Gaussian(B, b, s_zero)
    import FewBodyECG: _compute_matrix_element
    @test isapprox(
        _compute_matrix_element(bra1, ket1, K_op),
        _compute_matrix_element(ket1, bra1, K_op);
        atol = 1.0e-12
    )
    @test isapprox(
        _compute_matrix_element(bra1, ket1, C_op),
        _compute_matrix_element(ket1, bra1, C_op);
        atol = 1.0e-12
    )

    # Rank2 symmetry
    bra2 = Rank2Gaussian(A, a, b, s_zero)
    ket2 = Rank2Gaussian(B, a, b, s_zero)
    @test isapprox(
        _compute_matrix_element(bra2, ket2, K_op),
        _compute_matrix_element(ket2, bra2, K_op);
        atol = 1.0e-12
    )
    @test isapprox(
        _compute_matrix_element(bra2, ket2, C_op),
        _compute_matrix_element(ket2, bra2, C_op);
        atol = 1.0e-12
    )
end
