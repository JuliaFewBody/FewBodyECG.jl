using Test
using FewBodyECG
using LinearAlgebra

import FewBodyECG: _compute_matrix_element

@testset "compute_matrix_element for Rank0Gaussian and KineticOperator" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.9 0.1; 0.1 1.2]
    K = rand(2, 2)

    bra = Rank0Gaussian(A)
    ket = Rank0Gaussian(B)
    op = KineticOperator(K)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)
    expected = 6 * tr(B * K * A * R) * M0

    @test isapprox(result, expected; atol = 1.0e-10)
end
@testset "compute_matrix_element for Rank0Gaussian and CoulombOperator" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.9 0.1; 0.1 1.2]
    w = [1.0, -1.0]
    coefficient = 1.5

    bra = Rank0Gaussian(A)
    ket = Rank0Gaussian(B)
    op = CoulombOperator(coefficient, w)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    n = size(R, 1)
    β = 1 / (dot(w, R * w))
    M0 = (π^n / det(A + B))^(3 / 2)
    expected = coefficient * 2 * sqrt(β / π) * M0

    @test isapprox(result, expected; atol = 1.0e-10)
end

@testset "compute_matrix_element for Rank1Gaussian and CoulombOperator" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.9 0.1; 0.1 1.2]
    a = [0.5, -0.4]
    b = [-0.2, 0.7]
    w = [1.0, -1.0]
    coefficient = 1.0

    bra = Rank1Gaussian(A, a)
    ket = Rank1Gaussian(B, b)
    op = CoulombOperator(coefficient, w)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    n = size(R, 1)
    β = 1 / (dot(w, R * w))
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0
    Rw = R * w
    # Eq. 28: 2√(β/π) M₁ - √(β/π) β/3 bᵀRwwᵀRa M₀
    expected = coefficient * (2 * sqrt(β / π) * M1 - sqrt(β / π) * β / 3 * dot(a, Rw) * dot(Rw, b) * M0)

    @test isapprox(result, expected; atol = 1.0e-10)
end

@testset "compute_matrix_element for Rank1Gaussian and KineticOperator" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.9 0.1; 0.1 1.2]
    a = [0.4, -0.6]
    b = [-0.3, 0.8]
    K = [2.0 0.0; 0.0 2.0]

    bra = Rank1Gaussian(A, a)
    ket = Rank1Gaussian(B, b)
    op = KineticOperator(K)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0

    # Eq. 23 with code vars: A=bra.A, B=ket.A, a=bra.a, b=ket.a
    T1 = 6 * tr(B * K * A * R) * M1
    T2 = dot(a, K * b) * M0
    T3 = (dot(b, R * A * K * B * R * a) + dot(a, R * A * K * B * R * b)) * M0
    T4 = -dot(a, R * A * K * b) * M0
    T5 = -dot(b, R * B * K * a) * M0

    expected = T1 + T2 + T3 + T4 + T5

    @test isapprox(result, expected; atol = 1.0e-10)
end

@testset "compute_matrix_element for Rank2Gaussian and KineticOperator" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.9 0.1; 0.1 1.2]
    a = [0.5, -0.4]
    b = [-0.2, 0.7]
    c = [0.3, 0.6]
    d = [-0.1, -0.8]
    K = [2.0 0.1; 0.1 2.0]

    bra = Rank2Gaussian(A, a, b)
    ket = Rank2Gaussian(B, c, d)
    op = KineticOperator(K)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)

    M2 = 0.25 * (
        dot(a, R * b) * dot(c, R * d) +
            dot(a, R * c) * dot(b, R * d) +
            dot(a, R * d) * dot(b, R * c)
    ) * M0

    T1 = 6 * tr(B * K * A * R) * M2

    T2 = 0.5 * (
        dot(a, K * c) * dot(b, R * d) +
            dot(a, K * d) * dot(b, R * c) +
            dot(b, K * c) * dot(a, R * d) +
            dot(b, K * d) * dot(a, R * c)
    ) * M0

    RAKBR = R * A * K * B * R
    T3 = 0.5 * (
        dot(a, RAKBR * b) * dot(c, R * d) +
            dot(a, RAKBR * c) * dot(b, R * d) +
            dot(a, RAKBR * d) * dot(b, R * c) +
            dot(b, RAKBR * a) * dot(c, R * d) +
            dot(b, RAKBR * c) * dot(a, R * d) +
            dot(b, RAKBR * d) * dot(a, R * c) +
            dot(c, RAKBR * a) * dot(b, R * d) +
            dot(c, RAKBR * b) * dot(a, R * d) +
            dot(c, RAKBR * d) * dot(a, R * b) +
            dot(d, RAKBR * a) * dot(b, R * c) +
            dot(d, RAKBR * b) * dot(a, R * c) +
            dot(d, RAKBR * c) * dot(a, R * b)
    ) * M0

    RAK = R * A * K
    T4 = -0.5 * (
        dot(c, RAK * d) * dot(a, R * b) +
            dot(d, RAK * c) * dot(a, R * b) +
            dot(a, RAK * c) * dot(d, R * b) +
            dot(a, RAK * d) * dot(c, R * b) +
            dot(b, RAK * c) * dot(d, R * a) +
            dot(b, RAK * d) * dot(a, R * c)
    ) * M0

    KBR = K * B * R
    T5 = -0.5 * (
        dot(a, KBR * c) * dot(d, R * b) +
            dot(a, KBR * d) * dot(c, R * b) +
            dot(a, KBR * b) * dot(c, R * d) +
            dot(b, KBR * c) * dot(d, R * a) +
            dot(b, KBR * d) * dot(c, R * a) +
            dot(b, KBR * a) * dot(c, R * d)
    ) * M0

    expected = T1 + T2 + T3 + T4 + T5

    @test isapprox(result, expected; atol = 1.0e-10)
end

@testset "compute_matrix_element for Rank2Gaussian and CoulombOperator" begin
    A = [1.0 0.2; 0.2 1.5]
    B = [0.9 0.1; 0.1 1.2]
    a = [0.6, -0.5]
    b = [-0.3, 0.9]
    c = [0.2, 0.4]
    d = [-0.2, -0.7]
    w = [1.0, -1.0]
    coefficient = 2.0

    bra = Rank2Gaussian(A, a, b)
    ket = Rank2Gaussian(B, c, d)
    op = CoulombOperator(coefficient, w)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)

    β = 1 / dot(w, R * w)
    Rw = R * w

    M2 = 0.25 * (
        dot(a, R * b) * dot(c, R * d) +
            dot(a, R * c) * dot(b, R * d) +
            dot(a, R * d) * dot(b, R * c)
    ) * M0

    term1 = 2 * sqrt(β / π) * M2

    q2_1 = dot(a, Rw) * dot(Rw, b) * dot(c, R * d)
    q2_2 = dot(a, Rw) * dot(Rw, c) * dot(b, R * d)
    q2_3 = dot(a, Rw) * dot(Rw, d) * dot(b, R * c)
    q2_4 = dot(b, Rw) * dot(Rw, c) * dot(a, R * d)
    q2_5 = dot(b, Rw) * dot(Rw, d) * dot(a, R * c)
    q2_6 = dot(c, Rw) * dot(Rw, d) * dot(a, R * b)

    term2 = -2 * sqrt(β / π) * β / 3 * 0.25 * (
        q2_1 + q2_2 + q2_3 + q2_4 + q2_5 + q2_6
    ) * M0

    q4_1 = dot(a, Rw) * dot(Rw, b) * dot(c, Rw) * dot(Rw, d)
    q4_2 = dot(a, Rw) * dot(Rw, c) * dot(b, Rw) * dot(Rw, d)
    q4_3 = dot(a, Rw) * dot(Rw, d) * dot(b, Rw) * dot(Rw, c)

    term3 = 2 * sqrt(β / π) * β^2 / 10 * 0.5 * (
        q4_1 + q4_2 + q4_3
    ) * M0

    expected = coefficient * (term1 + term2 + term3)

    @test isapprox(result, expected; atol = 1.0e-10)
end
