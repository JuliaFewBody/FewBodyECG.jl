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
    M0 = (π^length(R) / det(A + B))^(3 / 2)
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
    β = 1 / (dot(w, R * w))
    M0 = (π^length(R) / det(A + B))^(3 / 2)
    expected = coefficient * 2 * sqrt(β / π) * M0

    @test isapprox(result, expected; atol = 1.0e-10)
end

@testset "compute_matrix_element for Rank1Gaussian and CoulombOperator" begin
    A = rand(2, 2)
    B = rand(2, 2)
    a = [0.5, -0.4]
    b = [-0.2, 0.7]
    w = [1.0, -1.0]
    coefficient = 1.0

    bra = Rank1Gaussian(A, a)
    ket = Rank1Gaussian(B, b)
    op = CoulombOperator(coefficient, w)

    result = _compute_matrix_element(bra, ket, op)

    R = inv(A + B)
    β = 1 / (dot(w, R * w))
    M0 = (π^length(R) / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(b, R * a) * M0
    q2 = 0.25 * dot(a .+ b, R * (w * w') * (a .+ b))
    expected = 2 * sqrt(β / π) * M1 - sqrt(β^3 / π) / 3 * q2 * M0

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
    M0 = (π^length(R) / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(b, R * a) * M0

    T1 = 6 * tr(B * K * A * R) * M1
    T2 = dot(b, a) * M0
    T3 = dot(a, R * B * A * R * b) * M0
    T4 = dot(b, R * B * a) * M0
    T5 = dot(a, R * A * b) * M0

    expected = T1 + T2 + T3 - T4 - T5

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
    M0 = (π^length(R) / det(A + B))^(3 / 2)

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

    T3 = 0.5 * (
        dot(a, R * B * K * A * R * b) * dot(c, R * d) +
            dot(a, R * B * K * A * R * c) * dot(b, R * d) +
            dot(a, R * B * K * A * R * d) * dot(b, R * c) +
            dot(b, R * B * K * A * R * a) * dot(c, R * d) +
            dot(b, R * B * K * A * R * c) * dot(a, R * d) +
            dot(b, R * B * K * A * R * d) * dot(a, R * c) +
            dot(c, R * B * K * A * R * a) * dot(b, R * d) +
            dot(c, R * B * K * A * R * b) * dot(a, R * d) +
            dot(c, R * B * K * A * R * d) * dot(a, R * b) +
            dot(d, R * B * K * A * R * a) * dot(b, R * c) +
            dot(d, R * B * K * A * R * b) * dot(a, R * c) +
            dot(d, R * B * K * A * R * c) * dot(a, R * b)
    ) * M0

    T4 = -0.5 * (
        dot(a, R * B * K * b) * dot(c, R * d) +
            dot(b, R * B * K * a) * dot(c, R * d) +
            dot(c, R * B * K * a) * dot(b, R * d) +
            dot(c, R * B * K * b) * dot(a, R * d) +
            dot(d, R * B * K * a) * dot(b, R * c) +
            dot(d, R * B * K * b) * dot(a, R * c)
    ) * M0

    T5 = -0.5 * (
        dot(c, K * A * a) * dot(b, R * d) +
            dot(c, K * A * b) * dot(a, R * d) +
            dot(c, K * A * d) * dot(a, R * b) +
            dot(d, K * A * a) * dot(b, R * c) +
            dot(d, K * A * b) * dot(a, R * c) +
            dot(d, K * A * c) * dot(a, R * b)
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
    M0 = (π^length(R) / det(A + B))^(3 / 2)

    β = 1 / dot(w, R * w)
    q = 0.5 * dot(w, R * (a + b + c + d))

    M2 = 0.25 * (
        dot(a, R * b) * dot(c, R * d) +
            dot(a, R * c) * dot(b, R * d) +
            dot(a, R * d) * dot(b, R * c)
    ) * M0

    term1 = 2 * sqrt(β / π) * M2

    q2_1 = dot(a, R * (w * w') * b) * dot(c, R * d)
    q2_2 = dot(a, R * (w * w') * c) * dot(b, R * d)
    q2_3 = dot(a, R * (w * w') * d) * dot(b, R * c)
    q2_4 = dot(b, R * (w * w') * c) * dot(a, R * d)
    q2_5 = dot(b, R * (w * w') * d) * dot(a, R * c)
    q2_6 = dot(c, R * (w * w') * d) * dot(a, R * b)

    q4_1 = dot(a, R * (w * w') * b) * dot(c, R * (w * w') * d)
    q4_2 = dot(a, R * (w * w') * c) * dot(b, R * (w * w') * d)
    q4_3 = dot(a, R * (w * w') * d) * dot(b, R * (w * w') * c)

    term2 = -2 * sqrt(β / π) * β / 3 * 0.25 * (
        q2_1 + q2_2 + q2_3 + q2_4 + q2_5 + q2_6
    ) * M0

    term3 = 2 * sqrt(β / π) * β^2 / 10 * 0.5 * (
        q4_1 + q4_2 + q4_3
    ) * M0

    expected = coefficient * (term1 + term2 + term3)

    @test isapprox(result, expected; atol = 1.0e-10)
end
