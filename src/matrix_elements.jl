using SpecialFunctions: erf

"""
compute_matrix_element(bra, ket, op)

Compute the matrix element ⟨bra|op|ket⟩ using analytic expressions.
"""

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    A, B = bra.A, ket.A
    a, b = bra.s, ket.s
    S = A + B
    R = inv(S)
    n = size(S, 1)
    M0 = (π^n / det(S))^(3 / 2)
    return exp(0.25 * (a + b)' * R * (a + b)) * M0
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian)
    A, B = bra.A, ket.A
    a, b = bra.s, ket.s
    S = A + B
    R = inv(S)
    n = size(S, 1)

    t = a + b
    Rt = R * t
    I0 = exp(0.25 * t' * R * t) * (π^n / det(S))^(3 / 2)

    cov = 0.5 * dot(bra.a, R * ket.a)
    mean_term = 0.25 * dot(bra.a, Rt) * dot(ket.a, Rt)
    return (cov + mean_term) * I0
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian)
    A, B = bra.A, ket.A
    a, b = bra.s, ket.s
    S = A + B
    R = inv(S)
    n = size(S, 1)

    t = a + b
    Rt = R * t
    I0 = exp(0.25 * t' * R * t) * (π^n / det(S))^(3 / 2)

    μ = 0.5 * Rt

    Xμ = dot(bra.a, μ)
    Yμ = dot(bra.b, μ)
    Zμ = dot(ket.a, μ)
    Wμ = dot(ket.b, μ)

    cov(v, w) = 0.5 * dot(v, R * w)
    XY = cov(bra.a, bra.b)
    XZ = cov(bra.a, ket.a)
    XW = cov(bra.a, ket.b)
    YZ = cov(bra.b, ket.a)
    YW = cov(bra.b, ket.b)
    ZW = cov(ket.a, ket.b)

    moment4 =
        Xμ * Yμ * Zμ * Wμ +
        Xμ * Yμ * ZW +
        Xμ * Zμ * YW +
        Xμ * Wμ * YZ +
        Yμ * Zμ * XW +
        Yμ * Wμ * XZ +
        Zμ * Wμ * XY +
        XY * ZW +
        XZ * YW +
        XW * YZ

    return moment4 * I0
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::KineticOperator)
    A, B = bra.A, ket.A
    a, b = bra.s, ket.s
    K = op.K
    S = A + B
    R = inv(S)
    M = _compute_matrix_element(bra, ket)
    term = 6 * tr(B * K * A * R) +
        b' * K * a +
        (a + b)' * R * B * K * A * R * (a + b) -
        (a + b)' * R * B * K * a -
        b' * K * A * R * (a + b)
    return term * M
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::KineticOperator)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("Rank1 kinetic matrix elements currently require zero shifts"))
    end

    A, B = bra.A, ket.A
    a = vec(bra.a)
    b = vec(ket.a)
    K = op.K
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0

    T1 = 6 * tr(B * K * A * R) * M1
    T2 = dot(a, K * b) * M0
    T3 = (dot(b, R * A * K * B * R * a) + dot(a, R * A * K * B * R * b)) * M0
    T4 = -dot(a, R * A * K * b) * M0
    T5 = -dot(b, R * B * K * a) * M0

    return T1 + T2 + T3 + T4 + T5
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian, op::KineticOperator)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("Rank2 kinetic matrix elements currently require zero shifts"))
    end

    A, B = bra.A, ket.A
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    K = op.K
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

    return T1 + T2 + T3 + T4 + T5
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::CoulombOperator)
    A, B = bra.A, ket.A
    a, b = bra.s, ket.s
    w = op.w
    S = A + B
    R = inv(S)
    M = _compute_matrix_element(bra, ket)
    β = 1 / (w' * R * w)
    q = 0.5 * (w' * R * (a + b))
    f = abs(q) < 1.0e-12 ? (2 * sqrt(β / π)) : (erf(sqrt(β) * q) / q)
    return op.coefficient * f * M
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::CoulombOperator)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("Rank1 Coulomb matrix elements currently require zero shifts"))
    end

    A, B, a, b, w = bra.A, ket.A, bra.a, ket.a, op.w
    R = inv(A + B)
    n = size(R, 1)
    β = 1 / (dot(w, R * w))
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0
    Rw = R * w

    return op.coefficient * (2 * sqrt(β / π) * M1 - sqrt(β / π) * β / 3 * dot(a, Rw) * dot(Rw, b) * M0)
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian, op::CoulombOperator)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("Rank2 Coulomb matrix elements currently require zero shifts"))
    end

    A, B = bra.A, ket.A
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    w = op.w
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

    return op.coefficient * (term1 + term2 + term3)
end
