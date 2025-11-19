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
    A, B, a, b, w = bra.A, ket.A, bra.a, ket.a, op.w
    R = inv(A + B)
    β = 1 / (dot(w, R * w))
    M0 = (π^length(R) / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(b, R * a) * M0
    q2 = 0.25 * dot(a .+ b, R * (w * w') * (a .+ b))
    return 2 * sqrt(β / π) * M1 - sqrt(β^3 / π) / 3 * q2 * M0
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::KineticOperator)
    a = bra.a isa AbstractVector{<:AbstractVector} ? vec(bra.a[1]) : vec(bra.a)
    b = ket.a isa AbstractVector{<:AbstractVector} ? vec(ket.a[1]) : vec(ket.a)

    A, B = bra.A, ket.A
    K = op.K
    R = inv(A + B)
    M0 = (π^length(R) / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(b, R * a) * M0

    T1 = 6 * tr(B * K * A * R) * M1
    T2 = dot(b, a) * M0
    T3 = dot(a, R * B * A * R * b) * M0
    T4 = dot(b, R * B * a) * M0
    T5 = dot(a, R * A * b) * M0

    return T1 + T2 + T3 - T4 - T5
end