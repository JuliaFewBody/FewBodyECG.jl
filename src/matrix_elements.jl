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
