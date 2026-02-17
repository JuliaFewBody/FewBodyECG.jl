using SpecialFunctions: erf

"""
compute_matrix_element(bra, ket, op)

Compute the matrix element ⟨bra|op|ket⟩ using analytic expressions.

The formulas follow Fedorov et al., Few-Body Syst (2024) 65:75.
Code variable convention: A = bra.A, B = ket.A (opposite to paper's A = ket, B = bra).
Paper equations are applied with the appropriate variable mapping.
"""

# Eq. 9: ⟨B|T|A⟩ = 6 Tr(B K A R) M₀
function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::KineticOperator)
    A, B = bra.A, ket.A
    K = op.K
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)
    return 6 * tr(B * K * A * R) * M0
end

# Eq. 10: ⟨B|1/|wᵀr||A⟩ = 2√(β/π) M₀
function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::CoulombOperator)
    A, B, w = bra.A, ket.A, op.w
    R = inv(A + B)
    n = size(R, 1)
    β = 1 / (dot(w, R * w))
    M0 = (π^n / det(A + B))^(3 / 2)
    return op.coefficient * 2 * sqrt(β / π) * M0
end

# Eq. 28: Rank-1 Coulomb
function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::CoulombOperator)
    # Code vars: A = bra.A (paper B), B = ket.A (paper A)
    # a = bra.a (paper b), b = ket.a (paper a)
    A, B, a, b, w = bra.A, ket.A, bra.a, ket.a, op.w
    R = inv(A + B)
    n = size(R, 1)
    β = 1 / (dot(w, R * w))
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0
    Rw = R * w
    # Eq. 28: 2√(β/π) M₁ - √(β/π) β/3 bᵀRwwᵀRa M₀
    return op.coefficient * (2 * sqrt(β / π) * M1 - sqrt(β / π) * β / 3 * dot(a, Rw) * dot(Rw, b) * M0)
end

# Eq. 23: Rank-1 Kinetic
function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::KineticOperator)
    # Code vars: A = bra.A (paper B), B = ket.A (paper A)
    # a = bra.a (paper b), b = ket.a (paper a)
    A, B = bra.A, ket.A
    a = vec(bra.a)
    b = vec(ket.a)
    K = op.K
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * dot(a, R * b) * M0

    # Eq. 18: 6 Tr(BKAR) M₁
    T1 = 6 * tr(B * K * A * R) * M1
    # Eq. 19: bᵀKa M₀ → aᵀKb M₀ (code vars)
    T2 = dot(a, K * b) * M0
    # Eq. 20: aᵀRBKARb M₀ + bᵀRBKARa M₀ → bᵀRAKBRa M₀ + aᵀRAKBRb M₀ (code vars)
    T3 = (dot(b, R * A * K * B * R * a) + dot(a, R * A * K * B * R * b)) * M0
    # Eq. 21: -bᵀRBKa M₀ → -aᵀRAKb M₀ (code vars)
    T4 = -dot(a, R * A * K * b) * M0
    # Eq. 22: -aᵀRAKb M₀ → -bᵀRBKa M₀ (code vars)
    T5 = -dot(b, R * B * K * a) * M0

    return T1 + T2 + T3 + T4 + T5
end

# Eqs. 34-38: Rank-2 Kinetic
function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian, op::KineticOperator)
    # Code vars: A = bra.A (paper B), B = ket.A (paper A)
    # a = bra.a (paper c), b = bra.b (paper d), c = ket.a (paper a), d = ket.b (paper b)
    A, B = bra.A, ket.A
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    K = op.K
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)

    # Eq. 33: M₂ overlap (symmetric in bra/ket)
    M2 = 0.25 * (
        dot(a, R * b) * dot(c, R * d) +
            dot(a, R * c) * dot(b, R * d) +
            dot(a, R * d) * dot(b, R * c)
    ) * M0

    # Eq. 34: 6 Tr(BKAR) M₂
    T1 = 6 * tr(B * K * A * R) * M2

    # Eq. 35: bᵀKa M terms (K only, no A or B dependence beyond R)
    T2 = 0.5 * (
        dot(a, K * c) * dot(b, R * d) +
            dot(a, K * d) * dot(b, R * c) +
            dot(b, K * c) * dot(a, R * d) +
            dot(b, K * d) * dot(a, R * c)
    ) * M0

    # Eq. 36: (a+b)ᵀRBKAR(a+b) M terms → uses R·A·K·B·R in code vars
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

    # Eq. 37: -(a+b+c+d)ᵀRBK(a+b) M terms → uses R·A·K in code vars
    RAK = R * A * K
    T4 = -0.5 * (
        dot(c, RAK * d) * dot(a, R * b) +
            dot(d, RAK * c) * dot(a, R * b) +
            dot(a, RAK * c) * dot(d, R * b) +
            dot(a, RAK * d) * dot(c, R * b) +
            dot(b, RAK * c) * dot(d, R * a) +
            dot(b, RAK * d) * dot(a, R * c)
    ) * M0

    # Eq. 38: -(c+d)ᵀKAR(a+b+c+d) M terms → uses K·B·R in code vars
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

# Eq. 41: Rank-2 Coulomb
function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian, op::CoulombOperator)
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

    # First term: 2√(β/π) M₂
    term1 = 2 * sqrt(β / π) * M2

    # Second term: q² contributions with Rwwᵀ R (note: xᵀRwwᵀRy = dot(x,Rw)*dot(Rw,y))
    q2_1 = dot(a, Rw) * dot(Rw, b) * dot(c, R * d)
    q2_2 = dot(a, Rw) * dot(Rw, c) * dot(b, R * d)
    q2_3 = dot(a, Rw) * dot(Rw, d) * dot(b, R * c)
    q2_4 = dot(b, Rw) * dot(Rw, c) * dot(a, R * d)
    q2_5 = dot(b, Rw) * dot(Rw, d) * dot(a, R * c)
    q2_6 = dot(c, Rw) * dot(Rw, d) * dot(a, R * b)

    term2 = -2 * sqrt(β / π) * β / 3 * 0.25 * (
        q2_1 + q2_2 + q2_3 + q2_4 + q2_5 + q2_6
    ) * M0

    # Third term: q⁴ contributions
    q4_1 = dot(a, Rw) * dot(Rw, b) * dot(c, Rw) * dot(Rw, d)
    q4_2 = dot(a, Rw) * dot(Rw, c) * dot(b, Rw) * dot(Rw, d)
    q4_3 = dot(a, Rw) * dot(Rw, d) * dot(b, Rw) * dot(Rw, c)

    term3 = 2 * sqrt(β / π) * β^2 / 10 * 0.5 * (
        q4_1 + q4_2 + q4_3
    ) * M0

    return op.coefficient * (term1 + term2 + term3)
end
