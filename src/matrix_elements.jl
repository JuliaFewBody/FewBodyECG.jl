using SpecialFunctions: erf

"""
compute_matrix_element(bra, ket, op)

Compute the matrix element ⟨bra|op|ket⟩ using analytic expressions.
"""

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    A, B = parent(bra.A), parent(ket.A)
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

    cov = 0.5 * _polar_contract(bra.a, R, ket.a)
    mean_term = 0.25 * _polar_project_dot(
        bra.a,
        Rt,
        ket.a,
        Rt,
    )
    return (cov + mean_term) * I0
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian)
    _check_polarization_compat(bra.a, bra.b)
    _check_polarization_compat(ket.a, ket.b)
    _check_polarization_compat(bra.a, ket.a)
    if (any(!iszero, bra.s) || any(!iszero, ket.s)) && _pol_ncomp(bra.a) > 1
        throw(ArgumentError(
            "Rank2 overlap with nonzero shifts currently requires single-component polarizations"
        ))
    end

    A, B = bra.A, ket.A
    a, b = bra.s, ket.s
    S = A + B
    R = inv(S)
    n = size(S, 1)

    t = a + b
    Rt = R * t
    I0 = exp(0.25 * t' * R * t) * (π^n / det(S))^(3 / 2)

    μ = 0.5 * Rt

    Xμ = sum(_polar_projection(bra.a, μ))
    Yμ = sum(_polar_projection(bra.b, μ))
    Zμ = sum(_polar_projection(ket.a, μ))
    Wμ = sum(_polar_projection(ket.b, μ))

    cov(v, w) = 0.5 * _polar_contract(v, R, w)
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
    A, B = parent(bra.A), parent(ket.A)
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
    a = bra.a
    b = ket.a
    K = op.K
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)
    M1 = 0.5 * _polar_contract(a, R, b) * M0

    T1 = 6 * tr(B * K * A * R) * M1
    T2 = _polar_contract(a, K, b) * M0
    T3 = (
        _polar_contract(b, R * A * K * B * R, a) +
        _polar_contract(a, R * A * K * B * R, b)
    ) * M0
    T4 = -_polar_contract(a, R * A * K, b) * M0
    T5 = -_polar_contract(b, R * B * K, a) * M0

    return T1 + T2 + T3 + T4 + T5
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian, op::KineticOperator)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("Rank2 kinetic matrix elements currently require zero shifts"))
    end

    A, B = bra.A, ket.A
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    _check_polarization_compat(a, b)
    _check_polarization_compat(c, d)
    _check_polarization_compat(a, c)
    K = op.K
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)

    M2 = 0.25 * (
        _polar_contract(a, R, b) *
            _polar_contract(c, R, d) +
            _polar_contract(a, R, c) *
            _polar_contract(b, R, d) +
            _polar_contract(a, R, d) *
            _polar_contract(b, R, c)
    ) * M0

    T1 = 6 * tr(B * K * A * R) * M2

    T2 = 0.5 * (
        _polar_contract(a, K, c) *
            _polar_contract(b, R, d) +
            _polar_contract(a, K, d) *
            _polar_contract(b, R, c) +
            _polar_contract(b, K, c) *
            _polar_contract(a, R, d) +
            _polar_contract(b, K, d) *
            _polar_contract(a, R, c)
    ) * M0

    RAKBR = R * A * K * B * R
    T3 = 0.5 * (
        _polar_contract(a, RAKBR, b) *
            _polar_contract(c, R, d) +
            _polar_contract(a, RAKBR, c) *
            _polar_contract(b, R, d) +
            _polar_contract(a, RAKBR, d) *
            _polar_contract(b, R, c) +
            _polar_contract(b, RAKBR, a) *
            _polar_contract(c, R, d) +
            _polar_contract(b, RAKBR, c) *
            _polar_contract(a, R, d) +
            _polar_contract(b, RAKBR, d) *
            _polar_contract(a, R, c) +
            _polar_contract(c, RAKBR, a) *
            _polar_contract(b, R, d) +
            _polar_contract(c, RAKBR, b) *
            _polar_contract(a, R, d) +
            _polar_contract(c, RAKBR, d) *
            _polar_contract(a, R, b) +
            _polar_contract(d, RAKBR, a) *
            _polar_contract(b, R, c) +
            _polar_contract(d, RAKBR, b) *
            _polar_contract(a, R, c) +
            _polar_contract(d, RAKBR, c) *
            _polar_contract(a, R, b)
    ) * M0

    RAK = R * A * K
    T4 = -0.5 * (
        _polar_contract(c, RAK, d) *
            _polar_contract(a, R, b) +
            _polar_contract(d, RAK, c) *
            _polar_contract(a, R, b) +
            _polar_contract(a, RAK, c) *
            _polar_contract(d, R, b) +
            _polar_contract(a, RAK, d) *
            _polar_contract(c, R, b) +
            _polar_contract(b, RAK, c) *
            _polar_contract(d, R, a) +
            _polar_contract(b, RAK, d) *
            _polar_contract(a, R, c)
    ) * M0

    KBR = K * B * R
    T5 = -0.5 * (
        _polar_contract(a, KBR, c) *
            _polar_contract(d, R, b) +
            _polar_contract(a, KBR, d) *
            _polar_contract(c, R, b) +
            _polar_contract(a, KBR, b) *
            _polar_contract(c, R, d) +
            _polar_contract(b, KBR, c) *
            _polar_contract(d, R, a) +
            _polar_contract(b, KBR, d) *
            _polar_contract(c, R, a) +
            _polar_contract(b, KBR, a) *
            _polar_contract(c, R, d)
    ) * M0

    return T1 + T2 + T3 + T4 + T5
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::CoulombOperator)
    A, B = parent(bra.A), parent(ket.A)
    a, b = bra.s, ket.s
    w = op.w
    S = A + B
    R = inv(S)
    M = _compute_matrix_element(bra, ket)
    β = 1 / (w' * R * w)
    q = 0.5 * (w' * R * (a + b))
    # Use the limiting form 2√(β/π) when the scaled argument x = √β·q is
    # small so that erf(x)/q = √β·erf(x)/x → 2√(β/π) accurately.
    # Thresholding on x (not q alone) correctly handles all β values.
    x = sqrt(β) * q
    f = abs(x) < 1.0e-7 ? (2 * sqrt(β / π)) : (erf(x) / q)
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
    M1 = 0.5 * _polar_contract(a, R, b) * M0
    Rw = R * w

    return op.coefficient * (
        2 * sqrt(β / π) * M1 -
        sqrt(β / π) * β / 3 *
        _polar_project_dot(a, Rw, b, Rw) * M0
    )
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian, op::CoulombOperator)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("Rank2 Coulomb matrix elements currently require zero shifts"))
    end

    A, B = bra.A, ket.A
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    _check_polarization_compat(a, b)
    _check_polarization_compat(c, d)
    _check_polarization_compat(a, c)
    w = op.w
    R = inv(A + B)
    n = size(R, 1)
    M0 = (π^n / det(A + B))^(3 / 2)

    β = 1 / dot(w, R * w)
    Rw = R * w

    M2 = 0.25 * (
        _polar_contract(a, R, b) *
            _polar_contract(c, R, d) +
            _polar_contract(a, R, c) *
            _polar_contract(b, R, d) +
            _polar_contract(a, R, d) *
            _polar_contract(b, R, c)
    ) * M0

    term1 = 2 * sqrt(β / π) * M2

    q2_1 = _polar_project_dot(a, Rw, b, Rw) *
        _polar_contract(c, R, d)
    q2_2 = _polar_project_dot(a, Rw, c, Rw) *
        _polar_contract(b, R, d)
    q2_3 = _polar_project_dot(a, Rw, d, Rw) *
        _polar_contract(b, R, c)
    q2_4 = _polar_project_dot(b, Rw, c, Rw) *
        _polar_contract(a, R, d)
    q2_5 = _polar_project_dot(b, Rw, d, Rw) *
        _polar_contract(a, R, c)
    q2_6 = _polar_project_dot(c, Rw, d, Rw) *
        _polar_contract(a, R, b)

    term2 = -2 * sqrt(β / π) * β / 3 * 0.25 * (
        q2_1 + q2_2 + q2_3 + q2_4 + q2_5 + q2_6
    ) * M0

    q4_1 = _polar_project_dot(a, Rw, b, Rw) *
        _polar_project_dot(c, Rw, d, Rw)
    q4_2 = _polar_project_dot(a, Rw, c, Rw) *
        _polar_project_dot(b, Rw, d, Rw)
    q4_3 = _polar_project_dot(a, Rw, d, Rw) *
        _polar_project_dot(b, Rw, c, Rw)

    term3 = 2 * sqrt(β / π) * β^2 / 10 * 0.5 * (
        q4_1 + q4_2 + q4_3
    ) * M0

    return op.coefficient * (term1 + term2 + term3)
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::GaussianOperator)
    A, B = parent(bra.A), parent(ket.A)
    a, b = bra.s, ket.s
    γ, w = op.γ, op.w
    n = size(A, 1)
    # V(r_ij) = exp(-γ (w'r)²) shifts the exponent matrix S → S' = S + γ ww'
    S_prime = Symmetric(A + B + γ * (w * w'))
    R_prime = inv(S_prime)
    return op.coefficient * exp(0.25 * (a + b)' * R_prime * (a + b)) * (π^n / det(S_prime))^(3 / 2)
end
