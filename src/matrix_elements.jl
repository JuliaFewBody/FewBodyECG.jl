using SpecialFunctions: erf

function _require_zero_shift(g::Union{Rank1Gaussian, Rank2Gaussian})
    all(iszero, g.s) || throw(ArgumentError("$(nameof(typeof(g))) matrix elements require zero shifts"))
    return nothing
end

_superdot(x, M, y) = tr(x' * M * y)

function _overlap_prefactor(B)
    return (π^size(B, 1) / det(B))^(3 / 2)
end

function _rank0_data(bra::Rank0Gaussian, ket::Rank0Gaussian)
    B, A = bra.A, ket.A
    R = inv(A + B)
    v = ket.s + bra.s
    M = exp(_superdot(v, R, v) / 4) * _overlap_prefactor(A + B)
    return (; A, B, R, v, M)
end

function _updated_rank0_data(bra::Rank0Gaussian, ket::Rank0Gaussian, W)
    data = _rank0_data(bra, ket)
    Bprime = data.A + data.B + W
    Rprime = inv(Bprime)
    Mprime = exp(_superdot(data.v, Rprime, data.v) / 4) * _overlap_prefactor(Bprime)
    return (; data..., Bprime, Rprime, Mprime)
end

"""
compute_matrix_element(bra, ket, op)

Compute the matrix element ⟨bra|op|ket⟩ using analytic expressions.
"""

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    return _rank0_data(bra, ket).M
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian)
    _require_zero_shift(bra)
    _require_zero_shift(ket)
    R = inv(bra.A + ket.A)
    M0 = _overlap_prefactor(bra.A + ket.A)
    return dot(bra.a, R * ket.a) * M0 / 2
end

function _compute_matrix_element(bra::Rank2Gaussian, ket::Rank2Gaussian)
    _require_zero_shift(bra)
    _require_zero_shift(ket)
    R = inv(bra.A + ket.A)
    M0 = _overlap_prefactor(bra.A + ket.A)
    return (
        dot(bra.a, R * bra.b) * dot(ket.a, R * ket.b) +
            dot(bra.a, R * ket.a) * dot(bra.b, R * ket.b) +
            dot(bra.a, R * ket.b) * dot(bra.b, R * ket.a)
    ) * M0 / 4
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::KineticOperator)
    data = _rank0_data(bra, ket)
    a, b = ket.s, bra.s
    term = 6 * tr(data.B * op.K * data.A * data.R) +
        _superdot(b, op.K, a) +
        _superdot(data.v, data.R * data.B * op.K * data.A * data.R, data.v) -
        _superdot(data.v, data.R * data.B * op.K, a) -
        _superdot(b, op.K * data.A * data.R, data.v)
    return term * data.M
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::KineticOperator)
    _require_zero_shift(bra)
    _require_zero_shift(ket)

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
    _require_zero_shift(bra)
    _require_zero_shift(ket)

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
    data = _rank0_data(bra, ket)
    β = 1 / (op.w' * data.R * op.w)[1]
    q = norm(vec(op.w' * data.R * data.v / 2))
    f = iszero(q) ? 2 * sqrt(β / π) : erf(sqrt(β) * q) / q
    return op.coefficient * f * data.M
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::GaussianPotential)
    data = _updated_rank0_data(bra, ket, op.gamma * op.w * op.w')
    return op.coefficient * data.Mprime
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::OscillatorPotential)
    data = _rank0_data(bra, ket)
    q = vec(op.w' * data.R * data.v / 2)
    radial_square = 3 * (op.w' * data.R * op.w)[1] / 2 + dot(q, q)
    return op.coefficient * radial_square * data.M
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::ManyBodyGaussianPotential)
    data = _updated_rank0_data(bra, ket, op.W)
    return op.coefficient * data.Mprime
end

function _compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian)
    return spin_overlap(bra.spin, ket.spin) * _compute_matrix_element(bra.orbital, ket.orbital)
end

function _compute_matrix_element(
        bra::SpinGaussian,
        ket::SpinGaussian,
        op::Union{
            KineticOperator,
            CoulombOperator,
            GaussianPotential,
            OscillatorPotential,
            ManyBodyGaussianPotential,
        },
    )
    return spin_overlap(bra.spin, ket.spin) *
        _compute_matrix_element(bra.orbital, ket.orbital, op)
end

function _check_spin_states(bra::SpinState, ket::SpinState, i::Integer, j::Integer)
    length(bra.projections) == length(ket.projections) ||
        throw(ArgumentError("spin states must have the same number of sites"))
    i in eachindex(bra.projections) || throw(ArgumentError("invalid spin site $i"))
    j in eachindex(bra.projections) || throw(ArgumentError("invalid spin site $j"))
    return nothing
end

function _spin_pair_element(
        bra::SpinState,
        ket::SpinState,
        i::Integer,
        j::Integer,
        alpha::Symbol,
        beta::Symbol,
    )
    _check_spin_states(bra, ket, i, j)
    result = 1.0 + 0.0im
    for site in eachindex(bra.projections)
        if site == i
            result *= spin_element(bra, ket, site, alpha)
        elseif site == j
            result *= spin_element(bra, ket, site, beta)
        elseif bra.projections[site] != ket.projections[site]
            return 0.0 + 0.0im
        end
    end
    return result
end

function _spin_single_element(bra::SpinState, ket::SpinState, i::Integer, alpha::Symbol)
    i in eachindex(bra.projections) || throw(ArgumentError("invalid spin site $i"))
    length(bra.projections) == length(ket.projections) ||
        throw(ArgumentError("spin states must have the same number of sites"))
    for site in eachindex(bra.projections)
        site == i && continue
        bra.projections[site] == ket.projections[site] || return 0.0 + 0.0im
    end
    return spin_element(bra, ket, i, alpha)
end

function _compute_matrix_element(
        bra::SpinGaussian{<:Rank0Gaussian},
        ket::SpinGaussian{<:Rank0Gaussian},
        op::GaussianTensorPotential,
    )
    data = _updated_rank0_data(bra.orbital, ket.orbital, op.gamma * op.w * op.w')
    q = vec(op.w' * data.Rprime * data.v / 2)
    variance = (op.w' * data.Rprime * op.w)[1] / 2
    axes = (:x, :y, :z)
    result = 0.0 + 0.0im

    for (a, alpha) in enumerate(axes), (b, beta) in enumerate(axes)
        coordinate = data.Mprime * (variance * (a == b) + q[a] * q[b])
        result += _spin_pair_element(bra.spin, ket.spin, op.i, op.j, alpha, beta) * coordinate
    end

    if op.traceless
        spin_dot = sum(
            _spin_pair_element(bra.spin, ket.spin, op.i, op.j, alpha, alpha)
                for alpha in axes
        )
        result -= spin_dot * data.Mprime * (3 * variance + dot(q, q)) / 3
    end

    return op.coefficient * result
end

function _compute_matrix_element(
        bra::SpinGaussian{<:Rank0Gaussian},
        ket::SpinGaussian{<:Rank0Gaussian},
        op::GaussianSpinOrbitPotential,
    )
    data = _updated_rank0_data(bra.orbital, ket.orbital, op.gamma * op.w * op.w')
    u = data.Rprime * data.v
    left = vec(op.w' * u)
    right = vec(op.w' * ket.orbital.s - op.w' * data.A * u)
    orbital_momentum = -im * cross(left, right) * data.Mprime / 4
    axes = (:x, :y, :z)
    spin_orbit = sum(
        orbital_momentum[a] * (
                _spin_single_element(bra.spin, ket.spin, op.i, alpha) +
                _spin_single_element(bra.spin, ket.spin, op.j, alpha)
            ) for (a, alpha) in enumerate(axes)
    )
    return op.coefficient * spin_orbit
end

function _compute_matrix_element(bra::Rank1Gaussian, ket::Rank1Gaussian, op::CoulombOperator)
    _require_zero_shift(bra)
    _require_zero_shift(ket)

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
    _require_zero_shift(bra)
    _require_zero_shift(ket)

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
