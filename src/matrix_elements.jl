using SpecialFunctions: erf

"""
compute_matrix_element(bra, ket, op)

Compute the matrix element ⟨bra|op|ket⟩ using analytic expressions.
"""

# Supervector contraction tr(xᵀ M y): reduces to xᵀ M y for length-N vectors,
# and correctly sums the three Cartesian channels for N×3 shifts.
_superdot(x, M, y) = tr(transpose(x) * M * y)

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    A, B = parent(bra.A), parent(ket.A)
    a, b = bra.s, ket.s
    S = A + B
    R = inv(S)
    n = size(S, 1)
    M0 = (π^n / det(S))^(3 / 2)
    v = a + b
    return exp(0.25 * _superdot(v, R, v)) * M0
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
        throw(
            ArgumentError(
                "Rank2 overlap with nonzero shifts currently requires single-component polarizations"
            )
        )
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
    v = a + b
    term = 6 * tr(B * K * A * R) +
        _superdot(b, K, a) +
        _superdot(v, R * B * K * A * R, v) -
        _superdot(v, R * B * K, a) -
        _superdot(b, K * A * R, v)
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
    v = a + b
    β = 1 / (w' * R * w)
    # With 3D shifts the mean displacement wᵀR v is a Cartesian 3-vector; the
    # shifted Coulomb kernel depends on its magnitude q = ‖wᵀR v‖/2.
    qvec = 0.5 * (transpose(w) * R * v)
    q = norm(qvec)
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

# Overlap prefactor (π^n / det B)^(3/2) for an n-coordinate, 3D ECG.
_overlap_prefactor(B) = (π^size(B, 1) / det(B))^(3 / 2)

# Central rank-0 datum after shifting the combined exponent by `W`: the inverse
# exponent `R`, the combined shift `v = bra.s + ket.s`, and the scalar overlap
# `M` of the W-shifted Gaussians.  Central operators reduce to `coefficient·M`.
function _updated_rank0_data(bra::Rank0Gaussian, ket::Rank0Gaussian, W)
    Bp = parent(bra.A) + parent(ket.A) + W
    R = inv(Bp)
    v = bra.s + ket.s
    M = exp(_superdot(v, R, v) / 4) * _overlap_prefactor(Bp)
    return (; R, v, M)
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::GaussianOperator)
    # V(r_ij) = coefficient·exp(-γ (w'r)²) shifts the exponent by γ w wᵀ.
    data = _updated_rank0_data(bra, ket, op.γ * (op.w * op.w'))
    return op.coefficient * data.M
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::ManyBodyGaussianOperator)
    # V = coefficient·exp(-rᵀ W r) shifts the exponent by the full matrix W.
    data = _updated_rank0_data(bra, ket, op.W)
    return op.coefficient * data.M
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::OscillatorOperator)
    # V = coefficient·|wᵀr|² is a second radial moment of the (unshifted-exponent)
    # overlap: ⟨(wᵀr)²⟩ = 3·(wᵀ R w)/2 (3D variance) + ‖wᵀ R v / 2‖² (mean²).
    w = op.w
    data = _updated_rank0_data(bra, ket, zero(parent(bra.A)))
    R, v, M = data.R, data.v, data.M
    mean = w' * R * v / 2
    moment = 3 * (w' * R * w) / 2 + dot(vec(mean), vec(mean))
    return op.coefficient * moment * M
end

# ── Spin-½ local matrix elements ─────────────────────────────────────────────
# ⟨br|Sᶜ|kt⟩ for a single site; c ∈ (1,2,3) = (x,y,z), in units of ℏ.
function _spin_element(c::Int, br::SpinProjection, kt::SpinProjection)
    if c == 3
        return br == kt ? (br == up ? 0.5 + 0.0im : -0.5 + 0.0im) : 0.0 + 0.0im
    elseif c == 1
        return br == kt ? (0.0 + 0.0im) : (0.5 + 0.0im)
    else
        return (br == up && kt == down) ? -0.5im :
            (br == down && kt == up) ? 0.5im : 0.0 + 0.0im
    end
end

_spin_vec(br::SpinProjection, kt::SpinProjection) =
    ComplexF64[_spin_element(c, br, kt) for c in 1:3]

# Product of spectator-site δ's; sites in `exclude` are handled by spin operators.
function _spectator(bra::SpinState, ket::SpinState, exclude)
    length(bra.projections) == length(ket.projections) || return 0.0
    for k in eachindex(bra.projections)
        k in exclude && continue
        bra.projections[k] == ket.projections[k] || return 0.0
    end
    return 1.0
end

_spin_overlap(bra::SpinState, ket::SpinState) = _spectator(bra, ket, ())

# Non-conjugating contraction Σₖ x[k] y[k].
_cdot(x, y) = sum(x[k] * y[k] for k in eachindex(x))

# Spin overlap and central operators factor through the spin overlap.
_compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian) =
    _compute_matrix_element(bra.orbital, ket.orbital) * _spin_overlap(bra.spin, ket.spin)

_compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian, op::FewBodyHamiltonians.Operator) =
    _compute_matrix_element(bra.orbital, ket.orbital, op) * _spin_overlap(bra.spin, ket.spin)

function _compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian, op::GaussianTensorOperator)
    i, j, w = op.i, op.j, op.w
    spectator = _spectator(bra.spin, ket.spin, (i, j))
    spectator == 0 && return 0.0 + 0.0im
    data = _updated_rank0_data(bra.orbital, ket.orbital, op.γ * (op.w * op.w'))
    R, v, M = data.R, data.v, data.M
    q = vec(w' * R * v / 2)          # 3-vector mean displacement
    var = (w' * R * w) / 2           # per-Cartesian variance
    Si = _spin_vec(bra.spin.projections[i], ket.spin.projections[i])
    Sj = _spin_vec(bra.spin.projections[j], ket.spin.projections[j])
    # Σ_ab ⟨rₐr_b⟩ Siₐ Sj_b = var (Si·Sj) + (Si·q)(Sj·q)
    spatial = var * _cdot(Si, Sj) + _cdot(Si, q) * _cdot(Sj, q)
    result = spatial
    if op.traceless
        trace = 3 * var + dot(q, q)  # ⟨r²⟩
        result -= trace / 3 * _cdot(Si, Sj)
    end
    return op.coefficient * M * spectator * result
end

function _compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian, op::GaussianSpinOrbitOperator)
    i, j, w = op.i, op.j, op.w
    data = _updated_rank0_data(bra.orbital, ket.orbital, op.γ * (op.w * op.w'))
    R, M = data.R, data.M
    left = vec(w' * R * parent(bra.orbital.s))    # 3-vector from bra shift
    right = vec(w' * R * parent(ket.orbital.s))   # 3-vector from ket shift
    Lvec = -im .* cross(left, right) .* (M / 4)   # complex orbital 3-vector
    spec_i = _spectator(bra.spin, ket.spin, (i,))
    spec_j = _spectator(bra.spin, ket.spin, (j,))
    Si = _spin_vec(bra.spin.projections[i], ket.spin.projections[i])
    Sj = _spin_vec(bra.spin.projections[j], ket.spin.projections[j])
    Sfull = spec_i .* Si .+ spec_j .* Sj          # ⟨Sᵢ + Sⱼ⟩ (3-vector)
    return op.coefficient * _cdot(Lvec, Sfull)
end
