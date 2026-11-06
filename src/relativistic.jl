"""
    MomentumFourthOperator(u; coefficient=1)

`coefficient * (pᵤ ⋅ pᵤ)^2`, where `pᵤ = -im * Σₖ u[k] ∇ₓₖ` in the
Jacobi frame. For particle `i` in the centre-of-mass rest frame, `u = J[:,i]`,
with `J` from [`jacobi_transform`](@ref). This is an individual momentum fourth
power, not the square of the total kinetic energy. Units have `ℏ = 1`.
Rank1/Rank2 elements follow by the usual shift derivatives.
"""
struct MomentumFourthOperator{T <: Real} <: Operator
    u::Vector{T}
    coefficient::T
    function MomentumFourthOperator(u::AbstractVector{<:Real}; coefficient::Real = 1)
        _check_weight(u, "momentum weights"; nonzero = false)
        _check_coefficient(coefficient)
        T = promote_type(Float64, eltype(u), typeof(coefficient))
        return new{T}(Vector{T}(u), T(coefficient))
    end
end

"""
    MassVelocityOperator(masses; coefficient=1)

Coefficient of `α²` in the leading mass–velocity correction, in atomic units:
`-coefficient * Σᵢ pᵢ⁴/(8mᵢ³)`. `masses` fixes the same Jacobi frame as the
nonrelativistic Hamiltonian. It includes the recoil momentum of every particle.

Evaluate perturbatively with [`build_operator_matrix`](@ref) on a
nonrelativistic solution; the negative fourth-order term is unbounded below.
No fine-structure constant or unit conversion is included. Masses define a
fixed frame; width, shift, and coupling derivatives remain supported.
"""
struct MassVelocityOperator{T <: Real} <: Operator
    terms::Vector{MomentumFourthOperator{T}}
end

function MassVelocityOperator(masses::AbstractVector{<:Real}; coefficient::Real = 1)
    _check_coefficient(coefficient)
    m, J, _ = _checked_jacobi(masses)
    return MassVelocityOperator(
        [MomentumFourthOperator(J[:, i]; coefficient = -coefficient / (8m[i]^3)) for i in eachindex(m)]
    )
end

"""
    OrbitOrbitOperator(w, u, v; coefficient=1)
    OrbitOrbitOperator(masses, charges; coefficient=1)

Orbit–orbit tensor `coefficient * Σₐᵦ pᵤₐ (δₐᵦ/r + rₐrᵦ/r³) pᵥᵦ`, where
`r = w' * x`, `pᵤ = -im * u' * ∇ₓ`, and likewise for `v`. This ordering is
Hermitian: the Coulomb tensor has zero divergence. The raw constructor supplies
one pair; its weights must all have the same Jacobi dimension.

The physical constructor sums all particle pairs with coefficient
`-qᵢqⱼ/(2mᵢmⱼ)`, using individual particle momenta in the COM rest frame.
It returns the coefficient of `α²` in atomic units, for perturbative evaluation
on a nonrelativistic state. Masses define a fixed Jacobi frame. No spin terms,
fine-structure constant, or conversion factors are included.

Rank0 elements use analytic displaced Gaussian moments, smooth at zero shift;
all higher-rank and parameter derivatives compose through ForwardDiff.
See Stanke et al., Phys. Rev. A **97**, 012513 (2018), Eq. (29).
"""
# One orbit–orbit pair: separation weight `w`, momentum weights `u`, `v`.
struct _OrbitOrbitPair{T <: Real}
    w::Vector{T}
    u::Vector{T}
    v::Vector{T}
    coefficient::T
end

struct OrbitOrbitOperator{T <: Real} <: Operator
    pairs::Vector{_OrbitOrbitPair{T}}
end

function OrbitOrbitOperator(
        w::AbstractVector{<:Real}, u::AbstractVector{<:Real}, v::AbstractVector{<:Real}; coefficient::Real = 1
    )
    length(w) == length(u) == length(v) || throw(DimensionMismatch("separation and momentum weights must have the same dimension"))
    _check_weight(w, "separation weights")
    _check_weight(u, "momentum weights"; nonzero = false)
    _check_weight(v, "momentum weights"; nonzero = false)
    _check_coefficient(coefficient)
    T = promote_type(Float64, eltype(w), eltype(u), eltype(v), typeof(coefficient))
    return OrbitOrbitOperator([_OrbitOrbitPair{T}(w, u, v, coefficient)])
end

function OrbitOrbitOperator(masses::AbstractVector{<:Real}, charges::AbstractVector{<:Real}; coefficient::Real = 1)
    Base.require_one_based_indexing(charges)
    length(masses) == length(charges) || throw(DimensionMismatch("each particle needs one charge"))
    all(isfinite, charges) || throw(ArgumentError("charges must be finite"))
    _check_coefficient(coefficient)
    m, J, U = _checked_jacobi(masses)
    T = promote_type(Float64, eltype(charges), typeof(coefficient))
    # Convert before multiplying: products of small low-precision charges underflow.
    q, c = collect(T, charges), T(coefficient)
    pairs = [
        _OrbitOrbitPair{T}(U[i, :] - U[j, :], J[:, i], J[:, j], -c * q[i] * q[j] / (2m[i] * m[j]))
            for i in eachindex(m) for j in (i + 1):lastindex(m)
    ]
    all(p -> isfinite(p.coefficient), pairs) || throw(ArgumentError("orbit–orbit pair coefficients must be finite"))
    return OrbitOrbitOperator(pairs)
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::MomentumFourthOperator)
    size(bra.A) == size(ket.A) == (length(op.u), length(op.u)) || throw(DimensionMismatch("momentum weights must match Gaussian dimensions"))
    A, B, u = bra.A, ket.A, op.u
    R, t, M = _updated_rank0_data(bra, ket)
    μ = R * t / 2
    Au, Bu = A * u, B * u
    l = vec(u' * bra.s - 2Au' * μ)
    r = vec(u' * ket.s - 2Bu' * μ)
    # A - A*R*A = A*R*B avoids subtracting nearly equal large widths.
    covariance = 2dot(Au, R * Bu)
    left = 3covariance - sum(abs2, l)
    right = 3covariance - sum(abs2, r)
    return op.coefficient * (left * right + 4covariance * dot(l, r) + 6covariance^2) * M
end

_compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::MassVelocityOperator) =
    sum(_compute_matrix_element(bra, ket, term) for term in op.terms)

# E[|r|] = h(β|μ|²)/√β for an isotropic density (β/π)^(3/2)exp(-β|r-μ|²).
# Polynomial derivatives avoid removable singularities under nested shift AD.
# COV_EXCL_START: precomputed during precompilation
const _DISTANCE_SERIES = ntuple(21) do j
    j == 1 ? 2 / sqrt(π) : 2 / sqrt(π) * (-1)^(j - 2) / factorial(j - 1) / (2j - 3) / (2j - 1)
end
const _DISTANCE_FIRST = ntuple(j -> j * _DISTANCE_SERIES[j + 1], 20)
const _DISTANCE_SECOND = ntuple(j -> j * (j + 1) * _DISTANCE_SERIES[j + 2], 19)
# COV_EXCL_STOP

function _distance_moments(z)
    if z < 1
        return evalpoly(z, _DISTANCE_SERIES), evalpoly(z, _DISTANCE_FIRST), evalpoly(z, _DISTANCE_SECOND)
    end
    f = _erf_ratio_from_square(z)
    e = exp(-z) / sqrt(π)
    fp = (2e - f) / (2z)
    fpp = -e / z - 3fp / (2z)
    return (z + 0.5) * f + e, f + (z + 0.5) * fp - e, 2fp + (z + 0.5) * fpp + e
end

function _orbit_orbit_element(bra::Rank0Gaussian, ket::Rank0Gaussian, w, u, v)
    size(bra.A) == size(ket.A) == (length(w), length(w)) || throw(DimensionMismatch("orbit–orbit weights must match Gaussian dimensions"))
    A, B = bra.A, ket.A
    R, t, M = _updated_rank0_data(bra, ket)
    μx = R * t / 2
    Rw = R * w
    β = inv(dot(w, Rw))
    μ = vec(w' * μx)
    Au, Bv = A * u, B * v
    l = vec(u' * bra.s - 2Au' * μx)
    r = vec(v' * ket.s - 2Bv' * μx)
    ql, qr = dot(Au, Rw), dot(Bv, Rw)
    a, b = -2β * ql, -2β * qr
    l0, r0 = l - a * μ, r - b * μ
    covariance = 2dot(Au, R * Bv) - 2β * ql * qr
    z = β * sum(abs2, μ)
    h, hp, hpp = _distance_moments(z)
    rootβ = sqrt(β)
    F = rootβ * _erf_ratio_from_square(z)
    lμ, rμ = dot(l0, μ), dot(r0, μ)
    moment = 2 * (F - rootβ * hp) * dot(l0, r0) - 4β * rootβ * hpp * lμ * rμ +
        4rootβ * hp * (b * lμ + a * rμ) + 2a * b * h / rootβ + 4covariance * F
    return moment * M
end

_compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::OrbitOrbitOperator) =
    sum(p.coefficient * _orbit_orbit_element(bra, ket, p.w, p.u, p.v) for p in op.pairs)
