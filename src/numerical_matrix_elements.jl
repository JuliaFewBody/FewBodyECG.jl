using QuadGK: quadgk

# sinh(x) / x evaluated from x².  The squared argument avoids taking a norm of
# the three-dimensional displacement, which is singular at zero for dual
# numbers used by the variational gradients.
function _sinhc_from_square(x2)
    if x2 < 1.0e-8
        return 1 + x2 / 6 + x2^2 / 120
    end
    x = sqrt(x2)
    return sinh(x) / x
end

function _numerical_radial_density(r, β, μ2)
    if iszero(μ2)
        return 4 * β^(3 / 2) / sqrt(π) * r^2 * exp(-β * r^2)
    end

    x2 = 4 * β^2 * r^2 * μ2
    return 4π * (β / π)^(3 / 2) * r^2 *
        exp(-β * (r^2 + μ2)) * _sinhc_from_square(x2)
end

function _radial_prefactor_coefficients(bra::Rank1Gaussian, ket::Rank1Gaussian, R, u)
    Ru, ρ = R * u, dot(u, R * u)
    c = 0.5 * (
        _polar_contract(bra.a, R, ket.a) -
            _polar_project_dot(bra.a, Ru, ket.a, Ru) / ρ
    )
    m = _polar_project_dot(bra.a, Ru, ket.a, Ru) / ρ^2
    return (c, m / 3, zero(c))
end

function _radial_prefactor_coefficients(bra::Rank2Gaussian, ket::Rank2Gaussian, R, u)
    x, y, z, t = bra.a, bra.b, ket.a, ket.b
    Ru, ρ = R * u, dot(u, R * u)
    C(v, w) = 0.5 * (
        _polar_contract(v, R, w) -
            _polar_project_dot(v, Ru, w, Ru) / ρ
    )
    M(v, w) = _polar_project_dot(v, Ru, w, Ru) / ρ^2

    Cxy, Cxz, Cxt = C(x, y), C(x, z), C(x, t)
    Cyz, Cyt, Czt = C(y, z), C(y, t), C(z, t)
    Mxy, Mxz, Mxt = M(x, y), M(x, z), M(x, t)
    Myz, Myt, Mzt = M(y, z), M(y, t), M(z, t)

    m0 = Cxy * Czt + Cxz * Cyt + Cxt * Cyz
    m2 = (
        Cxy * Mzt + Cxz * Myt + Cxt * Myz +
            Cyz * Mxt + Cyt * Mxz + Czt * Mxy
    ) / 3
    m4 = (Mxy * Mzt + Mxz * Myt + Mxt * Myz) / 15
    return (m0, m2, m4)
end

function _compute_prefactor_numerical_matrix_element(bra, ket, op)
    if any(!iszero, bra.s) || any(!iszero, ket.s)
        throw(ArgumentError("prefactor NumericalPotential matrix elements require zero shifts"))
    end

    A, B = parent(bra.A), parent(ket.A)
    n = size(A, 1)
    length(op.w) == n || throw(
        DimensionMismatch("NumericalPotential weight length must equal Gaussian dimension")
    )

    R = inv(A + B)
    ρ = dot(op.w, R * op.w)
    ρ > 0 || throw(ArgumentError("NumericalPotential weight has nonpositive Gaussian variance"))
    m0, m2, m4 = _radial_prefactor_coefficients(bra, ket, R, op.w)
    M0 = (π^n / det(A + B))^(3 / 2)

    density(r) = 4 / sqrt(π) * r^2 / ρ^(3 / 2) * exp(-r^2 / ρ)
    integrand(r) = begin
        value = op.f(r)
        value isa Number || throw(
            ArgumentError("NumericalPotential callable must return a numeric value")
        )
        isfinite(value) || throw(DomainError(value, "NumericalPotential returned a non-finite value"))
        density(r) * value * (m0 + m2 * r^2 + m4 * r^4)
    end

    integral, _ = quadgk(
        integrand,
        0.0,
        Inf;
        rtol = op.rtol,
        atol = op.atol,
        maxevals = op.maxevals,
    )
    isfinite(integral) || throw(DomainError(integral, "NumericalPotential quadrature returned a non-finite value"))
    result = M0 * integral
    isfinite(result) || throw(DomainError(result, "NumericalPotential matrix element is non-finite"))
    return result
end

for G in (Rank1Gaussian, Rank2Gaussian)
    @eval _compute_matrix_element(bra::$G, ket::$G, op::NumericalPotential) =
        _compute_prefactor_numerical_matrix_element(bra, ket, op)
end

function _compute_matrix_element(
        bra::Rank0Gaussian,
        ket::Rank0Gaussian,
        op::NumericalPotential
    )
    A, B = parent(bra.A), parent(ket.A)
    n = size(A, 1)
    length(op.w) == n || throw(
        DimensionMismatch("NumericalPotential weight length must equal Gaussian dimension")
    )

    S = A + B
    R = inv(S)
    σ2 = dot(op.w, R * op.w)
    σ2 > 0 || throw(ArgumentError("NumericalPotential weight has nonpositive Gaussian variance"))
    β = inv(σ2)

    v = bra.s + ket.s
    μ = 0.5 * (transpose(op.w) * R * v)
    μ2 = sum(abs2, μ)
    M = _compute_matrix_element(bra, ket)

    integrand = r -> begin
        value = op.f(r)
        value isa Number || throw(
            ArgumentError("NumericalPotential callable must return a numeric value")
        )
        isfinite(value) || throw(DomainError(value, "NumericalPotential returned a non-finite value"))
        _numerical_radial_density(r, β, μ2) * value
    end

    integral, _ = quadgk(
        integrand,
        0.0,
        Inf;
        rtol = op.rtol,
        atol = op.atol,
        maxevals = op.maxevals,
    )
    isfinite(integral) || throw(DomainError(integral, "NumericalPotential quadrature returned a non-finite value"))
    result = M * integral
    isfinite(result) || throw(DomainError(result, "NumericalPotential matrix element is non-finite"))
    return result
end
