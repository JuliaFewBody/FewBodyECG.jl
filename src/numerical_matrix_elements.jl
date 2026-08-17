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
