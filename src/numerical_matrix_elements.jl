using QuadGK: quadgk

# Maclaurin coefficients of sinh(√x)/√x in x: 1/(2k+1)!, k = 0…9.
const _SINHC_SERIES = ntuple(k -> 1 / factorial(2k - 1), 10)

# exp(−β(r² + μ²)) · sinh(x)/x with x = 2βrμ, taking μ² rather than μ: the norm
# of the three-dimensional mean displacement is singular at zero for dual
# numbers.  Below x² = 1 the power series in x² is used; it is smooth at μ² = 0,
# so derivatives with respect to the shifts stay exact at zero shift.  Above it
# the product is evaluated as a difference of two Gaussians, because sinh(x)
# overflows while exp(−β(r² + μ²)) underflows at the large r that the
# quadrature samples.
function _shifted_radial_kernel(r, β, μ2)
    x2 = 4 * β^2 * r^2 * μ2
    x2 < 1 && return exp(-β * (r^2 + μ2)) * evalpoly(x2, _SINHC_SERIES)
    μ = sqrt(μ2)
    return (exp(-β * (r - μ)^2) - exp(-β * (r + μ)^2)) / (4 * β * r * μ)
end

# Radial density of |wᵀr| for an isotropic 3D Gaussian with inverse variance 2β
# per Cartesian component and squared mean μ².
_numerical_radial_density(r, β, μ2) = 4π * (β / π)^(3 / 2) * r^2 * _shifted_radial_kernel(r, β, μ2)

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
