"""
    FewBodyECG.jacobi_transform(masses) -> (J, U)

Compute the Jacobi coordinate transformation matrix `J` and its pseudo-inverse `U`
for a system with the given particle `masses`.

Returns `(J, U)` where:
- `J` is the ``(N-1) \\times N`` matrix mapping particle coordinates to Jacobi
  relative coordinates (centre-of-mass motion is factored out).
- `U = \\operatorname{pinv}(J)` is the ``N \\times (N-1)`` back-transformation.

`masses` may be any one-based vector of reals; the computation and the returned
`J` and `U` are in `Float64`.
The weight vectors for `CoulombOperator` are constructed as `U' * charge_vector`.
"""
function jacobi_transform(masses::AbstractVector{<:Real})
    Base.require_one_based_indexing(masses)
    m = collect(Float64, masses)
    N = length(m)
    N ≥ 2 || throw(ArgumentError("at least two masses are required for a Jacobi transformation, got $N"))
    J = zeros(Float64, N - 1, N)

    for k in 1:(N - 1)
        mk = m[k + 1]
        Mk = sum(m[1:k])
        μk = sqrt(mk * Mk / (mk + Mk))
        for j in 1:N
            if j ≤ k
                J[k, j] = μk * m[j] / Mk
            elseif j == k + 1
                J[k, j] = -μk
            else
                J[k, j] = 0.0
            end
        end
    end

    U = pinv(J)
    return J, U
end

# `(m, J, U)` for operators built from particle masses: validated `Float64`
# masses and their Jacobi transformation.
function _checked_jacobi(masses::AbstractVector{<:Real})
    Base.require_one_based_indexing(masses)
    length(masses) ≥ 2 ||
        throw(ArgumentError("at least two particle masses are required, got $(length(masses))"))
    all(m -> isfinite(m) && m > 0, masses) ||
        throw(ArgumentError("masses must be finite and positive, got $masses"))
    m = collect(Float64, masses)
    J, U = jacobi_transform(m)
    all(isfinite, J) && all(isfinite, U) ||
        throw(ArgumentError("masses $masses do not give a finite Jacobi transformation"))
    return m, J, U
end

"""
    FewBodyECG.Λ(masses) -> Symmetric matrix

Compute the kinetic-energy matrix in Jacobi coordinates for a system with the
given particle `masses` (in atomic units).

Returns the symmetric matrix ``\\Lambda = J M^{-1} J^T / 2``, where ``J`` is
the Jacobi transformation matrix and ``M = \\operatorname{diag}(m_i)``.
Pass the result directly to `KineticOperator`.
"""
function Λ(masses::AbstractVector{<:Real})
    m = collect(Float64, masses)
    J, _ = jacobi_transform(m)
    Minv = Diagonal(0.5 ./ m)
    Λ = Symmetric(J * Minv * J')
    return Λ
end

function KineticOperator(masses::AbstractVector{<:Real})
    return KineticOperator(Λ(masses))
end

function _transform_coordinates(J::AbstractMatrix, r::AbstractVector)
    axes(J, 2) == axes(r, 1) || throw(
        DimensionMismatch("columns of J $(axes(J, 2)) must match the axes of r $(axes(r, 1))")
    )
    return J * r
end

function _inverse_transform_coordinates(U::AbstractMatrix, x::AbstractVector)
    axes(U, 2) == axes(x, 1) || throw(
        DimensionMismatch("columns of U $(axes(U, 2)) must match the axes of x $(axes(x, 1))")
    )
    return U * x
end
