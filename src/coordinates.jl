"""
    _jacobi_transform(masses) -> (J, U)

Compute the Jacobi coordinate transformation matrix `J` and its pseudo-inverse `U`
for a system with the given particle `masses`.

Returns `(J, U)` where:
- `J` is the ``(N-1) \\times N`` matrix mapping particle coordinates to Jacobi
  relative coordinates (centre-of-mass motion is factored out).
- `U = \\operatorname{pinv}(J)` is the ``N \\times (N-1)`` back-transformation.

The weight vectors for `CoulombOperator` are constructed as `U' * charge_vector`.
"""
function _jacobi_transform(masses::Vector{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}
    N = length(masses)
    @assert N ≥ 2 "At least two masses are required for Jacobi transformation."
    J = zeros(Float64, N - 1, N)

    for k in 1:(N - 1)
        mk = masses[k + 1]
        Mk = sum(masses[1:k])
        μk = sqrt(mk * Mk / (mk + Mk))
        for j in 1:N
            if j ≤ k
                J[k, j] = μk * masses[j] / Mk
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

"""
    Λ(masses) -> Symmetric matrix

Compute the kinetic-energy matrix in Jacobi coordinates for a system with the
given particle `masses` (in atomic units).

Returns the symmetric matrix ``\\Lambda = J M^{-1} J^T / 2``, where ``J`` is
the Jacobi transformation matrix and ``M = \\operatorname{diag}(m_i)``.
Pass the result directly to `KineticOperator`.
"""
function Λ(masses::Vector{<:Real})
    J, _ = _jacobi_transform(masses)
    Minv = Diagonal(0.5 ./ masses)
    Λ = Symmetric(J * Minv * J')
    return Λ
end

function KineticOperator(masses::Vector{<:Real})
    return KineticOperator(Λ(masses))
end

function _transform_coordinates(J::Matrix{Float64}, r::Vector{Float64})::Vector{Float64}
    @assert size(J, 2) == length(r) "Matrix `J` columns must match length of vector `r`."
    return J * r
end

function _inverse_transform_coordinates(U::Matrix{Float64}, x::Vector{Float64})::Vector{Float64}
    @assert size(U, 2) == length(x) "Matrix `U` columns must match length of vector `x`."
    return U * x
end
