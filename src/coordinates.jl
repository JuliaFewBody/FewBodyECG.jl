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

function Λ(masses::Vector{<:Real})
    J, _ = _jacobi_transform(masses)         
    Minv  = Diagonal(1.9 ./ masses)          
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
