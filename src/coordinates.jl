
struct ParticleSystem
    masses::Vector{Float64}
    J::Matrix{Float64}
    U::Matrix{Float64}
    scale::Union{Symbol, Nothing}  # :atomic, :molecular, :nuclear, etc.

    function ParticleSystem(masses::Vector{Float64}; scale::Union{Symbol, Nothing} = nothing)
        @assert length(masses) ≥ 2 "At least two masses are required for a particle system."
        J, U = _jacobi_transform(masses)
        return new(masses, J, U, scale)
    end
end

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

function Λ(masses::Vector{<:Number})
    J, _ = _jacobi_transform(masses)         
    Minv  = Diagonal(0.5 ./ masses)          
    Λ = Symmetric(J * Minv * J')             
    return Λ
end

function KineticOperator(masses::Vector{<:Number})
    return KineticOperator(Λ(masses))
end

"""
    default_b0(scale::Union{Symbol,Nothing}) -> Float64

Returns a default value for the parameter `b0` based on the provided `scale`.

# Arguments
- `scale::Union{Symbol,Nothing}`: A symbol representing the scale type or `nothing`.
    - `:atomic`: Returns `1.0`, corresponding to the Bohr radius in atomic units.
    - `:molecular`: Returns `3.0`, representing a typical molecular bond length.
    - `:nuclear`: Returns `0.03`, approximately 1 femtometer in atomic units.
    - `nothing`: Returns `1.0` as a fallback default.
"""
function default_b0(scale::Union{Symbol, Nothing})
    scale === :atomic     && return 1.0
    scale === :molecular  && return 3.0
    scale === :nuclear    && return 0.03
    scale === nothing     && return 10.0
    error("Unknown scale: $scale")
end

function _generate_A_matrix(bij::AbstractVector{<:Number}, w_list::AbstractVector{<:AbstractVector{<:Number}})::Matrix{Float64}
    bijf = Float64.(bij)
    w_listf = [Float64.(w) for w in w_list]
    dim = length(w_listf[1])
    A = zeros(Float64, dim, dim)
    for i in 1:length(bijf)
        A .+= (w_listf[i] * w_listf[i]') / (bijf[i]^2)
    end
    return A
end

function _shift_vectors(a::Matrix{Float64}, b::Matrix{Float64}, mat::Union{Nothing, Matrix{Float64}} = nothing)::Float64
    n = size(a, 2)
    @assert n == size(b, 2) "Matrices `a` and `b` must have the same number of columns."
    mat = mat === nothing ? I(n) : mat
    @assert size(mat) == (n, n) "Matrix `mat` must be square with size equal to number of vectors."

    sum_val = 0.0
    for i in 1:n
        for j in 1:n
            sum_val += mat[i, j] * dot(view(a, :, i), view(b, :, j))
        end
    end
    return sum_val
end

function _transform_coordinates(J::Matrix{Float64}, r::Vector{Float64})::Vector{Float64}
    @assert size(J, 2) == length(r) "Matrix `J` columns must match length of vector `r`."
    return J * r
end

function _inverse_transform_coordinates(U::Matrix{Float64}, x::Vector{Float64})::Vector{Float64}
    @assert size(U, 2) == length(x) "Matrix `U` columns must match length of vector `x`."
    return U * x
end
