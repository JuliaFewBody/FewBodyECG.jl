using QuasiMonteCarlo

export generate_bij, generate_shift, _generate_A_matrix, build_rank0

function _qmc_point(i::Int, d::Int; sampler = HaltonSample())
    return QuasiMonteCarlo.sample(i + 1, d, sampler)[:, end]
end

function generate_bij(method::Symbol, i::Int, n_terms::Int, b1::Float64; qmc_sampler = HaltonSample(), bmin = 0.02 * b1, bmax = 20b1)
    bmin > 0 || throw(ArgumentError("bmin must be > 0"))
    bmax > bmin || throw(ArgumentError("bmax must be > bmin"))
    u = method === :quasirandom ? _qmc_point(i, n_terms; sampler = qmc_sampler) :
        method === :random ? rand(n_terms) :
        error("Unsupported method $method")
    return bmin .* (bmax / bmin) .^ u
end

function generate_shift(method::Symbol, i::Int, dim::Int, scale::Real; qmc_sampler = HaltonSample())
    u = method === :quasirandom ? _qmc_point(i, dim; sampler = qmc_sampler) :
        method === :random ? rand(dim) :
        error("Unsupported method $method")
    return scale .* (2.0 .* u .- 1.0)
end

function _generate_A_matrix(bij::AbstractVector{<:Real}, w_list::AbstractVector{<:AbstractVector{<:Real}})
    b = Float64.(bij)
    m = length(b)
    m == length(w_list) || throw(ArgumentError("Length(bij) must equal number of w vectors"))

    d = length(w_list[1])
    @inbounds for k in 2:m
        length(w_list[k]) == d || throw(ArgumentError("All w vectors must have same length"))
    end

    W = Matrix{Float64}(undef, d, m)
    @inbounds for k in 1:m
        @views W[:, k] = Float64.(w_list[k])
    end

    A = W * Diagonal(1.0 ./ (b .^ 2)) * W'
    return A
end

function build_rank0(bij::AbstractVector{<:Real}, w_list::AbstractVector{<:AbstractVector{<:Real}}, s::AbstractVector{<:Real})
    A = _generate_A_matrix(bij, w_list)
    return Rank0Gaussian(A, Float64.(s))
end
