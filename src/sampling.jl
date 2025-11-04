using QuasiMonteCarlo

export generate_bij, generate_shift, _generate_A_matrix, build_rank0

function _qmc_point(i::Int, d::Int; sampler = HaltonSample())
    QuasiMonteCarlo.sample(i + 1, d, sampler)[:, end]
end

function generate_bij(method::Symbol, i::Int, n_terms::Int, b1::Float64;
                      qmc_sampler = HaltonSample(),
                      bmin = 0.02*b1, bmax = 20b1)
    bmin > 0 || throw(ArgumentError("bmin must be > 0"))
    bmax > bmin || throw(ArgumentError("bmax must be > bmin"))
    u = method === :quasirandom ? _qmc_point(i, n_terms; sampler=qmc_sampler) :
        method === :random      ? rand(n_terms) :
        error("Unsupported method $method")
    bmin .* (bmax / bmin) .^ u
end

# symmetric shifts in [-sscale, +sscale]
function generate_shift(method::Symbol, i::Int, dim::Int, sscale::Real;
                        qmc_sampler = HaltonSample())
    u = method === :quasirandom ? _qmc_point(i, dim; sampler=qmc_sampler) :
        method === :random      ? rand(dim) :
        error("Unsupported method $method")
    sscale .* (2u .- 1)
end

# A = W * Diag(1/b.^2) * W'
function _generate_A_matrix(bij::AbstractVector{<:Real},
                            w_list::AbstractVector{<:AbstractVector{<:Real}})
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
    A
end

function build_rank0(bij::AbstractVector{<:Real},
                     w_list::AbstractVector{<:AbstractVector{<:Real}},
                     s::AbstractVector{<:Real})
    A = _generate_A_matrix(bij, w_list)
    Rank0Gaussian(A, Float64.(s))
end