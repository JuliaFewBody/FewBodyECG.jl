using QuasiMonteCarlo

export generate_bij, generate_shift, _generate_A_matrix, build_rank0

function _qmc_point(i::Int, d::Int; sampler = HaltonSample())
    QuasiMonteCarlo.sample(i + 1, d, sampler)[:, end]
end

function generate_bij(method::Symbol, i::Int, n_terms::Int, b1::Float64;
                      qmc_sampler=HaltonSample(), bmin=0.02*b1, bmax=20b1)
    u = method === :quasirandom ? QuasiMonteCarlo.sample(i + 1, n_terms, qmc_sampler)[:, end] :
        method === :random      ? rand(n_terms) :
        error("Unsupported method $method")
    bmin .* (bmax/bmin) .^ u
end

function generate_shift(method::Symbol, i::Int, dim::Int, sscale::Real;
                        qmc_sampler=HaltonSample())
    u = method === :quasirandom ? QuasiMonteCarlo.sample(i + 1, dim, qmc_sampler)[:, end] :
        method === :random      ? rand(dim) :
        error("Unsupported method $method")
    sscale .* (2u .- 1)  
end


function _generate_A_matrix(bij::AbstractVector{<:Real}, w_list::AbstractVector{<:AbstractVector{<:Real}})
    b = Float64.(bij)
    W = [Float64.(w) for w in w_list]
    length(b) == length(W) || throw(ArgumentError("Length of bij must equal number of w vectors"))
    d = length(W[1])
    A = zeros(Float64, d, d)
    for k in eachindex(b)
        A .+= (W[k] * W[k]') / (b[k]^2)
    end
    s = zeros(Float64, d)
    A, s
end
