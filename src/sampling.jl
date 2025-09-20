using QuasiMonteCarlo

export generate_bij

"""
    generate_bij(method::Symbol, i::Int, n_terms::Int, b1::Float64) -> Vector{Float64}

Generate a bij vector using the specified sampling method.
"""
function generate_bij(method::Symbol, i::Int, n_terms::Int, b1::Float64; qmc_sampler = HaltonSample())
    if method == :quasirandom
        return QuasiMonteCarlo.sample(i + 1, n_terms, qmc_sampler)[:, end] * b1
    elseif method == :random
        return rand(n_terms) * b1
    else
        error("Unsupported sampling method: $method")
    end
end
