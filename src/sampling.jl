module Sampling

using ..Types
using ..Coordinates
using ..Hamiltonian
using LinearAlgebra
using QuasiMonteCarlo

export generate_basis, generate_bij


"""
generate_basis(widths::Vector{Matrix{Float64}}, rank::Int=0)

Construct a `ECGBasis` from a list of correlation matrices and optional rank.
"""
function generate_basis(widths::Vector{Matrix{Float64}}, rank::Int = 0)
    if rank == 0
        funcs = [Rank0Gaussian(A) for A in widths]
    else
        error("Only Rank0Gaussian implemented in generate_basis")
    end
    return ECGBasis(funcs)
end

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

end
