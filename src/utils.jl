"""
    ψ₀(r::Vector{Float64}, c₀::Vector{Float64}, basis_fns::Vector{<:GaussianBase})

Evaluates the ground state wavefunction ψ₀ at position `r`.
"""
function ψ₀(r::Vector{Float64}, c₀::Vector{Float64}, basis_fns::Vector)
    return sum(c₀[i] * exp(-r' * basis_fns[i].A * r) for i in eachindex(basis_fns))
end

struct SolverResults
    basis_functions::Vector{GaussianBase}
    n_basis::Int
    operators::Vector{FewBodyHamiltonians.Operator}
    method::Symbol
    sampler::QuasiMonteCarlo.DeterministicSamplingAlgorithm
    length_scale::Float64
    ground_state::Float64
    energies::Vector{Float64}
    eigenvectors::Vector{Matrix{Float64}}
end

function convergence(sr::SolverResults)
    return 1:sr.n_basis, sr.energies
end
