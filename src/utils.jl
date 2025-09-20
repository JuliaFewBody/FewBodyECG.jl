"""
    ψ₀(r::Vector{Float64}, c₀::Vector{Float64}, basis_fns::Vector{<:GaussianBase})

Evaluates the ground state wavefunction ψ₀ at position `r`.
"""
function ψ₀(r::Vector{Float64}, c₀::Vector{Float64}, basis_fns::Vector)
    return sum(c₀[i] * exp(-r' * basis_fns[i].A * r) for i in eachindex(basis_fns))
end
