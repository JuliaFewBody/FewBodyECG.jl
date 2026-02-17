using FewBodyECG
using LinearAlgebra
using Plots

import FewBodyECG: _generate_A_matrix

masses = [1.0e15, 1.0]
psys = ParticleSystem(masses)

K = Diagonal([0.0, 0.5])
K_transformed = psys.J * K * psys.J'

w_raw = [psys.U' * [1, -1]]
coeffs = [-1.0]

ops = Operator[
    KineticOperator(K_transformed);
    (CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw))...
]

# Polarization vector for p-wave (selects one spatial direction)
a_vec = [1.0]

# Use a range of Gaussian widths that spans the spatial extent of the 2p orbital.
# The 2p state is more diffuse than 1s, so we need wider Gaussians (smaller α).
alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

basis_fns = GaussianBase[]
E₀_list = Float64[]

for (i, α) in enumerate(alphas)
    push!(basis_fns, Rank1Gaussian([α;;], a_vec))

    basis = BasisSet(basis_fns)

    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)

    λs, Us = eigen(S)
    keep = λs .> 1.0e-10
    S⁻¹₂ = Us[:, keep] * Diagonal(1 ./ sqrt.(λs[keep])) * Us[:, keep]'
    H̃ = Symmetric(S⁻¹₂ * H * S⁻¹₂)
    E₀ = minimum(eigen(H̃).values)

    vals, vecs = solve_generalized_eigenproblem(H, S)
    global c₀ = vecs[:, 1]

    push!(E₀_list, E₀)
    println("Step $i (α=$α): E₀ = $E₀")
end

E_exact = -0.125
E_min = minimum(E₀_list)
println("\nBest energy: $E_min")
println("Exact:       $E_exact")
@show ΔE = abs(E_min - E_exact)
