using FewBodyECG
using LinearAlgebra
using Plots

masses = [1.0e15, 1.0]

Λmat = Λ(masses)
kin = KineticOperator(Λmat)

J, U = _jacobi_transform(masses)

w_raw = [U' * [1, -1]]
coeffs = [-1.0]

ops = Operator[
    kin;
    (CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw))...
]

# Polarization vector for p-wave (selects one spatial direction)
a_vec = [1.0]
s_zero = [0.0]

# Use a range of Gaussian widths that spans the spatial extent of the 2p orbital.
# The 2p state is more diffuse than 1s, so we need wider Gaussians (smaller α).
alphas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

basis_fns = GaussianBase[]
E₀_list = Float64[]

for (i, α) in enumerate(alphas)
    push!(basis_fns, Rank1Gaussian([α;;], a_vec, s_zero))

    basis = BasisSet(basis_fns)

    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)

    global vals, vecs = solve_generalized_eigenproblem(H, S)
    E₀ = minimum(vals)

    push!(E₀_list, E₀)
    println("Step $i (α=$α): E₀ = $E₀")
end

E_exact = -0.125
E_min = minimum(E₀_list)
println("\nBest energy: $E_min")
println("Exact:       $E_exact")
@show ΔE = abs(E_min - E_exact)

# Plot convergence
p1 = plot(1:length(E₀_list), E₀_list,
    xlabel = "Basis size", ylabel = "E₀ (Hartree)",
    label = "p-wave energy", lw = 2, marker = :circle)
hline!([E_exact], label = "Exact (-0.125)", ls = :dash, color = :red)
title!("Hydrogen 2p convergence")

display(p1)

# Plot the radial correlation function r²|ψ(r)|²
# For Rank1Gaussians: ψ(r) = Σ cᵢ (aᵢ'r) exp(-r'Aᵢr)
c₀ = vecs[:, 1]
function ψ_p(r_vec, coeffs, bfs)
    return sum(
        coeffs[i] * dot(bfs[i].a, r_vec) * exp(-r_vec' * bfs[i].A * r_vec)
            for i in eachindex(bfs)
    )
end

r_grid = range(0.01, 15.0, length = 400)
ρ_r = [rval^2 * abs2(ψ_p([rval], c₀, basis_fns)) for rval in r_grid]

# Normalize
dr = step(r_grid)
ρ_r ./= sum(ρ_r) * dr

p2 = plot(r_grid, ρ_r,
    xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²",
    label = "2p correlation", lw = 2)
title!("Hydrogen 2p radial correlation function")

display(p2)
