using FewBodyECG
using Plots

n_gaussians = 150
masses = [1.0, 1.0, 1.0, 1.0]
charges = [+1.0, -1.0, +1.0, -1.0]

ops = Kinetic(masses) + Coulomb(charges)
symmetry = PermutationSymmetry(masses, [[3, 2, 1, 4], [1, 4, 3, 2], [2, 1, 4, 3]])

ps2_ref = -0.516003778
sol = solve(
    ops,
    SVM(basis = n_gaussians, candidates = 30);
    tol = 1.0e-4,
    window = 15,
    symmetry,
)

println("Ps₂ E₀ = ", sol.E₀, " Ha")
println("reference (SVM, K = 800) = ", ps2_ref, " Ha   Δ = ", sol.E₀ - ps2_ref)

plot(convergence(sol); xlabel = "step", ylabel = "E (Ha)", label = "ECG", title = "Dipositronium convergence")
hline!([ps2_ref]; label = "reference", linestyle = :dash)
plot(wavefunction(sol))

# Contact density: Suzuki–Varga, Table 8.5, p. 157, in atomic units.
# Positrons are 1,3 and electrons are 2,4.
# δ³(rᵢ-rⱼ) is a density per volume, not a radial delta function.
using LinearAlgebra: dot

function pair_contact(basis, coefficients, w)
    S = FewBodyECG.build_overlap_matrix(basis)
    D = FewBodyECG.build_operator_matrix(basis, ContactDensityOperator(w))
    return real(dot(coefficients, D * coefficients) / dot(coefficients, S * coefficients))
end

_, U = FewBodyECG.jacobi_transform(masses)
c = sol.coefficients[:, sol.state]
unlike_pairs = [(1, 2), (1, 4), (3, 2), (3, 4)]
contacts = map(unlike_pairs) do (i, j)
    e = zeros(length(masses))
    e[i], e[j] = 1.0, -1.0
    pair_contact(sol.basis, c, U' * e)
end
contact_mean = sum(contacts) / length(contacts)
contact_ref = 0.0221151  # Per unlike pair, a₀⁻³; Table 8.5, L = 0.

@assert all(>(0), contacts) && all(isfinite, contacts)
@assert isapprox(contact_mean, contact_ref; rtol = 0.1)
println("mean unlike-pair contact = ", contact_mean, " a₀⁻³  (book ", contact_ref, ")")
# Contact densities converge more slowly than energies; increase the basis
# to resolve more digits of the tabulated contact density.
