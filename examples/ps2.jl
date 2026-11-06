# Dipositronium Ps₂ (positrons 1, 3; electrons 2, 4) with the exchange
# symmetry of the ground state, and its unlike-pair contact density.
using FewBodyECG
using Plots

masses, charges = ones(4), [+1.0, -1.0, +1.0, -1.0]
symmetry = PermutationSymmetry(masses, [[3, 2, 1, 4], [1, 4, 3, 2], [2, 1, 4, 3]])
sol = solve(Kinetic(masses) + Coulomb(charges), SVM(basis = 150, candidates = 30); tol = 1.0e-4, window = 15, symmetry)

reference = -0.516003778   # SVM, K = 800
@info "Ps₂" E₀ = sol.E₀ reference Δ = sol.E₀ - reference

# ⟨δ³(rᵢ - rⱼ)⟩ per unlike pair, in a₀⁻³ (a density per volume).  Suzuki–Varga,
# Table 8.5, p. 157, gives 0.0221151; contact densities converge more slowly than
# energies, so a larger basis is needed for more digits.
_, U = FewBodyECG.jacobi_transform(masses)
pair_weight(i, j) = U[i, :] - U[j, :]
contacts = [expectation(sol, ContactDensityOperator(pair_weight(i, j))) for (i, j) in ((1, 2), (1, 4), (3, 2), (3, 4))]
@info "Ps₂ unlike-pair contact (a₀⁻³)" contacts mean = sum(contacts) / length(contacts) reference = 0.0221151

display(plot(sol, reference; title = "Dipositronium convergence"))
plot(wavefunction(sol))
