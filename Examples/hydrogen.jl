# # Hydrogen: s-, p- and d-waves
#
# Exact non-relativistic hydrogen energies are -1/2, -1/8 and -1/18 Ha for
# the lowest s, p and d states.

using FewBodyECG
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0))
sol

plot(sol, -0.5)

# ## p- and d-waves
#
# Rank-1 and Rank-2 prefactors are built manually and solved through the
# power-user matrix layer.
αs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
basis₁ = BasisSet([Rank1Gaussian([α;;], [1.0], [0.0]) for α in αs])
E₁, _ = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis₁, ops),
    build_overlap_matrix(basis₁),
)
println("2p energy: ", minimum(E₁), "  (exact -0.125)")

a = reshape([1.0, 0.0, 0.0], 1, 3)
b = reshape([0.0, 1.0, 0.0], 1, 3)
αd = exp10.(range(log10(0.002), log10(0.8), length = 24))
basis₂ = BasisSet([Rank2Gaussian([α;;], a, b, [0.0]) for α in αd])
E₂, _ = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis₂, ops),
    build_overlap_matrix(basis₂),
)
println("3d energy: ", minimum(E₂), "  (exact -1/18 = ", -1 / 18, ")")

plot(wavefunction(sol); coord = 1, rmax = 12.0)
