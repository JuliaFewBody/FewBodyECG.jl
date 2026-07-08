# # Hydrogen: s-, p- and d-waves
#
# Exact non-relativistic hydrogen energies are -1/2, -1/8 and -1/18 Ha for
# the lowest s, p and d states.

using FewBodyECG
import Antique
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

H = Antique.HydrogenAtom(Z = 1)
exact₁ = Antique.E(H, n = 1)
exact₂ = Antique.E(H, n = 2)
exact₃ = Antique.E(H, n = 3)

sol = solve(ops, GrowVariational(basis = 10, candidates = 20, scale = 1.0))
println("1s energy: ", sol.E₀, " Ha  (Antique ", exact₁, ", Δ = ", sol.E₀ - exact₁, ")")
sol

plot(sol, exact₁)

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
E₂p = minimum(E₁)
println("2p energy: ", E₂p, " Ha  (Antique ", exact₂, ", Δ = ", E₂p - exact₂, ")")

a = reshape([1.0, 0.0, 0.0], 1, 3)
b = reshape([0.0, 1.0, 0.0], 1, 3)
αd = exp10.(range(log10(0.002), log10(0.8), length = 24))
basis₂ = BasisSet([Rank2Gaussian([α;;], a, b, [0.0]) for α in αd])
E₂, _ = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis₂, ops),
    build_overlap_matrix(basis₂),
)
E₃d = minimum(E₂)
println("3d energy: ", E₃d, " Ha  (Antique ", exact₃, ", Δ = ", E₃d - exact₃, ")")

ψ = wavefunction(sol)
rs = range(1.0e-3, 12.0, length = 400)
p = plot(ψ; coord = 1, rmax = 12.0)
plot!(p, rs, [r^2 * abs2(Antique.ψ(H, r, 0.0, 0.0; n = 1, l = 0, m = 0)) for r in rs]; linestyle = :dash, label = "Antique.jl")
p
