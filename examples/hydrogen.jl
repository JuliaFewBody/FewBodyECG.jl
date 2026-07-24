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

sol₂s = solve(
    ops, GrowVariational(basis = 15, candidates = 10, scale = 1.0, maxiter_step = 40);
    state = 2,
)
println("2s energy: ", sol₂s.E₀, " Ha  (Antique ", exact₂, ", Δ = ", sol₂s.E₀ - exact₂, ")")
sol₂s

αs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
basis₁ = BasisSet([Rank1Gaussian([α;;], [1.0], [0.0]) for α in αs])
E₁, c₁ = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis₁, ops),
    build_overlap_matrix(basis₁),
)
E₂p = minimum(E₁)
println("2p energy: ", E₂p, " Ha  (Antique ", exact₂, ", Δ = ", E₂p - exact₂, ")")

a = reshape([1.0, 0.0, 0.0], 1, 3)
b = reshape([0.0, 1.0, 0.0], 1, 3)
αd = exp10.(range(log10(0.002), log10(0.8), length = 24))
basis₂ = BasisSet([Rank2Gaussian([α;;], a, b, [0.0]) for α in αd])
E₂, c₂ = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis₂, ops),
    build_overlap_matrix(basis₂),
)
E₃d = minimum(E₂)
println("3d energy: ", E₃d, " Ha  (Antique ", exact₃, ", Δ = ", E₃d - exact₃, ")")

ψs = (
    wavefunction(sol),
    wavefunction(sol₂s),
    Wavefunction(basis₁, c₁[:, 1]),
    Wavefunction(basis₂, c₂[:, 1]),
)
states = (("1s", 1, 0), ("2s", 2, 0), ("2p", 2, 1), ("3d", 3, 2))
rs = range(0.0, 12.0, length = 400)

function antique_density(n, l)
    density = [r^2 * abs2(Antique.ψ(H, r, 0.0, 0.0; n = n, l = l, m = 0)) for r in rs]
    area = sum((density[i] + density[i + 1]) * (rs[i + 1] - rs[i]) / 2 for i in 1:(length(rs) - 1))
    return density ./ area
end

p = plot(; xlabel = "r (Jacobi coordinate, mass-weighted)", ylabel = "normalized r²|ψ(r)|²", legend = :topright)
for (i, (ψ, (label, n, l))) in enumerate(zip(ψs, states))
    r, density = radial_profile(ψ; rmax = 12.0, npoints = 400)
    plot!(p, r, density; color = i, label = "ECG $label", linewidth = 2)
    plot!(p, rs, antique_density(n, l); color = i, label = "Antique $label", linestyle = :dash)
end
p
