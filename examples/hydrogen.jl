# Hydrogen 1s and 2s from the gradient solver, 2p and 3d from hand-built
# polarized bases, compared with the exact radial densities.
using FewBodyECG
using FewBodyECG: build_hamiltonian_matrix, build_overlap_matrix, solve_generalized_eigenproblem
import Antique
using OptimKit: LBFGS
using Plots

ops = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])
atom = Antique.HydrogenAtom(Z = 1)
exact(n) = Antique.energy(atom; n)

s1 = solve(ops, DynamicGVM(basis = 10, candidates = 20))
s2 = solve(ops, DynamicGVM(basis = 15, candidates = 10, optimizer = LBFGS(; maxiter = 40, gradtol = 1.0e-6)); state = 2)
display(plot(s1, exact(1); title = "Hydrogen 1s"))

# Lowest state of a fixed basis: E and its wavefunction.
function lowest(basis)
    E, c = solve_generalized_eigenproblem(build_hamiltonian_matrix(basis, ops), build_overlap_matrix(basis))
    return first(E), Wavefunction(basis, c[:, 1])
end

# 2p: z-polarized Rank1 Gaussians.  3d: xy-polarized Rank2 Gaussians, pure L = 2.
p_widths = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
d_widths = exp10.(range(log10(0.002), log10(0.8), length = 24))
E2p, ψ2p = lowest(BasisSet([Rank1Gaussian([α;;], [1.0], [0.0]) for α in p_widths]))
E3d, ψ3d = lowest(BasisSet([Rank2Gaussian([α;;], [1 0 0], [0 1 0], [0.0]) for α in d_widths]))

@info "Hydrogen (Ha)" E1s = s1.E₀ E2s = s2.E₀ E2p E3d exact_n1 = exact(1) exact_n2 = exact(2) exact_n3 = exact(3)

# Normalized radial densities; the xy-polarized 3d state vanishes on the z axis,
# so it is sampled along (1, 1, 0).
rs = range(0.0, 12.0, length = 400)
function exact_density(n, l)
    density = [r^2 * abs2(Antique.wavefunction(atom, r, 0.0, 0.0; n, l, m = 0)) for r in rs]
    return density ./ sum((density[1:(end - 1)] + density[2:end]) .* diff(rs) / 2)
end
states = [
    ("1s", wavefunction(s1), 1, 0, (0, 0, 1)), ("2s", wavefunction(s2), 2, 0, (0, 0, 1)),
    ("2p", ψ2p, 2, 1, (0, 0, 1)), ("3d", ψ3d, 3, 2, (1, 1, 0)),
]
p = plot(; xlabel = "r (a₀)", ylabel = "normalized r²|ψ(r)|²")
for (i, (label, ψ, n, l, direction)) in enumerate(states)
    r, density = radial_profile(ψ; direction, rmax = 12.0, npoints = 400)
    plot!(p, r, density; color = i, label = "ECG $label", linewidth = 2)
    plot!(p, rs, exact_density(n, l); color = i, label = "exact $label", linestyle = :dash)
end
p
