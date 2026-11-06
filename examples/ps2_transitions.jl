# # Ps₂ excited state and electric-dipole decay
#
# Suzuki–Varga, Tables 8.4–8.5 and Eqs. (8.2)–(8.5), pp. 155–158;
# [Usukura, Varga and Suzuki (1998)](https://arxiv.org/abs/physics/9804023).
# The book's B(E1) ≈ 0.87 e²a₀² was revised to ≈ 0.994 by
# [Puchalski and Czarnecki (2008)](https://arxiv.org/abs/0810.0013).

using FewBodyECG
using FewBodyECG: build_operator_matrix, jacobi_transform
using LinearAlgebra
using Plots

masses, charges = ones(4), [1.0, -1.0, 1.0, -1.0]
H = Kinetic(masses) + Coulomb(charges)

## Ground state 0⁺ (C = +1) and the P-wave state 1⁻ (C = −1).
generators = [[3, 2, 1, 4], [1, 4, 3, 2], [2, 1, 4, 3]]
method(rank) = SVM(100; rank, candidates = 30) → Refine(2; candidates = 15)
s = solve(H, method(0); symmetry = PermutationSymmetry(masses, generators))
p = solve(H, method(1); symmetry = PermutationSymmetry(masses, generators; signs = [1, 1, -1]))

M = build_operator_matrix(s.basis, p.basis, DipoleOperator(masses, charges))
dipole = abs(dot(s.coefficients[:, 1], M * p.coefficients[:, 1]))
photon_energy = energy(p) - energy(s)
B_E1 = 3 * dipole^2 / (4π)   # spherical-harmonic convention of the book

## Unlike-pair contact densities (Table 8.5, in a₀⁻³); they converge more slowly
## than the energies.  The common α³ factor cancels in the branching ratio.
_, U = jacobi_transform(masses)
contact = ContactDensityOperator(U' * [1.0, -1.0, 0.0, 0.0])
contacts = [expectation(sol, contact) for sol in (s, p)]
branching = photon_energy^3 * dipole^2 / (photon_energy^3 * dipole^2 + 3π * contacts[2])
@info "Ps₂ 1⁻ → 0⁺" photon_energy B_E1 contacts E1_branching = branching

energy_plot = plot(convergence(s); label = "0⁺", xlabel = "step", ylabel = "E (Ha)", title = "Ps₂ convergence")
plot!(energy_plot, convergence(p); label = "1⁻")
hline!(energy_plot, [-0.516003778, -0.334408177]; label = "book, K = 800", color = :gray, linestyle = :dot)
hline!(energy_plot, [-0.3125]; label = "Ps(1s) + Ps(2p)", color = :black, linestyle = :dash)

contact_plot = bar(
    [0.85, 1.85], contacts; bar_width = 0.3, label = "ECG",
    xticks = ([1, 2], ["0⁺", "1⁻"]), ylabel = "⟨δ³(r₁₂)⟩ (a₀⁻³)", title = "Unlike-pair contact",
)
bar!(contact_plot, [1.15, 2.15], [0.0221151, 0.0112091]; bar_width = 0.3, label = "book, Table 8.5")
plot(energy_plot, contact_plot; layout = (1, 2), size = (1000, 400))
