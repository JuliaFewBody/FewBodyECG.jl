# # Ps₂ excited state and electric-dipole decay
# Suzuki–Varga, Tables 8.4–8.5 and Eqs. (8.2)–(8.5), pp. 155–158.
# [Usukura, Varga and Suzuki (1998)](https://arxiv.org/abs/physics/9804023).
# Improved E1 benchmark: [Puchalski–Czarnecki (2008)](https://arxiv.org/abs/0810.0013).
# The book's B(E1) ≈ 0.87 e²a₀² was revised to ≈ 0.994 in 2008.

using FewBodyECG
using LinearAlgebra
using Plots

n_gaussians = 100
masses, charges = ones(4), [1.0, -1.0, 1.0, -1.0]
H = Kinetic(masses) + Coulomb(charges)
generators = [[3, 2, 1, 4], [1, 4, 3, 2], [2, 1, 4, 3]]
even = PermutationSymmetry(masses, generators)
odd = PermutationSymmetry(masses, generators; signs = [1, 1, -1])
method(rank) = SVM(n_gaussians; rank, candidates = 30) → Refine(2; candidates = 15)
s = solve(H, method(0); symmetry = even)
p = solve(H, method(1); symmetry = odd)

cs, cp = s.coefficients[:, 1], p.coefficients[:, 1]
M = FewBodyECG.build_operator_matrix(s.basis, p.basis, DipoleOperator(masses, charges))
dipole = abs(dot(cs, M * cp))
photon_energy = energy(p) - energy(s)
B_E1 = 3 * dipole^2 / (4π)  # Spherical-harmonic convention of the book.

# Table 8.5: each unlike pair has the same contact, in a₀⁻³.
# Contacts converge more slowly than energies; increase n_gaussians for accuracy.
_, U = FewBodyECG.jacobi_transform(masses)
contact_op = ContactDensityOperator(U' * [1.0, -1.0, 0.0, 0.0])
contacts = map((s, p)) do sol
    c = sol.coefficients[:, 1]
    D = FewBodyECG.build_operator_matrix(sol.basis, contact_op)
    real(dot(c, D * c))
end
# The common α³ / tₐ factor cancels between the leading decay rates.
branching = photon_energy^3 * dipole^2 / (photon_energy^3 * dipole^2 + 3π * contacts[2])

println("Ps₂: ΔE = ", round(photon_energy; digits = 6), " Ha; B(E1) = ", round(B_E1; digits = 3), " e²a₀²")
println("⟨δ³(r₁₂)⟩ = ", round.(contacts; digits = 6), " a₀⁻³ (0⁺, 1⁻); E1 branch = ", round(100branching; digits = 1), "%")

style = (;
    framestyle = :box, gridalpha = 0.12, linewidth = 2, legendfontsize = 9,
    titlefontsize = 12, guidefontsize = 10,
    left_margin = 6Plots.mm, bottom_margin = 7Plots.mm,
)
energy_plot = plot(convergence(s); label = "0⁺, C=+1", legend = :right, xlabel = "step", ylabel = "E (Ha)", title = "Ps₂ convergence", style...)
plot!(energy_plot, convergence(p); label = "1⁻, C=−1")
hline!(energy_plot, [-0.516003778, -0.334408177]; label = "book, K=800", color = :gray, linestyle = :dot)
hline!(energy_plot, [-0.3125]; label = "Ps(1s)+Ps(2p)", color = :black, linestyle = :dash)
contact_plot = bar(
    [0.85, 1.85], collect(contacts);
    label = "ECG", bar_width = 0.3, linealpha = 0,
    xticks = ([1, 2], ["0⁺", "1⁻"]),
    ylabel = "⟨δ³(r₁₂)⟩ (a₀⁻³)", ylims = (0, 0.025), title = "Unlike-pair contact", style...
)
bar!(contact_plot, [1.15, 2.15], [0.0221151, 0.0112091]; label = "book, Table 8.5", bar_width = 0.3, linealpha = 0)
physics_plot = plot(energy_plot, contact_plot; layout = (1, 2), size = (1000, 400), dpi = 180)
physics_plot
