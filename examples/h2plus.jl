# # H₂⁺ without Born–Oppenheimer
#
# The dihydrogen cation is solved as a direct proton–proton–electron Coulomb
# problem. The non-Born–Oppenheimer reference energy is about −0.597139 Ha;
# being below −0.5 Ha means it is bound against H + p⁺ dissociation.

using FewBodyECG
using Plots

mₚ = 1836.15267343
ops = Kinetic([mₚ, mₚ, 1.0]) + Coulomb([+1.0, +1.0, -1.0])
sol = solve(ops, SVM(basis = 40, candidates = 25) → Refine(sweeps = 2, candidates = 25))

reference = -0.597139
@info "H₂⁺" E₀ = sol.E₀ reference Δ = sol.E₀ - reference bound = sol.E₀ < -0.5

display(plot(sol, reference))
plot(wavefunction(sol); coord = 1, rmax = 100.0)
