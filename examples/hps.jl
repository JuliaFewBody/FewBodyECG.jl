# Positronium hydride HPs: a proton, two electrons and a positron.
using FewBodyECG
using Plots

ops = Kinetic([1.0e15, 1.0, 1.0, 1.0]) + Coulomb([+1.0, -1.0, -1.0, +1.0])
sol = solve(ops, SVM(basis = 50))

reference = -0.7891964   # SVM, K = 1200
@info "HPs" E₀ = sol.E₀ reference Δ = sol.E₀ - reference

display(plot(sol, reference; title = "Positronium-hydride convergence"))
plot(wavefunction(sol))
