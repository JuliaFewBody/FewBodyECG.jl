# One particle in an attractive Gaussian well, and the ground-state energy as
# the well deepens.
using FewBodyECG
using Plots

well(depth) = Kinetic([1.0e15, 1.0]) - depth * Gaussian(1, 2; exponent = 1.0)

sol = solve(well(5.0), SVM(basis = 30, candidates = 20))
@info "Gaussian well, V₀ = 5 Ha" E₀ = sol.E₀

depths = 1.0:1.0:8.0
energies = [solve(well(depth), SVM(basis = 15, candidates = 10)).E₀ for depth in depths]

p = plot(depths, energies; label = "ground state", marker = :circle, xlabel = "well depth V₀ (Ha)", ylabel = "E₀ (Ha)")
hline!(p, [0.0]; linestyle = :dash, label = "continuum")
display(p)
plot(wavefunction(sol); coord = 1, rmax = 6.0)
