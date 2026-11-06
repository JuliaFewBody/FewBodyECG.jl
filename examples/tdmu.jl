# The muonic molecular ion tdμ.
using FewBodyECG
using FewBodyDB: db
using Plots

ops = Kinetic([5496.918, 3670.481, 206.7686]) + Coulomb([+1.0, +1.0, -1.0])
sol = solve(ops, SVM(basis = 200, candidates = 25); tol = 1.0e-3, window = 10)

reference = db(:Suzuki2003Jul, "tdμ", :energy, "¹Sᵉ").value
@info "tdμ" E₀ = sol.E₀ reference Δ = sol.E₀ - reference

display(plot(sol, reference))
plot(wavefunction(sol); coord = 1, rmax = 2, npoints = 300)
