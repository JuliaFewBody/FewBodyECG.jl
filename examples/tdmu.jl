using FewBodyECG
using FewBodyDB: db
using Plots

ops = Kinetic([5496.918, 3670.481, 206.7686]) + Coulomb([+1.0, +1.0, -1.0])

sol = solve(
    ops,
    SVM(basis = 200, candidates = 25);
    tol = 1.0e-3,
    window = 10,
)
sol

tdmu_ref = db(:Suzuki2003Jul, "tdμ", :energy, "¹Sᵉ").value
println("tdμ E0 = ", sol.E₀, " Ha  (Suzuki–Varga ", tdmu_ref, ", Δ = ", sol.E₀ - tdmu_ref, ")")
plot(convergence(sol); xlabel = "step", ylabel = "E (Ha)", label = "ECG")
hline!([tdmu_ref]; label = "reference", linestyle = :dash)

ψ = wavefunction(sol)
plot(ψ; coord = 1, rmax = 2, npoints = 300)
