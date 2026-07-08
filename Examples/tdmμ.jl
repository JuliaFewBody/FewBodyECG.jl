using FewBodyECG
using Plots

ops = Operators([5496.918, 3670.481, 206.7686], [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(
    ops,
    SVM(basis = 40, candidates = 25, scale = 0.03);
    tol = 1.0e-2,
    window = 10,
)
sol

tdmu_ref = -111.36444
println("tdmu E0 = ", sol.E₀, " Ha  (reference ", tdmu_ref, ", Δ = ", sol.E₀ - tdmu_ref, ")")
plot(sol, tdmu_ref)

ψ = wavefunction(sol)
plot(ψ; coord = 1, rmax = 2, npoints = 300)
