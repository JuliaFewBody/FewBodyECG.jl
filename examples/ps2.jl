using FewBodyECG
using Plots

masses = [1.0, 1.0, 1.0, 1.0]
charges = [+1.0, -1.0, +1.0, -1.0]

ops = Operators(masses, charges)
ops += "Kinetic"
ops += "Coulomb"

ps2_ref = -0.516003778
sol = solve(
    ops,
    SVM(basis = 150, candidates = 30, scale = 1.0);
    tol = 1.0e-4,
    window = 15,
)

println("Ps₂ E₀ = ", sol.E₀, " Ha")
println("reference (SVM, K = 800) = ", ps2_ref, " Ha   Δ = ", sol.E₀ - ps2_ref)
println("variational upper bound respected: ", sol.E₀ ≥ ps2_ref)

plot(sol, ps2_ref; title = "Dipositronium convergence")
plot(wavefunction(sol))
