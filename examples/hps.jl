using FewBodyECG
using Plots

masses = [1.0e15, 1.0, 1.0, 1.0]
charges = [+1.0, -1.0, -1.0, +1.0]

ops = Kinetic(masses) + Coulomb(charges)

hps_ref = -0.7891964
sol = solve(ops, SVM(basis = 50))

println("HPs⁺ E₀ = ", sol.E₀, " Ha")
println("reference (SVM, K = 1200) = ", hps_ref, " Ha   Δ = ", sol.E₀ - hps_ref)
println("variational upper bound respected: ", sol.E₀ ≥ hps_ref)

plot(convergence(sol); xlabel = "step", ylabel = "E (Ha)", label = "ECG", title = "Positronium-hydride convergence")
hline!([hps_ref]; label = "reference", linestyle = :dash)
plot(wavefunction(sol))
