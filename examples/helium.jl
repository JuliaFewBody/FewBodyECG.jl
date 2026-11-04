using FewBodyECG
using FewBodyDB: db
using Plots

helium = Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2.0, -1.0, -1.0])

he_ref = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value
he = solve(helium, SVM(basis = 50, candidates = 25))
println("Helium E0 = ", he.E₀, " Ha  (Suzuki–Varga ", he_ref, ", Δ = ", he.E₀ - he_ref, ")")

hminus = Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+1.0, -1.0, -1.0])

hm_ref = db(:Suzuki2003Jul, "∞H⁻", :energy, "¹Sᵉ").value
hm = solve(hminus, SVM(basis = 50, candidates = 20))
println("H- E0 = ", hm.E₀, " Ha  (Suzuki–Varga ", hm_ref, ", Δ = ", hm.E₀ - hm_ref, ")")

plot(convergence(he); xlabel = "step", ylabel = "E (Ha)", label = "ECG")
hline!([he_ref]; label = "reference", linestyle = :dash)
plot(wavefunction(hm); coord = 1, rmax = 10.0)
