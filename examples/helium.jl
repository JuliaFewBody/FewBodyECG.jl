using FewBodyECG
using FewBodyDB: db
using Plots

helium = Operators([1.0e15, 1.0, 1.0], [+2.0, -1.0, -1.0])
helium += "Kinetic"
helium += "Coulomb"

he_ref = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value
he = solve(helium, SVM(basis = 50, candidates = 25, scale = 1.0))
println("Helium E0 = ", he.E₀, " Ha  (Suzuki–Varga ", he_ref, ", Δ = ", he.E₀ - he_ref, ")")

hminus = Operators([1.0e15, 1.0, 1.0], [+1.0, -1.0, -1.0])
hminus += "Kinetic"
hminus += "Coulomb"

hm_ref = db(:Suzuki2003Jul, "∞H⁻", :energy, "¹Sᵉ").value
hm = solve(hminus, SVM(basis = 50, candidates = 20, scale = 4.0))
println("H- E0 = ", hm.E₀, " Ha  (Suzuki–Varga ", hm_ref, ", Δ = ", hm.E₀ - hm_ref, ")")

plot(he, he_ref)
plot(wavefunction(hm); coord = 1, rmax = 10.0)
