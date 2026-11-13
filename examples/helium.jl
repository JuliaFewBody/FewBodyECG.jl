# Helium and the hydride ion H⁻ with an infinitely heavy nucleus.
using FewBodyECG
using FewBodyDB: db
using Plots

helium = solve(Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2.0, -1.0, -1.0]), GVM(basis = 50))
helium_ref = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value
@info "He" E₀ = helium.E₀ reference = helium_ref Δ = helium.E₀ - helium_ref

hydride = solve(Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+1.0, -1.0, -1.0]), GVM(basis = 50))
hydride_ref = db(:Suzuki2003Jul, "∞H⁻", :energy, "¹Sᵉ").value
@info "H⁻" E₀ = hydride.E₀ reference = hydride_ref Δ = hydride.E₀ - hydride_ref

display(plot(helium, helium_ref; title = "He"))
plot(wavefunction(hydride); coord = 1, rmax = 10.0)
