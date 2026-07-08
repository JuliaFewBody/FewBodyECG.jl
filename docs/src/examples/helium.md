```@meta
EditURL = "../../../examples/helium.jl"
```

# Helium and H-

A fixed nucleus plus two electrons exercises all three Coulomb pairs:
nucleus-electron attraction and electron-electron repulsion.

````@example helium
using FewBodyECG
using Plots

helium = Operators([1.0e15, 1.0, 1.0], [+2.0, -1.0, -1.0])
helium += "Kinetic"
helium += "Coulomb"

he_ref = -2.9037
he = solve(helium, SVM(basis = 35, candidates = 25, scale = 1.0))
println("Helium E0 = ", he.E₀, " Ha  (reference ", he_ref, ", Δ = ", he.E₀ - he_ref, ")")

hminus = Operators([1.0e15, 1.0, 1.0], [+1.0, -1.0, -1.0])
hminus += "Kinetic"
hminus += "Coulomb"

hm_ref = -0.52775
hm = solve(hminus, SVM(basis = 30, candidates = 20, scale = 1.0))
println("H- E0 = ", hm.E₀, " Ha  (reference ", hm_ref, ", Δ = ", hm.E₀ - hm_ref, ")")

plot(he, he_ref)
plot(wavefunction(hm); coord = 1, rmax = 10.0)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

