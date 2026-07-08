```@meta
EditURL = "../../../examples/positronium.jl"
```

# Positronium

Positronium is the two-body electron-positron Coulomb problem.  With equal
masses the exact ground-state energy is -0.25 Ha.

````@example positronium
using FewBodyECG
using Plots

ops = Operators([1.0, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.4))
sol

println("E0 = ", sol.E₀, " Ha  (exact -0.25)")
plot(sol, -0.25)

ψ = wavefunction(sol)
plot(ψ; coord = 1, rmax = 15.0)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

