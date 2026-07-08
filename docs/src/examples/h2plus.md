```@meta
EditURL = "../../../examples/h2plus.jl"
```

# H2+ without Born-Oppenheimer

The dihydrogen cation is solved as a direct proton-proton-electron Coulomb
problem.  The non-Born-Oppenheimer reference energy is about -0.597139 Ha;
being below -0.5 Ha means it is bound against H + p+ dissociation.

````@example h2plus
using FewBodyECG
using Plots

mₚ = 1836.15267343
ops = Operators([mₚ, mₚ, 1.0], [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(
    ops,
    SVM(basis = 40, candidates = 25, scale = 1.0) →
        Refine(sweeps = 2, candidates = 25, scale = 1.0),
)
sol

println("H2+ E0 = ", sol.E₀, " Ha  (reference -0.597139)")
println("bound below H + p+ threshold? ", sol.E₀ < -0.5)

plot(sol, -0.597139)
plot(wavefunction(sol); coord = 1, rmax = 80.0)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

