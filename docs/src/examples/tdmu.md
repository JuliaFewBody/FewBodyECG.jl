```@meta
EditURL = "../../../examples/tdmu.jl"
```

# tdmu muonic molecular ion

The tdmu ion is deeply bound because the muon is much heavier than an
electron.  A loose stochastic run is enough to show the energy scale; use a
tighter `tol` and larger basis for production values near -111.36444 Ha.

````@example tdmu
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

println("tdmu E0 = ", sol.E₀, " Ha  (reference -111.36444)")
plot(sol, -111.36444)

ψ = wavefunction(sol)
plot(ψ; coord = 1, rmax = 0.3, npoints = 300)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

