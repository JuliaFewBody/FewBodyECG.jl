```@meta
EditURL = "../../../examples/workflow.jl"
```

# Solver comparison on hydrogen

Hydrogen has an analytical ground-state energy, so it is a compact benchmark
for comparing solver methods.

````@example workflow
using FewBodyECG
import Antique
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

exact = Antique.E(Antique.HydrogenAtom(Z = 1), n = 1)

function run_method(label, alg, ops, exact)
    sol = solve(ops, alg)
    println(label, ": E0 = ", sol.E₀, " Ha, Δ = ", sol.E₀ - exact)
    return sol
end

svm = run_method("SVM", SVM(basis = 25, candidates = 20, scale = 1.0), ops, exact)
refined = run_method(
    "SVM → Refine",
    SVM(basis = 25, candidates = 20, scale = 1.0) →
        Refine(sweeps = 1, candidates = 20, scale = 1.0),
    ops,
    exact,
)
variational = run_method("Variational", Variational(basis = 12, scale = 1.0, maxiter = 100), ops, exact)
grown = run_method("GrowVariational", GrowVariational(basis = 8, candidates = 20, scale = 1.0), ops, exact)

plot(grown, exact)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

