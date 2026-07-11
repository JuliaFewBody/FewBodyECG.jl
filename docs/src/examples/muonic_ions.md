```@meta
EditURL = "../../../examples/muonic_ions.jl"
```

# Muonic molecular ions (dtμ, ttμ)

Compare the ECG results with both the Suzuki–Varga K = 200 SVM values and
the higher-basis values listed in Table 8.1.

````@example muonic_ions
using FewBodyECG

mμ, md, mt = 206.7686, 3670.481, 5496.918

systems = [
    (
        "dtμ",
        [md, mt, mμ],
        -111.36444,       # Suzuki–Varga SVM, K = 200
        -111.364511474,   # other method, K = 1400
    ),
    (
        "ttμ",
        [mt, mt, mμ],
        -112.97300,       # Suzuki–Varga SVM, K = 200
        -112.9730179,     # other method, K = 500
    ),
]

for (name, masses, svm200, best_ref) in systems
    ops = Operators(masses, [+1.0, +1.0, -1.0])
    ops += "Kinetic"
    ops += "Coulomb"

    sol = solve(
        ops,
        SVM(basis = 200, candidates = 40, scale = 0.02);
        tol = 1.0e-4,
        window = 15,
    )

    println(name)
    println("  ECG E₀              = ", sol.E₀, " Ha")
    println("  Δ vs SVM K=200      = ", sol.E₀ - svm200, " Ha")
    println("  Δ vs high-K result  = ", sol.E₀ - best_ref, " Ha")
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

