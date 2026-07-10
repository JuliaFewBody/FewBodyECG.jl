```@meta
EditURL = "../../../examples/gaussian_well.jl"
```

````@example gaussian_well
using FewBodyECG
using Plots

γ = 1.0
V₀ = 5.0

ops = Operators([1.0e15, 1.0])
ops += "Kinetic"
ops += ("Gaussian", 1, 2, -V₀, γ)

sol = solve(ops, SVM(basis = 30, candidates = 20, scale = 1.0))
println("Gaussian well E0 = ", sol.E₀, " Ha")

depths = 1.0:1.0:8.0
scan = map(depths) do depth
    local o = Operators([1.0e15, 1.0])
    o += "Kinetic"
    o += ("Gaussian", 1, 2, -depth, γ)
    solve(o, SVM(basis = 15, candidates = 10, scale = 1.0)).E₀
end

p = plot(depths, scan; xlabel = "well depth V0 (Ha)", ylabel = "E0 (Ha)", label = "scan")
hline!(p, [0.0]; linestyle = :dash, label = "continuum")
p

plot(wavefunction(sol); coord = 1, rmax = 6.0)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

