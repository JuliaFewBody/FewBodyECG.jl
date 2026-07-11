```@meta
EditURL = "../../../examples/three_body_gaussian.jl"
```

# Three-body Gaussian model with a many-body regulator

A heavy centre binds two identical light particles through pairwise Gaussian
attractions.  A repulsive many-body Gaussian `exp(-rᵀWr)` acts as a regulator
that lifts the energy — a physically useful interaction that stays compatible
with the rank-0 stochastic solver.

````@example three_body_gaussian
using FewBodyECG
using Plots

masses = [1.0e15, 1.0, 1.0]

build() = begin
    o = Operators(masses)
    o += "Kinetic"
    o += ("Gaussian", 1, 2, -4.0, 0.5)   # centre–particle-2 attraction
    o += ("Gaussian", 1, 3, -4.0, 0.5)   # centre–particle-3 attraction
    o
end

# Without the regulator (λ = 0)
ops₀ = build()
sol₀ = solve(ops₀, SVM(basis = 30, candidates = 20, scale = 1.0))

# With a repulsive many-body Gaussian regulator (λ > 0, W positive-definite)
W = [0.6 0.1; 0.1 0.6]
λ = 3.0
ops₁ = build()
ops₁ += ManyBodyGaussianOperator(λ, W)
sol₁ = solve(ops₁, SVM(basis = 30, candidates = 20, scale = 1.0))

ΔE = sol₁.E₀ - sol₀.E₀
println("E₀ (λ = 0)  = ", sol₀.E₀, " Ha")
println("E₀ (λ = ", λ, ") = ", sol₁.E₀, " Ha")
println("regulator-induced shift ΔE = ", ΔE, " Ha")

# Convergence and radial profile through the utilities
steps, history = convergence(sol₁)
pc = plot(steps, history; xlabel = "step", ylabel = "E (Ha)", label = "with regulator")
plot!(pc, convergence(sol₀)...; label = "no regulator")
pc

plot(wavefunction(sol₁); coord = 1, rmax = 6.0)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

