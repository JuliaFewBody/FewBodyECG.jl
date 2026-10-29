# # Three-body Gaussian model with a many-body regulator
#
# A heavy centre binds two identical light particles through pairwise Gaussian
# attractions.  A repulsive many-body Gaussian `exp(-rᵀWr)` acts as a regulator
# that lifts the energy — a physically useful interaction that stays compatible
# with the rank-0 stochastic solver.

using FewBodyECG
using Plots
using FewBodyECG: jacobi_transform

masses = [1.0e15, 1.0, 1.0]

## Without the regulator (λ = 0)
ops₀ = Kinetic(masses) -
    4.0 * Gaussian(1, 2; exponent = 0.5) -   # centre–particle-2 attraction
    4.0 * Gaussian(1, 3; exponent = 0.5)     # centre–particle-3 attraction
sol₀ = solve(ops₀, SVM(basis = 30, candidates = 20))

## With a repulsive many-body Gaussian regulator (λ > 0).  `W` is given in
## Jacobi coordinates r = Jx; in particle coordinates the exponent is xᵀ(JᵀWJ)x.
W = [0.6 0.1; 0.1 0.6]
λ = 3.0
J, _ = jacobi_transform(masses)
ops₁ = ops₀ + ManyBodyGaussian(J' * W * J; coefficient = λ)
sol₁ = solve(ops₁, SVM(basis = 30, candidates = 20))

ΔE = sol₁.E₀ - sol₀.E₀
println("E₀ (λ = 0)  = ", sol₀.E₀, " Ha")
println("E₀ (λ = ", λ, ") = ", sol₁.E₀, " Ha")
println("regulator-induced shift ΔE = ", ΔE, " Ha")

## Convergence and radial profile through the utilities
steps, history = convergence(sol₁)
pc = plot(steps, history; xlabel = "step", ylabel = "E (Ha)", label = "with regulator")
plot!(pc, convergence(sol₀)...; label = "no regulator")
pc

plot(wavefunction(sol₁); coord = 1, rmax = 6.0)
