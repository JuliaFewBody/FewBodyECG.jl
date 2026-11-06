# # Three-body Gaussian model with a many-body regulator
#
# A heavy center binds two identical light particles through pairwise Gaussian
# attractions. A repulsive many-body Gaussian `exp(-rᵀWr)` acts as a regulator
# that raises the energy while staying compatible with the rank-0 stochastic
# solver.

using FewBodyECG
using FewBodyECG: jacobi_transform
using Plots

masses = [1.0e15, 1.0, 1.0]
attraction = Kinetic(masses) - 4.0 * Gaussian(1, 2; exponent = 0.5) - 4.0 * Gaussian(1, 3; exponent = 0.5)

## `W` is given in Jacobi coordinates r = Jx; in particle coordinates the
## exponent is xᵀ(JᵀWJ)x.
W = [0.6 0.1; 0.1 0.6]
λ = 3.0
J, _ = jacobi_transform(masses)
regulated = attraction + ManyBodyGaussian(J' * W * J; coefficient = λ)

sol₀ = solve(attraction, SVM(basis = 30, candidates = 20))
sol₁ = solve(regulated, SVM(basis = 30, candidates = 20))
@info "Regulator shift (Ha)" E_without = sol₀.E₀ E_with = sol₁.E₀ ΔE = sol₁.E₀ - sol₀.E₀

p = plot(convergence(sol₁); xlabel = "step", ylabel = "E (Ha)", label = "with regulator")
display(plot!(p, convergence(sol₀); label = "no regulator"))
plot(wavefunction(sol₁); coord = 1, rmax = 6.0)
