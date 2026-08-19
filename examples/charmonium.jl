using FewBodyECG
using Plots

mass = [1.836, 1.836]
σσ = -3
Λ = 0.8321
λ = 0.1653
κ = 0.5069
κκ = 1.8609
A = 1.6553
B = 0.2204
r₀ = A * (2 * mass[1] * mass[2] / (mass[1] + mass[2]))^(-B)

ops = Operators(mass)
ops += "Kinetic"
ops += ("Coulomb", 1, 2, -κ)
ops += (r -> λ * r, numerical, 1, 2)
γ = 1 / r₀^2
coefficient = 2 * π * κκ / 3 / mass[1] / mass[2] / (sqrt(π) * r₀)^3 * σσ
ops += ("Gaussian", 1, 2, coefficient, γ)
ops

sol = solve(ops, SVM(150))

rest_energy = sum(mass)
E_total = sol.E₀ + rest_energy - Λ
println("Charmonium ground state: E = ", E_total, " GeV")
println("  (relative-coordinate eigenvalue ", sol.E₀, " GeV, close to the physical ηc(1S) mass of 2.984 GeV)")
sol

plot(wavefunction(sol); coord = 1, rmax = 5.0)
plot(convergence(sol))