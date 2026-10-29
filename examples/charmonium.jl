using FewBodyECG
using OptimKit: LBFGS
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

γ = 1 / r₀^2
coefficient = 2 * π * κκ / 3 / mass[1] / mass[2] / (sqrt(π) * r₀)^3 * σσ
ops = Kinetic(mass; units = UnitSystem("GeV", "GeV⁻¹")) +   # ħ = c = 1
    Coulomb(1, 2; coefficient = -κ) +
    Custom(r -> λ * r, 1, 2) +
    Gaussian(1, 2; coefficient, exponent = γ)

sol = solve(ops, GVM(basis = 5, optimizer = LBFGS(; maxiter = 100)))

rest_energy = sum(mass)
E_total = sol.E₀ + rest_energy - Λ
println("Charmonium ground state: E = ", E_total, " GeV")

plot(wavefunction(sol); coord = 1, rmax = 5.0)
plot(convergence(sol))
