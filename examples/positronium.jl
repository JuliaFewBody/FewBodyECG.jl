# Positronium, and the positronium negative ion Ps⁻ with the electron-exchange
# symmetry imposed.
using FewBodyECG
import Antique
using Plots

ops = Kinetic([1.0, 1.0]) + Coulomb([+1.0, -1.0])
sol = solve(ops, SVM(basis = 25, candidates = 20))

ps = Antique.CoulombTwoBody(z₁ = 1, z₂ = -1, m₁ = 1.0, m₂ = 1.0, mₑ = 1.0, a₀ = 1.0, Eₕ = 1.0, ħ = 1.0)
exact = Antique.energy(ps, n = 1)
@info "Ps" E₀ = sol.E₀ exact Δ = sol.E₀ - exact
display(plot(sol, exact; title = "Ps"))

# The Jacobi coordinate is the relative distance scaled by √μ, μ = 1/2.
μ = 1 / 2
rs = range(1.0e-3, 15.0, length = 400)
exact_ψ(r) = μ^(-3 / 4) * Antique.wavefunction(ps, r / sqrt(μ), 0.0, 0.0; n = 1, l = 0, m = 0)
exact_density = [r^2 * abs2(exact_ψ(r)) for r in rs]
p = plot(wavefunction(sol); coord = 1, rmax = 15.0, label = "ECG")
display(plot!(p, rs, exact_density; linestyle = :dash, label = "exact"))

# Ps⁻: electrons 2 and 3 form a spin singlet, so their spatial exchange is even.
# Suzuki–Varga §4.2, p. 43; reference energy from Table 8.3, p. 154 (K = 600).
masses = [1.0, 1.0, 1.0]
exchange = [1, 3, 2]
psm = solve(
    Kinetic(masses) + Coulomb([+1.0, -1.0, -1.0]), SVM(basis = 100, candidates = 25);
    symmetry = PermutationSymmetry(masses, [exchange])
)
psm_ref = -0.262005070226
exchange_parity = expectation(psm, PermutationOperator(masses, exchange))
@info "Ps⁻" E₀ = psm.E₀ reference = psm_ref exchange_parity
plot(psm, psm_ref; title = "Ps⁻ with electron-exchange symmetry")
