using FewBodyECG
import Antique
using Plots

ops = Kinetic([1.0, 1.0]) + Coulomb([+1.0, -1.0])

sol = solve(ops, SVM(basis = 25, candidates = 20))
sol

ps = Antique.CoulombTwoBody(
    z₁ = 1, z₂ = -1, m₁ = 1.0, m₂ = 1.0, mₑ = 1.0, a₀ = 1.0, Eₕ = 1.0, ħ = 1.0
)
exact = Antique.energy(ps, n = 1)
println("E0 = ", sol.E₀, " Ha  (Antique ", exact, ", Δ = ", sol.E₀ - exact, ")")
plot(convergence(sol); xlabel = "step", ylabel = "E (Ha)", label = "ECG")
hline!([exact]; label = "reference", linestyle = :dash)

ψ = wavefunction(sol)
μ = inv(1 / 1.0 + 1 / 1.0)
rs = range(1.0e-3, 15.0, length = 400)
p = plot(ψ; coord = 1, rmax = 15.0)
plot!(
    p, rs,
    [r^2 * abs2(μ^(-3 / 4) * Antique.wavefunction(ps, r / sqrt(μ), 0.0, 0.0; n = 1, l = 0, m = 0)) for r in rs];
    linestyle = :dash,
    label = "Antique.jl",
)
p

# Ps⁻: electrons 2,3 form a spin singlet, so their spatial exchange is even.
# Suzuki–Varga §4.2, p. 43 and Table 8.3, p. 154.
using LinearAlgebra: dot

psm_masses = [1.0, 1.0, 1.0]
psm_ops = Kinetic(psm_masses) + Coulomb([+1.0, -1.0, -1.0])

psm_ref = -0.262005070226  # Table 8.3, p. 154; K = 600, atomic units.
symmetry = PermutationSymmetry(psm_masses, [[1, 3, 2]])
psm = solve(psm_ops, SVM(basis = 100, candidates = 25); symmetry)
basis = psm.basis
P23 = PermutationOperator(psm_masses, [1, 3, 2])
S = FewBodyECG.build_overlap_matrix(basis)
MP = FewBodyECG.build_operator_matrix(basis, P23)
c = psm.coefficients[:, 1]
psm_exchange_parity = real(dot(c, MP * c) / dot(c, S * c))
println("Ps⁻ E = ", energy(psm), " Ha; ⟨P₂₃⟩ = ", psm_exchange_parity)
p_psm = plot(convergence(psm); xlabel = "step", ylabel = "E (Ha)", label = "ECG", title = "Ps⁻ electron-exchange projection")
hline!(p_psm, [psm_ref]; label = "reference", linestyle = :dash)
