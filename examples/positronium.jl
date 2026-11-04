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
plot(sol, exact)

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

# Ps⁻: electron-exchange symmetry, Suzuki–Varga §4.2, p. 43, Eq. (4.2).
# Particle 1 is the positron; electrons 2 and 3 have a spin singlet, so the
# spatial basis is Π₊g = (g + P₂₃g)/2. The SVM supplies the Gaussian widths;
# the matrix-level solve below imposes this symmetry explicitly.
using LinearAlgebra: Diagonal, Symmetric, diag, dot, eigen

psm_masses = [1.0, 1.0, 1.0]
psm_ops = Kinetic(psm_masses) + Coulomb([+1.0, -1.0, -1.0])

psm_ref = -0.262005070226  # Table 8.3, p. 154; K = 600, atomic units.
psm = solve(psm_ops, SVM(basis = 100, candidates = 25))
basis = psm.basis
P23 = PermutationOperator(psm_masses, [1, 3, 2])
S = FewBodyECG.build_overlap_matrix(basis)
H = FewBodyECG.build_hamiltonian_matrix(basis, psm.operators)
MP = FewBodyECG.build_operator_matrix(basis, P23)
HP = sum(FewBodyECG.build_operator_matrix(basis, op * P23) for op in psm.operators)

# H commutes with P₂₃, hence Π₊HΠ₊ = HΠ₊ and Π₊² = Π₊.
S_plus, H_plus = (S + MP) / 2, (H + HP) / 2

# Normalize the projected functions and remove nearly dependent directions
# before whitening, so the solve needs no overlap regularization.
D = Diagonal(1 ./ sqrt.(diag(S_plus)))
F = eigen(Symmetric(D * S_plus * D))
keep = F.values .> 1.0e-10 * maximum(F.values)
X = D * F.vectors[:, keep] * Diagonal(1 ./ sqrt.(F.values[keep]))
F_plus = eigen(Symmetric(X' * H_plus * X))
c_plus = X * F_plus.vectors[:, 1]
psm_projected_energy = first(F_plus.values)

# Π₊P₂₃Π₊ = (P₂₃ + P₂₃²)/2; evaluate the composed permutation too.
MP2 = FewBodyECG.build_operator_matrix(basis, P23 * P23)
psm_exchange_parity = dot(c_plus, (MP + MP2) * c_plus) / (2 * dot(c_plus, S_plus * c_plus))
@assert isapprox(psm_exchange_parity, 1; atol = 1.0e-9, rtol = 0)
@assert psm_ref - 1.0e-8 ≤ psm_projected_energy < -0.25  # Bound below Ps + e⁻.

println("Ps⁻ unprojected E = ", psm.E₀, " Ha")
println("Ps⁻ projected E = ", psm_projected_energy, " Ha  (book ", psm_ref, ")")
println("electron-exchange expectation ⟨P₂₃⟩ = ", psm_exchange_parity)
p_psm = plot(psm, psm_ref; title = "Ps⁻ electron-exchange projection")
scatter!(p_psm, [length(basis)], [psm_projected_energy]; label = "Π₊-projected basis")
