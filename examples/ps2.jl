using FewBodyECG
using Plots

masses = [1.0, 1.0, 1.0, 1.0]
charges = [+1.0, -1.0, +1.0, -1.0]

ops = Kinetic(masses) + Coulomb(charges)

ps2_ref = -0.516003778
sol = solve(
    ops,
    SVM(basis = 150, candidates = 30);
    tol = 1.0e-4,
    window = 15,
)

println("Ps₂ E₀ = ", sol.E₀, " Ha")
println("reference (SVM, K = 800) = ", ps2_ref, " Ha   Δ = ", sol.E₀ - ps2_ref)
println("variational upper bound respected: ", sol.E₀ ≥ ps2_ref)

plot(sol, ps2_ref; title = "Dipositronium convergence")
plot(wavefunction(sol))

# Contact density and annihilation: Suzuki–Varga §8.3, p. 157,
# Table 8.5 and Eq. (8.2). Positrons are 1,3 and electrons are 2,4.
# δ³(rᵢ-rⱼ) is a density per volume, not a radial delta function.
using LinearAlgebra: dot

function pair_contact(basis, coefficients, w)
    S = FewBodyECG.build_overlap_matrix(basis)
    D = FewBodyECG.build_operator_matrix(basis, ContactDensityOperator(w))
    return real(dot(coefficients, D * coefficients) / dot(coefficients, S * coefficients))
end

_, U = FewBodyECG.jacobi_transform(masses)
c = sol.coefficients[:, sol.state]
unlike_pairs = [(1, 2), (1, 4), (3, 2), (3, 4)]
contacts = map(unlike_pairs) do (i, j)
    e = zeros(length(masses))
    e[i], e[j] = 1.0, -1.0
    pair_contact(sol.basis, c, U' * e)
end
contact_mean = sum(contacts) / length(contacts)
contact_ref = 0.0221151  # Per unlike pair, a₀⁻³; Table 8.5, L = 0.

# In the product of electron and positron singlets, each unlike pair has
# singlet probability 1/4. Four equivalent pairs give the book's rate
# Γ/ℏ = 4πα⁴(c/a₀)⟨δ³(r₁-r₂)⟩. Average the finite-basis pair estimates.
# This is the leading two-photon lifetime, with rounded physical constants.
α = 1 / 137.036
c_light = 299_792_458.0  # m/s
a₀ = 5.29177e-11        # m
annihilation_rate = 4π * α^4 * c_light / a₀ * contact_mean  # s⁻¹
annihilation_lifetime_ns = 1.0e9 / annihilation_rate

@assert all(>(0), contacts) && all(isfinite, contacts)
@assert isapprox(contact_mean, contact_ref; rtol = 0.1)
@assert isapprox(annihilation_lifetime_ns, 0.22; rtol = 0.1)
for (pair, contact) in zip(unlike_pairs, contacts)
    println("Ps₂ contact ", pair, " = ", contact, " a₀⁻³")
end
println("mean unlike-pair contact = ", contact_mean, " a₀⁻³  (book ", contact_ref, ")")
println("two-photon annihilation lifetime = ", annihilation_lifetime_ns, " ns  (book ≈ 0.22 ns)")
# Contact densities converge more slowly than energies; increase the basis
# to resolve more digits of the tabulated contact density.
