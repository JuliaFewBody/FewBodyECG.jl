# # Hydrogen spectroscopy: electric dipole transitions
#
# The 1s → 2p₀ dipole matrix element and oscillator strength in the
# fixed-proton limit, in atomic units, as the basis grows.  The 1s states use
# plain Gaussians and the 2p₀ states use z-polarized Rank1 Gaussians.

using FewBodyECG
using FewBodyECG: build_operator_matrix
using LinearAlgebra
using Plots

masses, charges = [1.0e15, 1.0], [1.0, -1.0]
μ = prod(masses) / sum(masses)
H = Kinetic(masses) + Coulomb(charges)
Dz = DipoleOperator(masses, charges)

basis_sizes = [1, 3, 5, 7, 9, 11, 13, 15]
s_states = [solve(H, SVM(k; widths = :wide, indep_tol = 1.0e-8)) for k in basis_sizes]
p_states = [solve(H, SVM(k; rank = 1, widths = :wide, indep_tol = 1.0e-8)) for k in basis_sizes]

## Coefficients are overlap-normalized, so c₁ₛ' D c₂ₚ is the transition element.
dipoles = map(s_states, p_states) do s, p
    M = build_operator_matrix(s.basis, p.basis, Dz)
    abs(dot(s.coefficients[:, 1], M * p.coefficients[:, 1]))
end
gaps = [energy(p) - energy(s) for (s, p) in zip(s_states, p_states)]
oscillator_strengths = 2 .* gaps .* abs2.(dipoles)   # summed over the three 2p substates
exact_strength = 8192 / (19683μ)

s, p = last(s_states), last(p_states)
@info(
    "1s → 2p₀", dipole = last(dipoles), f = last(oscillator_strengths), f_exact = exact_strength,
    Qzz_2p = expectation(p, QuadrupoleOperator(masses, charges)),
)

## Radial amplitudes u(r) = √(4π) r ψ(r); the Jacobi coordinate is x = √μ r.
## `normalize = false` keeps the full-space normalization of ψ.
x, density_s = radial_profile(wavefunction(s); rmax = 20sqrt(μ), npoints = 800, normalize = false)
_, density_p = radial_profile(wavefunction(p); rmax = 20sqrt(μ), npoints = 800, normalize = false)
r = x ./ sqrt(μ)
u_1s = sqrt.(4π * sqrt(μ) .* density_s)
u_2p = sqrt.(4π / 3 * sqrt(μ) .* density_p)
exact_1s = 2μ^(3 / 2) .* r .* exp.(-μ .* r)
exact_2p = μ^(5 / 2) / (2sqrt(6)) .* r .^ 2 .* exp.(-μ .* r ./ 2)

energy_plot = plot(
    convergence(s); label = "1s", marker = :circle,
    xlabel = "step", ylabel = "E (Ha)", title = "Variational energies",
)
plot!(energy_plot, convergence(p); label = "2p", marker = :diamond)
hline!(energy_plot, [-μ / 2, -μ / 8]; label = "exact", color = :gray, linestyle = :dash)

strength_plot = plot(
    basis_sizes, oscillator_strengths; label = "ECG", marker = :circle,
    xlabel = "Gaussians per state", ylabel = "oscillator strength f", title = "Transition convergence",
)
hline!(strength_plot, [exact_strength]; label = "exact", color = :black, linestyle = :dash)

radial_plot = plot(
    r, [u_1s u_2p]; label = ["1s ECG" "2p ECG"],
    xlabel = "r (a₀)", ylabel = "u(r) (a₀^-1/2)", title = "Radial wavefunctions",
)
plot!(radial_plot, r, [exact_1s exact_2p]; label = ["1s exact" "2p exact"], color = [1 2], linestyle = :dash)

## The dipole element is the area under u₁ₛ(r) r u₂ₚ(r) / √3.
density_plot = plot(
    r, u_1s .* r .* u_2p ./ sqrt(3); label = "area = $(round(last(dipoles); digits = 6)) e a₀",
    fillrange = 0, fillalpha = 0.15,
    xlabel = "r (a₀)", ylabel = "u₁ₛ r u₂ₚ / √3", title = "Where the dipole comes from",
)

plot(energy_plot, strength_plot, radial_plot, density_plot; layout = (2, 2), size = (1000, 760))
