# # Hydrogen spectroscopy: electric dipole transitions
# 1s → 2p₀ in the fixed-proton limit, in atomic units.
# Ps₂ application: Suzuki–Varga, §8.3, pp. 157–158;
# [Usukura, Varga and Suzuki (1998)](https://arxiv.org/abs/physics/9804023).

using FewBodyECG
using LinearAlgebra
using Plots

n_gaussians = 15

masses, charges = [1.0e15, 1.0], [1.0, -1.0]
μ = prod(masses) / sum(masses)
H = Kinetic(masses) + Coulomb(charges)
Dz = DipoleOperator(masses, charges)

basis_sizes = unique(round.(Int, range(1, n_gaussians; length = min(n_gaussians, 8))))
s_states = [solve(H, SVM(k; widths = :wide, indep_tol = 1.0e-8)) for k in basis_sizes]
p_states = [solve(H, SVM(k; rank = 1, widths = :wide, indep_tol = 1.0e-8)) for k in basis_sizes]
dipoles = map(s_states, p_states) do s, p
    M = FewBodyECG.build_operator_matrix(s.basis, p.basis, Dz)
    abs(dot(s.coefficients[:, 1], M * p.coefficients[:, 1]))
end
gaps = [energy(p) - energy(s) for (s, p) in zip(s_states, p_states)]

# Sum over the three 2p substates: f=2ΔE|D_z|².
oscillator_strengths = 2 .* gaps .* abs2.(dipoles)
exact_oscillator = 8192 / (19683μ)
s, p = last(s_states), last(p_states)
Qzz = QuadrupoleOperator(masses, charges)
quadrupole = real(dot(p.coefficients[:, 1], FewBodyECG.build_operator_matrix(p.basis, Qzz) * p.coefficients[:, 1]))

# x=√μ r; normalize=false preserves the full-space norm.
x, density_s = radial_profile(wavefunction(s); rmax = 20sqrt(μ), npoints = 800, normalize = false)
_, density_p = radial_profile(wavefunction(p); rmax = 20sqrt(μ), npoints = 800, normalize = false)
radii = x ./ sqrt(μ)
# Positive radial amplitudes for the nodeless 1s and 2p states.
us = sqrt.(4π * sqrt(μ) .* density_s)
up = sqrt.(4π / 3 * sqrt(μ) .* density_p)
exact_us = 2μ^(3 / 2) .* radii .* exp.(-μ .* radii)
exact_up = μ^(5 / 2) / (2sqrt(6)) .* radii .^ 2 .* exp.(-μ .* radii ./ 2)
transition_density = us .* radii .* up ./ sqrt(3)

s_color, p_color = "#b82a6b", "#426b9b"
style = (;
    framestyle = :box, gridalpha = 0.12, linewidth = 2,
    titlefontsize = 12, guidefontsize = 11, tickfontsize = 10, legendfontsize = 9,
    left_margin = 4Plots.mm, bottom_margin = 3Plots.mm,
)
energy_plot = plot(
    convergence(s);
    label = "1s", color = s_color, marker = :circle, markerstrokewidth = 0,
    legend = :right,
    xlabel = "step", ylabel = "E (Ha)",
    title = "(a) Variational energies", style...
)
plot!(energy_plot, convergence(p); label = "2p", color = p_color, marker = :diamond, markerstrokewidth = 0)
hline!(energy_plot, [-μ / 2, -μ / 8]; label = "exact", color = :gray, linestyle = :dash, linewidth = 1)

strength_plot = plot(
    basis_sizes, oscillator_strengths;
    label = "ECG", color = s_color, marker = :circle, markerstrokewidth = 0,
    xlabel = "requested Gaussians per state", ylabel = "oscillator strength f",
    title = "(b) Transition convergence", style...
)
hline!(strength_plot, [exact_oscillator]; label = "exact", color = :black, linestyle = :dash, linewidth = 1)

radial_plot = plot(
    radii, us; label = "1s ECG", color = s_color,
    xlabel = "r (a₀)", ylabel = "u(r) [1/sqrt(a₀)]", title = "(c) Radial wavefunctions", style...
)
plot!(radial_plot, radii, up; label = "2p ECG", color = p_color)
plot!(radial_plot, radii, exact_us; label = "1s exact", color = s_color, linestyle = :dash, linewidth = 1)
plot!(radial_plot, radii, exact_up; label = "2p exact", color = p_color, linestyle = :dash, linewidth = 1)

density_plot = plot(
    radii, transition_density;
    label = "area = $(round(last(dipoles); digits = 6)) e a₀",
    color = s_color, fillrange = 0, fillalpha = 0.12,
    xlabel = "r (a₀)", ylabel = "u₁ₛ(r) r u₂ₚ(r) / sqrt(3)",
    title = "(d) Where the dipole comes from", style...
)

physics_plot = plot(energy_plot, strength_plot, radial_plot, density_plot; layout = (2, 2), size = (1000, 760), dpi = 180)
physics_plot
