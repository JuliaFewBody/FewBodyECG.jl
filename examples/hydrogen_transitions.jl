# # Hydrogen spectroscopy: electric dipole transitions
#
# A transition probes the overlap of two different wavefunctions. Here we solve
# the even 1s and odd 2p₀ states in separate Gaussian bases, then compute the
# electric dipole matrix element connecting them. The energy difference sets
# the photon energy; the dipole matrix element sets the transition strength.
#
# This is the analytic benchmark for the Ps₂ dipole-decay calculation discussed
# by Suzuki–Varga, §8.3, Eqs. (8.3)–(8.5), pp. 157–158, and
# [Usukura, Varga and Suzuki (1998)](https://arxiv.org/abs/physics/9804023).
# Atomic units are used, with a very heavy proton approximating a fixed nucleus.

using FewBodyECG
using LinearAlgebra
using Plots

masses, charges = [1.0e15, 1.0], [1.0, -1.0]
μ = prod(masses) / sum(masses)
H = Kinetic(masses) + Coulomb(charges)
Dz = DipoleOperator(masses, charges)

# Each basis has its own width ladder. Adding functions gives nested variational
# spaces; ordering widths around the atomic scale makes small bases useful too.
widths_s = sort(exp10.(range(-3.5, 4, length = 28)); by = a -> abs(log10(a) + 0.5))
widths_p = sort(exp10.(range(-3.5, 2.5, length = 26)); by = a -> abs(log10(a) + 1.25))
s_orbitals = [Rank0Gaussian([a;;], zeros(1)) for a in widths_s]
p_orbitals = [Rank1Gaussian([a;;], [1.0], zeros(1)) for a in widths_p]

function atomic_level(orbitals, H)
    basis = BasisSet(orbitals)
    S = FewBodyECG.build_overlap_matrix(basis)
    D = Diagonal(1 ./ sqrt.(diag(S)))
    cond(D * S * D) < 1.0e12 || throw(ArgumentError("Gaussian widths are too closely spaced"))
    energies, vectors = FewBodyECG.solve_generalized_eigenproblem(
        D * FewBodyECG.build_hamiltonian_matrix(basis, H) * D, D * S * D
    )
    c = D * vectors[:, 1]  # c'Sc = 1 in the original, unscaled basis
    return (; energy = first(energies), basis, c, ψ = Wavefunction(basis, c))
end

basis_sizes = collect(4:4:28)
s_states = [atomic_level(s_orbitals[1:k], H) for k in basis_sizes]
p_states = [atomic_level(p_orbitals[1:(k - 2)], H) for k in basis_sizes]
dipoles = map(s_states, p_states) do s, p
    M = FewBodyECG.build_operator_matrix(s.basis, p.basis, Dz)
    abs(dot(s.c, M * p.c))
end
gaps = [p.energy - s.energy for (s, p) in zip(s_states, p_states)]

# The Cartesian matrix element is for m=0 and z polarization. Summing the three
# degenerate 2p substates gives f=(2/3)ΔE Σₘ|⟨1s|D|2pₘ⟩|²=2ΔE|D_z|².
# Transition strengths need not converge monotonically, unlike the energies.
oscillator_strengths = 2 .* gaps .* abs2.(dipoles)
exact_dipole = 128sqrt(2) / (243μ)
exact_oscillator = 8192 / (19683μ)
s, p = last(s_states), last(p_states)
Qzz = QuadrupoleOperator(masses, charges)
quadrupole = real(dot(p.c, FewBodyECG.build_operator_matrix(p.basis, Qzz) * p.c))
exact_quadrupole = 24 * (masses[2] - masses[1]) / sum(masses) / μ^2
forbidden = dot(s.c, FewBodyECG.build_operator_matrix(s.basis, p.basis, DipoleOperator(masses, charges; component = 1)) * p.c)

println("Hydrogen 1s → 2p electric-dipole transition")
println("  E(1s) = ", s.energy, " Ha; E(2p) = ", p.energy, " Ha")
println("  photon energy = ", last(gaps), " Ha  (exact = ", 3μ / 8, ")")
println("  |⟨1s|Dz|2p₀⟩| = ", last(dipoles), " e a₀  (exact = ", exact_dipole, ")")
println("  total 1s→2p oscillator strength = ", last(oscillator_strengths), "  (exact = ", exact_oscillator, ")")
println("  forbidden x-polarized 1s→2p₀ amplitude = ", forbidden)
println("  ⟨2p₀|Qzz|2p₀⟩ = ", quadrupole, " e a₀²  (exact = ", exact_quadrupole, ")")

# The quadrupole is even under parity: it measures the elongated 2p₀ charge
# distribution, whereas the dipole connects the opposite-parity s and p states.
# These are Cartesian moments, not reduced spherical transition probabilities.

# Convert the mass-scaled coordinate x=√μ r back to physical radius. For the
# s and p₀ states, ψ(r,z-axis)=R(r)Yₗ₀(0); u=rR has ∫|u|²dr=1.
# Keep the full-space normalization instead of normalizing the plotting window.
x, density_s = radial_profile(s.ψ; rmax = 20sqrt(μ), npoints = 800, normalize = false)
_, density_p = radial_profile(p.ψ; rmax = 20sqrt(μ), npoints = 800, normalize = false)
radii = x ./ sqrt(μ)
# These lowest s/p radial states have no nodes; choose their positive phase.
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
    basis_sizes, [state.energy for state in s_states];
    label = "1s", color = s_color, marker = :circle, markerstrokewidth = 0,
    legend = :right,
    xlabel = "radial Gaussians N (s), N−2 (p)", ylabel = "E (Ha)",
    title = "(a) Variational energies", style...
)
plot!(energy_plot, basis_sizes, [state.energy for state in p_states]; label = "2p", color = p_color, marker = :diamond, markerstrokewidth = 0)
hline!(energy_plot, [-μ / 2, -μ / 8]; label = "exact", color = :gray, linestyle = :dash, linewidth = 1)

strength_plot = plot(
    basis_sizes, oscillator_strengths;
    label = "ECG", color = s_color, marker = :circle, markerstrokewidth = 0,
    xlabel = "radial Gaussians N (s), N−2 (p)", ylabel = "oscillator strength f",
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
