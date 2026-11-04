# # Why the deuteron binds: spin and isospin exchange
# Minnesota central S-wave model: compare (S,T)=(1,0) and (0,1).
# Suzuki–Varga, Table 11.1, p. 217; deuteron reference: Table 11.9, p. 237.

using FewBodyECG
using LinearAlgebra
using Plots
include(joinpath(@__DIR__, "nuclear_exchange_model.jl"))

masses = fill(1 / 41.47, 2)  # ℏ²/m = 41.47 MeV fm²
_, U = FewBodyECG.jacobi_transform(masses)
w = U' * [1.0, -1.0]
Pr = PermutationOperator(masses, [2, 1])

# V = Σ Vₖ exp(-γₖr²)(Wₖ + MₖPʳ + BₖPσ - HₖPτ) in a Pauli-allowed channel.
widths = exp10.(range(-4, 2, length = 24))
orbitals = pair_orbitals(0, widths, w)
basis = nucleon_basis(orbitals)
overlap = FewBodyECG.build_overlap_matrix(basis)
H = FewBodyECG.build_operator_matrix(basis, KineticOperator(masses)) +
    exchange_matrix(basis, minnesota_rows(1.0), w, Pr; isospin_form = true)
C_triplet = channel_map(length(orbitals), 1, 0)
C_singlet = channel_map(length(orbitals), 0, 1)
H_triplet, S_triplet = C_triplet' * H * C_triplet, C_triplet' * overlap * C_triplet
H_singlet, S_singlet = C_singlet' * H * C_singlet, C_singlet' * overlap * C_singlet

# Positive singlet levels approximate the continuum.
basis_sizes = collect(4:2:24)
triplet_states = [channel_solution(H_triplet[1:k, 1:k], S_triplet[1:k, 1:k], orbitals[1:k]) for k in basis_sizes]
singlet_states = [channel_solution(H_singlet[1:k, 1:k], S_singlet[1:k, 1:k], orbitals[1:k]) for k in basis_sizes]
triplet_energies = [state.energy for state in triplet_states]
singlet_energies = [state.energy for state in singlet_states]
deuteron = last(triplet_states)

S_orb = FewBodyECG.build_overlap_matrix(BasisSet(orbitals))
R2 = FewBodyECG.build_operator_matrix(BasisSet(orbitals), OscillatorOperator(1.0, w))
c = deuteron.coefficients
rms_separation = sqrt(real(dot(c, R2 * c) / dot(c, S_orb * c)))
println(
    "Deuteron: E = ", round(deuteron.energy; digits = 6), " MeV; rms separation = ",
    round(rms_separation; digits = 3), " fm"
)

triplet_potential(r) = 200exp(-1.487r^2) - 178exp(-0.639r^2)
singlet_potential(r) = 200exp(-1.487r^2) - 91.85exp(-0.465r^2)
r_potential = range(0, 6, length = 400)
triplet_color, singlet_color = "#b82a6b", "#426b9b"
style = (;
    framestyle = :box, gridalpha = 0.12, linewidth = 2,
    titlefontsize = 12, guidefontsize = 11, tickfontsize = 10, legendfontsize = 9,
    left_margin = 4Plots.mm, bottom_margin = 3Plots.mm,
)

potential_plot = plot(
    r_potential, triplet_potential.(r_potential);
    label = "S=1, T=0", color = triplet_color, xlabel = "r (fm)", ylabel = "V(r) (MeV)",
    title = "(a) Spin-dependent attraction", style...
)
plot!(
    potential_plot, r_potential, singlet_potential.(r_potential);
    label = "S=0, T=1", color = singlet_color, linestyle = :dash
)
hline!(potential_plot, [0.0]; color = :gray, linewidth = 1, label = "")

convergence_plot = plot(
    convergence(triplet_energies; steps = basis_sizes);
    label = "triplet", marker = :circle, markersize = 3, color = triplet_color,
    markerstrokecolor = triplet_color, markerstrokewidth = 0.5,
    xlabel = "radial Gaussians N", ylabel = "E (MeV)", title = "(b) Basis convergence", style...
)
plot!(
    convergence_plot, convergence(singlet_energies; steps = basis_sizes);
    label = "singlet", color = singlet_color, marker = :diamond, markersize = 3, linestyle = :dash,
    markerstrokecolor = singlet_color, markerstrokewidth = 0.5
)
hline!(convergence_plot, [0.0]; color = :gray, linewidth = 1, linestyle = :dash, label = "p+n threshold")
hline!(convergence_plot, [-2.202]; color = :black, linewidth = 1, linestyle = :dot, label = "book (rounded)")

# u(r)=√(4π)rψ(r), normalized in physical fm.
radii = range(0, 20, length = 600)
amplitude_plot = plot(;
    xlabel = "r (fm)", ylabel = "u(r) [1/sqrt(fm)]",
    title = "(c) Radial wavefunction", style...
)
for (k, color, line) in ((12, :gray, :dot), (16, singlet_color, :dash), (24, triplet_color, :solid))
    state = triplet_states[findfirst(==(k), basis_sizes)]
    plot!(
        amplitude_plot, radii, s_wave_radial_amplitude(state.wavefunction, radii, w);
        label = "N=$k", color, linestyle = line
    )
end
u = s_wave_radial_amplitude(deuteron.wavefunction, radii, w)
probability_plot = plot(
    radii, abs2.(u);
    label = "deuteron", color = triplet_color, fillrange = 0, fillalpha = 0.12,
    xlabel = "r (fm)", ylabel = "P(r) [1/fm]", title = "(d) Separation probability", style...
)
vline!(
    probability_plot, [rms_separation]; color = :black, linestyle = :dash, linewidth = 1,
    label = "r_rms = $(round(rms_separation; digits = 2)) fm"
)

physics_plot = plot(
    potential_plot, convergence_plot, amplitude_plot, probability_plot;
    layout = (2, 2), size = (1000, 760), dpi = 180
)
physics_plot
