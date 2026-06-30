# =============================================================================
# H₂⁺ (dihydrogen cation) as a *direct* three-body Coulomb bound state.
#
# Instead of the Born–Oppenheimer approximation, we treat e⁻ + p + p as a full
# non-adiabatic three-body problem and solve for the ground state with
# explicitly correlated Gaussians.  The true non-relativistic, non-BO ground
# state is E₀ = -0.597139 Ha; it lies *below* the H + p⁺ dissociation threshold
# (≈ -0.4997 Ha), which is what makes H₂⁺ a bound molecule.
#
# We solve it two independent ways — a stochastic competitive search and a
# variational (LBFGS) optimisation — built on the same matrix elements but with
# completely different search strategies.  Agreement on the energy is a strong
# correctness check.  The variational solver additionally adapts each Gaussian's
# width, so it resolves the proton–proton bond: that Jacobi coordinate is
# mass-weighted (scaled by √(mₚ/2) ≈ 30), so it needs much broader Gaussians
# than the light electron coordinate, which a single fixed width cannot span.
# =============================================================================

using FewBodyECG, LinearAlgebra
using Plots

mp = 1836.15267343                      # proton / electron mass ratio (CODATA)
masses = [mp, mp, 1.0]                  # proton, proton, electron
ops = Operators(masses, [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"                        # auto: p–p (+1), two p–e (−1)

E_exact = -0.597139063
E_threshold = -0.5                      # H + p⁺ dissociation threshold

# --- two independent solvers -------------------------------------------------
sr_stoch = solve_ECG_competitive(ops, 120; n_candidates = 40, scale = 1.0, verbose = false)
sr_var = solve_ECG_variational(ops, 15; scale = 1.0, verbose = false)

println("stochastic (competitive, 120 fns):  E = $(round(sr_stoch.ground_state, digits = 6)) Ha")
println("variational (LBFGS, 30 fns):         E = $(round(sr_var.ground_state, digits = 6)) Ha")
println("exact (non-BO):                          $E_exact Ha")
println("methods agree to:                    $(round(abs(sr_stoch.ground_state - sr_var.ground_state), digits = 6)) Ha")
println("both bound below H + p⁺ threshold?   ", max(sr_stoch.ground_state, sr_var.ground_state) < E_threshold)

# --- convergence: the stochastic build-up meets the variational / exact energy
ns, Es = convergence(sr_stoch)
p1 = plot(
    ns, Es;
    xlabel = "number of basis functions", ylabel = "E (Ha)",
    label = "stochastic (competitive)", lw = 2,
    title = "H₂⁺ ground state — method agreement",
    ylims = (-0.62, -0.45), legend = :topright,
)
hline!(p1, [sr_var.ground_state]; ls = :dashdot, color = :green, label = "variational (30 fns)")
hline!(p1, [E_exact]; ls = :dash, color = :red, label = "exact (−0.5971)")
hline!(p1, [E_threshold]; ls = :dot, color = :gray, label = "H + p⁺ threshold")
display(p1)

# --- proton–proton radial correlation density (the molecular bond) -----------
# We use the variational wavefunction here: the stochastic single-scale basis
# reproduces the energy but compresses the bond (it cannot reach far enough
# along the mass-weighted proton–proton axis).  The package works in
# mass-weighted Jacobi coordinates; convert the first Jacobi axis (proton–
# proton) back to a physical distance in bohr.
J, _ = _jacobi_transform(masses)
μ_pp = abs(J[1, 2])                     # scaled distance = μ_pp × physical distance

r_scaled, ρ = correlation_function(
    sr_var; coord_index = 1, rmin = 1.0e-3,
    rmax = 160.0, npoints = 600
)
r_pp = r_scaled ./ μ_pp
p2 = plot(
    r_pp, ρ;
    xlabel = "proton–proton distance (a₀)", ylabel = "ρ(r) = r² |ψ(r)|²",
    label = "p–p density", lw = 2, xlims = (0, 5),
    title = "H₂⁺ proton–proton correlation",
)
vline!(p2, [2.0]; ls = :dash, color = :gray, label = "≈ equilibrium bond (2 a₀)")
display(p2)
