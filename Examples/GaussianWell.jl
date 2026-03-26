using FewBodyECG
using Plots

# ─────────────────────────────────────────────────────────────────────────────
# Gaussian well: a unit-mass particle bound by a finite-range Gaussian potential
#
#   V(r) = −V₀ exp(−γ r²)
#
# Unlike the Coulomb 1/r potential, a Gaussian well has finite range and a
# binding threshold: no bound state exists below a critical depth V₀c.  This
# makes it a useful model for nuclear interactions (short-range forces) and
# is the same form factor used in nuclear models with explicit mesons.
#
# The GaussianOperator matrix element is exact and requires no special
# functions — it simply shifts the Gaussian exponent matrix S → S + γ ww',
# making it the cheapest operator to evaluate after the overlap itself.
# ─────────────────────────────────────────────────────────────────────────────

masses = [1.0e15, 1.0]   # infinitely heavy source + unit-mass particle

γ  = 1.0    # inverse-square range  (a₀⁻²);  spatial range ≈ 1/√γ = 1 a₀
V₀ = 5.0   # well depth  (Ha);  attractive → coefficient = −V₀

ops = Operators(masses)
ops += "Kinetic"
ops += ("Gaussian", 1, 2, -V₀, γ)   # V(r₁₂) = −V₀ exp(−γ r₁₂²)

result = solve_ECG(ops, 60; scale = 1.0, verbose = false)

println("Gaussian well  (V₀ = $V₀ Ha,  γ = $γ a₀⁻²)")
println("  Ground state energy:  E = $(round(result.ground_state; digits = 8)) Ha")

# ── Convergence ───────────────────────────────────────────────────────────────
n_conv, E_conv = convergence(result)

p1 = plot(n_conv, E_conv;
    xlabel = "Basis size", ylabel = "E (Ha)",
    label = "Ground state", lw = 2,
    title = "Gaussian well convergence")
hline!(p1, [result.ground_state]; ls = :dash, color = :gray, label = "Converged value")
display(p1)

# ── Binding threshold scan ────────────────────────────────────────────────────
# The Gaussian well supports a bound state only above a critical depth V₀c.
# Scan V₀ and record the variational ground state energy.
V₀_range = 0.5:0.5:8.0
E_scan = map(V₀_range) do V₀_i
    ops_i = Operators(masses)
    ops_i += "Kinetic"
    ops_i += ("Gaussian", 1, 2, -V₀_i, γ)
    solve_ECG(ops_i, 30; scale = 1.0, verbose = false).ground_state
end

p2 = plot(V₀_range, E_scan;
    xlabel = "Well depth V₀ (Ha)", ylabel = "E (Ha)",
    label = "Ground state", lw = 2,
    title = "Gaussian well: binding threshold  (γ = $γ a₀⁻²)")
hline!(p2, [0.0]; ls = :dash, color = :gray, label = "Continuum threshold")
display(p2)

# ── Radial density ────────────────────────────────────────────────────────────
r_grid, ρ = correlation_function(result; rmax = 6.0, npoints = 300)

p3 = plot(r_grid, ρ;
    xlabel = "r (a₀)", ylabel = "r² |ψ(r)|²",
    label = "Gaussian well  (V₀ = $V₀ Ha)", lw = 2,
    title = "Radial density: Gaussian well vs hydrogen")

# Compare with hydrogen (Coulomb 1/r) on the same plot
ops_H = Operators(masses)
ops_H += "Kinetic"
ops_H += ("Coulomb", 1, 2, -1.0)
result_H = solve_ECG(ops_H, 60; scale = 1.0, verbose = false)
r_H, ρ_H = correlation_function(result_H; rmax = 6.0, npoints = 300)
plot!(p3, r_H, ρ_H; lw = 2, ls = :dash, label = "Hydrogen  (Coulomb 1/r)")
display(p3)
