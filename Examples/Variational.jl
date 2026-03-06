# Variational ECG optimisation example
#
# Demonstrates solve_ECG_variational for two benchmark systems:
#
#   1. Hydrogen anion H⁻ (3-body: nucleus + 2 electrons)
#      Exact ground-state energy: -0.527751016523 Ha
#
#   2. Muonic molecule tdμ (3-body: triton + deuteron + muon)
#      Exact ground-state energy: -111.36444 Ha
#
# Two usage patterns are shown:
#
#   (a) Fresh optimisation with loss_type = :energy (default)
#       — starts from a QMC-generated initial basis.
#
#   (b) Warm-start from solve_ECG result with loss_type = :trace
#       — refines an already-good stochastic basis by minimising
#         Tr(S⁻¹H).  Requires the initial trace to be negative,
#         which is guaranteed when the stochastic basis is close
#         to the physical ground state.

using FewBodyECG
using LinearAlgebra
import FewBodyECG: default_scale, BasisSet, Rank0Gaussian

masses_Hm = [1.0e15, 1.0, 1.0]   # fixed nucleus + 2 electrons

ops_Hm = Operators(masses_Hm, [+1, -1, -1])   # proton, e₁, e₂
ops_Hm += "Kinetic"
ops_Hm += "Coulomb"

E_exact_Hm = -0.527751016523
n = 30

println("\n(a) Fresh start, loss_type = :energy")
sr_var = solve_ECG_variational(ops_Hm, n; scale = 1.0, max_iterations = 500, verbose = false)
ΔE_var = sr_var.ground_state - E_exact_Hm
println("  Variational  E₀ = $(round(sr_var.ground_state, digits=8))  ΔE = $(round(ΔE_var, sigdigits=3))")

sr_stoch = solve_ECG(ops_Hm, n; scale = 1.0, verbose = false)
ΔE_stoch = sr_stoch.ground_state - E_exact_Hm
println("  Stochastic   E₀ = $(round(sr_stoch.ground_state, digits=8))  ΔE = $(round(ΔE_stoch, sigdigits=3))")
println("  Exact        E₀ = $E_exact_Hm")

println("\n(b) Warm-start from stochastic, loss_type = :trace")
basis0_Hm = BasisSet(Rank0Gaussian[sr_stoch.basis_functions...])
sr_warm = solve_ECG_variational(ops_Hm, n;
    initial_basis = basis0_Hm,
    loss_type = :trace,
    max_iterations = 500,
    verbose = false,
)
ΔE_warm = sr_warm.ground_state - E_exact_Hm
println("  Warm-start   E₀ = $(round(sr_warm.ground_state, digits=8))  ΔE = $(round(ΔE_warm, sigdigits=3))")
println("  Stochastic   E₀ = $(round(sr_stoch.ground_state, digits=8))  ΔE = $(round(ΔE_stoch, sigdigits=3))")
println("  Exact        E₀ = $E_exact_Hm")

r_grid, ρ = correlation_function(sr_var; rmin = 0.01, rmax = 15.0, npoints = 200)
println("\n  Correlation function computed: $(length(r_grid)) points, max ρ at r = $(round(r_grid[argmax(ρ)], digits=3)) a.u.")

masses_tdμ = [5496.918, 3670.481, 206.7686]   # t, d, μ in electron masses

ops_tdμ = Operators(masses_tdμ, [+1, +1, -1])   # triton, deuteron, muon
ops_tdμ += "Kinetic"
ops_tdμ += "Coulomb"   # t-d repulsion (+1), t-μ and d-μ attraction (-1)

E_exact_tdμ = -111.36444
scale_tdμ = 0.03   # nuclear scale
n_tdμ = 25

println("\n(a) Fresh start, loss_type = :energy")
sr_var_tdμ = solve_ECG_variational(ops_tdμ, n_tdμ;
    scale = scale_tdμ, max_iterations = 500, verbose = false
)
ΔE_var_tdμ = sr_var_tdμ.ground_state - E_exact_tdμ
println("  Variational  E₀ = $(round(sr_var_tdμ.ground_state, digits=4))  ΔE = $(round(ΔE_var_tdμ, sigdigits=3))")

sr_stoch_tdμ = solve_ECG(ops_tdμ, n_tdμ; scale = scale_tdμ, verbose = false)
ΔE_stoch_tdμ = sr_stoch_tdμ.ground_state - E_exact_tdμ
println("  Stochastic   E₀ = $(round(sr_stoch_tdμ.ground_state, digits=4))  ΔE = $(round(ΔE_stoch_tdμ, sigdigits=3))")
println("  Exact        E₀ = $E_exact_tdμ")


using Plots
import FewBodyECG: convergence_history

n_fg, E_fg = convergence_history(sr_var)
p1 = plot(n_fg, E_fg;
    label = "Variational",
    xlabel = "fg evaluations",
    ylabel = "Energy (Ha)",
    title = "H⁻ variational convergence  (n = $n)",
    lw = 2,
)
hline!(p1, [E_exact_Hm]; label = "Exact", linestyle = :dot, color = :black, lw = 1)

n_s, E_s = convergence(sr_stoch)
p2 = plot(n_s, E_s;
    label = "Stochastic greedy",
    xlabel = "Basis size",
    ylabel = "Energy (Ha)",
    title = "H⁻ stochastic convergence  (n = $n)",
    lw = 2,
)
hline!(p2, [sr_var.ground_state]; label = "Variational (n=$n)", linestyle = :dash, lw = 1)
hline!(p2, [E_exact_Hm]; label = "Exact", linestyle = :dot, color = :black, lw = 1)

plot(p1, p2; layout = (2, 1), size = (700, 600))