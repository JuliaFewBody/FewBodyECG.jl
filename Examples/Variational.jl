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
using QuasiMonteCarlo
import FewBodyECG: default_scale, BasisSet, Rank0Gaussian

# ============================================================
# 1. Hydrogen anion H⁻
# ============================================================

println("=" ^ 60)
println("Hydrogen anion H⁻")
println("=" ^ 60)

masses_Hm = [1.0e15, 1.0, 1.0]   # fixed nucleus + 2 electrons
Λ_Hm = Λ(masses_Hm)
_, U_Hm = _jacobi_transform(masses_Hm)

w_pairs = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
w_raw_Hm = [U_Hm' * Float64.(w) for w in w_pairs]
coeffs_Hm = [-1.0, -1.0, +1.0]   # e-nucleus (×2) and e-e repulsion

ops_Hm = Operator[
    KineticOperator(Λ_Hm);
    [CoulombOperator(c, w) for (c, w) in zip(coeffs_Hm, w_raw_Hm)]...
]

E_exact_Hm = -0.527751016523
n = 30

# --- (a) fresh optimisation with :energy loss ---
println("\n(a) Fresh start, loss_type = :energy")
sr_var = solve_ECG_variational(ops_Hm, n; scale = 1.0, max_iterations = 500, verbose = false)
ΔE_var = sr_var.ground_state - E_exact_Hm
println("  Variational  E₀ = $(round(sr_var.ground_state, digits=8))  ΔE = $(round(ΔE_var, sigdigits=3))")

# --- stochastic baseline ---
sr_stoch = solve_ECG(ops_Hm, n; scale = 1.0, verbose = false)
ΔE_stoch = sr_stoch.ground_state - E_exact_Hm
println("  Stochastic   E₀ = $(round(sr_stoch.ground_state, digits=8))  ΔE = $(round(ΔE_stoch, sigdigits=3))")
println("  Exact        E₀ = $E_exact_Hm")

# --- (b) warm-start from stochastic result with :trace loss ---
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

# --- downstream utilities work unchanged ---
r_grid, ρ = correlation_function(sr_var; rmin = 0.01, rmax = 15.0, npoints = 200)
println("\n  Correlation function computed: $(length(r_grid)) points, max ρ at r = $(round(r_grid[argmax(ρ)], digits=3)) a.u.")

# ============================================================
# 2. Muonic molecule tdμ
# ============================================================

println()
println("=" ^ 60)
println("Muonic molecule tdμ  (triton + deuteron + muon)")
println("=" ^ 60)

masses_tdμ = [5496.918, 3670.481, 206.7686]   # t, d, μ in electron masses
Λ_tdμ = Λ(masses_tdμ)
_, U_tdμ = _jacobi_transform(masses_tdμ)

w_raw_tdμ = [U_tdμ' * Float64.(w) for w in w_pairs]
coeffs_tdμ = [+1.0, -1.0, -1.0]   # t-d repulsion, t-μ and d-μ attraction

ops_tdμ = Operator[
    KineticOperator(Λ_tdμ);
    [CoulombOperator(c, w) for (c, w) in zip(coeffs_tdμ, w_raw_tdμ)]...
]

E_exact_tdμ = -111.36444
scale_tdμ = 0.03   # nuclear scale (much smaller than atomic)
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

# --- Variational convergence (energy vs fg evaluations) ---
n_fg, E_fg = convergence_history(sr_var)
p1 = plot(n_fg, E_fg;
    label = "Variational",
    xlabel = "fg evaluations",
    ylabel = "Energy (Ha)",
    title = "H⁻ variational convergence  (n = $n)",
    lw = 2,
)
hline!(p1, [E_exact_Hm]; label = "Exact", linestyle = :dot, color = :black, lw = 1)

# --- Stochastic greedy convergence (energy vs basis size) ---
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