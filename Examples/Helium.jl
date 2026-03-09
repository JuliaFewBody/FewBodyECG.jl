using FewBodyECG
using Antique
using Plots
using QuasiMonteCarlo

masses = [1e15, 1.0, 1.0]

os = Operators(masses, [+2, -1, -1])   # nucleus (Z=2), e₁, e₂
os += "Kinetic"
os += "Coulomb"   # auto: nucleus-e₁ (-2), nucleus-e₂ (-2), e₁-e₂ (+1)

E_gs_exact = -2.9037242   # 1s² ¹S  (ground state)
E_ex_exact = -2.17523     # 1s2s ¹S (first excited singlet)

println("Ground state (1s²)...")
result_gs = solve_ECG(os, 250; scale = 1.0, verbose = false)
ΔE_gs = result_gs.ground_state - E_gs_exact
println("  E = $(round(result_gs.ground_state; digits=6))  (exact: $E_gs_exact,  error: $(round(ΔE_gs; digits=6)))")

println("\nFirst excited ¹S state (1s2s)...")
result_ex = solve_ECG(os, 200; scale = 1.0, verbose = false, state = 2)
ΔE_ex = result_ex.ground_state - E_ex_exact
println("  E = $(round(result_ex.ground_state; digits=6))  (exact: $E_ex_exact,  error: $(round(ΔE_ex; digits=6)))")

n_gs, E_gs = convergence(result_gs)
n_ex, E_ex = convergence(result_ex)

p1 = plot(n_gs, E_gs,
    label = "1s² (ground)", lw = 2,
    xlabel = "Basis size", ylabel = "E (Ha)",
    title = "Helium convergence")
plot!(p1, n_ex, E_ex, label = "1s2s (excited)", lw = 2, ls = :dash)
hline!(p1, [E_gs_exact, E_ex_exact], ls = :dot, color = :gray, label = "Exact")
display(p1)

r_gs, ρ_gs = correlation_function(result_gs; rmax = 5.0)
r_ex, ρ_ex = correlation_function(result_ex; rmax = 5.0)

p2 = plot(r_gs, ρ_gs,
    label = "1s² (ground)", lw = 2,
    xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²",
    title = "Helium radial correlation")
    plot!(p2, r_ex, ρ_ex, label = "1s2s (excited)", lw = 2, ls = :dash)
display(p2)

HeP = HydrogenAtom(Z = 2)   # atomic units: Eₕ=1, a₀=1, mₑ=1, ℏ=1 (defaults)
E_hep_exact = Antique.E(HeP; n = 1)   # = -2.0 Ha

os_hep = Operators([1e15, 1.0], [+2, -1])
os_hep += "Kinetic"
os_hep += "Coulomb"

result_hep = solve_ECG(os_hep, 250; scale = 0.5, verbose = false)
ΔE_hep = result_hep.ground_state - E_hep_exact
println("  E (ECG)    = $(round(result_hep.ground_state; digits=8))")
println("  E (Antique)= $(round(E_hep_exact; digits=8))   error: $(round(ΔE_hep; sigdigits=3))")

r_hep, ρ_ecg = correlation_function(result_hep; rmax = 3.0, npoints = 300)
ρ_antique = [r^2 * Antique.R(HeP, r; n = 1, l = 0)^2 for r in r_hep]

p3 = plot(r_hep, ρ_ecg,
    label = "ECG", lw = 2,
    xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²",
    title = "He⁺ 1s radial density: ECG vs Antique.jl")
plot!(p3, r_hep, ρ_antique, label = "Antique (exact)", lw = 2, ls = :dash)
display(p3)
