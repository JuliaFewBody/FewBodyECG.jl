using FewBodyECG
using LinearAlgebra
using Plots
using QuasiMonteCarlo

masses = [1.0e15, 1.0, 1.0]

os = Operators(masses, [+1, -1, -1])   # proton, e₁, e₂
os += "Kinetic"
os += "Coulomb"

result = solve_ECG(os, 250, scale = 1.0, verbose=false)

E_exact = -0.527751016523
ΔE = abs(result.ground_state - E_exact)
@info "Energy difference" ΔE

n_conv, E_conv = convergence(result)
p1 = plot(n_conv, E_conv,
    xlabel = "Basis size", ylabel = "E (Ha)",
    label = "Ground state", lw = 2,
    title = "Hydrogen anion convergence")
display(p1)

r_grid, ρ = correlation_function(result; rmax = 10.0)
p2 = plot(r_grid, ρ,
    xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²",
    label = "Hydrogen anion", lw = 2,
    title = "Hydrogen radial correlation")
display(p2)
