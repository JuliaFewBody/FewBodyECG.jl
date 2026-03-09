using FewBodyECG, LinearAlgebra
using QuasiMonteCarlo
using Plots
import FewBodyECG: default_scale, convergence

masses = [1.0, 1.0, 1.0]

os = Operators(masses, [+1, -1, -1])   # e⁺, e⁻, e⁻
os += "Kinetic"
os += "Coulomb"

scale = default_scale(masses)

# Ps⁻ has only one bound state; for excited-state examples see Helium.jl.
result = solve_ECG(os, 300, sampler = SobolSample(); scale = scale, verbose=false, state = 1)
println("E ≈ ", result.ground_state)

n_conv, E_conv = convergence(result)
p1 = plot(n_conv, E_conv,
    xlabel = "Basis size", ylabel = "E (Ha)",
    label = "Ground state", lw = 2,
    title = "Positronium convergence")
display(p1)

r_grid, ρ = correlation_function(result; rmax = 15.0)
p2 = plot(r_grid, ρ,
    xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²",
    label = "Positronium", lw = 2,
    title = "Positronium radial correlation")
display(p2)
