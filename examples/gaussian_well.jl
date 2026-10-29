using FewBodyECG
using Plots

γ = 1.0
V₀ = 5.0

ops = Kinetic([1.0e15, 1.0]) - V₀ * Gaussian(1, 2; exponent = γ)

sol = solve(ops, SVM(basis = 30, candidates = 20))
println("Gaussian well E0 = ", sol.E₀, " Ha")

depths = 1.0:1.0:8.0
scan = map(depths) do depth
    o = Kinetic([1.0e15, 1.0]) - depth * Gaussian(1, 2; exponent = γ)
    solve(o, SVM(basis = 15, candidates = 10)).E₀
end

p = plot(depths, scan; xlabel = "well depth V0 (Ha)", ylabel = "E0 (Ha)", label = "scan")
hline!(p, [0.0]; linestyle = :dash, label = "continuum")
p

plot(wavefunction(sol); coord = 1, rmax = 6.0)
