# Hydrogen with SVM, comparing quasi-random samplers for the candidate widths.
using FewBodyECG
import Antique
using Plots
using QuasiMonteCarlo: GoldenSample, GridSample, HaltonSample, LatinHypercubeSample, LatticeRuleSample, SobolSample

ops = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])
exact = Antique.energy(Antique.HydrogenAtom(Z = 1), n = 1)

samplers = [HaltonSample(), SobolSample(), GoldenSample(), LatticeRuleSample(), LatinHypercubeSample(), GridSample()]
p = plot(; xlabel = "step", ylabel = "E₀ (Ha)", title = "Hydrogen SVM convergence by sampler")
for sampler in samplers
    sol = solve(ops, SVM(; basis = 25, candidates = 25, sampler))
    name = string(nameof(typeof(sampler)))
    @info name E₀ = sol.E₀ Δ = sol.E₀ - exact
    plot!(p, convergence(sol); label = name)
end
hline!(p, [exact]; label = "exact", color = :black, linestyle = :dash)
