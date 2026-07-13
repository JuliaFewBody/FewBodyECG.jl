# # Solver comparison on hydrogen
#
# Hydrogen has an analytical ground-state energy, so it is a compact benchmark
# for comparing solver methods.

using FewBodyECG
import Antique
using Plots
using QuasiMonteCarlo: GoldenSample, HaltonSample, SobolSample

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

exact = Antique.E(Antique.HydrogenAtom(Z = 1), n = 1)

function run_method(label, alg, ops, exact)
    sol = solve(ops, alg)
    println(label, ": E0 = ", sol.E₀, " Ha, Δ = ", sol.E₀ - exact)
    return sol
end

svm_halton = run_method(
    "SVM (Halton)",
    SVM(basis = 25, candidates = 20, scale = 1.0, sampler = HaltonSample()),
    ops,
    exact,
)
svm_sobol = run_method(
    "SVM (Sobol)",
    SVM(basis = 25, candidates = 20, scale = 1.0, sampler = SobolSample()),
    ops,
    exact,
)
svm_golden = run_method(
    "SVM (Golden)",
    SVM(basis = 25, candidates = 20, scale = 1.0, sampler = GoldenSample()),
    ops,
    exact,
)
refined = run_method(
    "SVM (Halton) → Refine",
    SVM(basis = 25, candidates = 20, scale = 1.0) →
        Refine(sweeps = 1, candidates = 20, scale = 1.0),
    ops,
    exact,
)
variational = run_method("Variational", Variational(basis = 12, scale = 1.0, maxiter = 100), ops, exact)
grown = run_method("GrowVariational", GrowVariational(basis = 8, candidates = 20, scale = 1.0), ops, exact)

p = plot(
    convergence(svm_halton)...;
    label = "SVM (Halton)",
    linewidth = 2,
    xlabel = "solver iteration",
    ylabel = "E₀ (Ha)",
    title = "Hydrogen solver convergence",
    legend = :bottomright,
)
for (label, sol) in [
    "SVM (Sobol)" => svm_sobol,
    "SVM (Golden)" => svm_golden,
    "SVM (Halton) → Refine" => refined,
    "Variational" => variational,
    "GrowVariational" => grown,
]
    plot!(p, convergence(sol)...; label, linewidth = 2)
end
hline!(p, [exact]; label = "exact", color = :black, linestyle = :dash)
p
