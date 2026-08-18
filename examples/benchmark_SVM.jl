using FewBodyECG
import Antique
using Plots
using QuasiMonteCarlo: GoldenSample, HaltonSample, SobolSample, LatticeRuleSample, LatinHypercubeSample, GridSample

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

exact = Antique.E(Antique.HydrogenAtom(Z = 1), n = 1)

function run_method(label, alg, ops, exact)
    sol = solve(ops, alg)
    println(label, ": E0 = ", sol.E₀, " Ha, Δ = ", sol.E₀ - exact)
    return sol
end

nsamples = 25
scale = 1.0
svm_halton = run_method(
    "SVM (Halton)",
    SVM(basis = 25, candidates = nsamples, scale = scale, sampler = HaltonSample()),
    ops,
    exact,
)
svm_sobol = run_method(
    "SVM (Sobol)",
    SVM(basis = 25, candidates = nsamples, scale = scale, sampler = SobolSample()),
    ops,
    exact,
)
svm_golden = run_method(
    "SVM (Golden)",
    SVM(basis = 25, candidates = nsamples, scale = scale, sampler = GoldenSample()),
    ops,
    exact,
)

svm_lattice = run_method(
    "SVM (Lattice)",
    SVM(basis = 25, candidates = nsamples, scale = scale, sampler = LatticeRuleSample()),
    ops,
    exact,
)

svm_hypercube = run_method(
    "SVM (Latin Hypercube)",
    SVM(basis = 25, candidates = nsamples, scale = scale, sampler = LatinHypercubeSample()),
    ops,
    exact,
)

svm_grid = run_method(
    "SVM (Grid sampling)",
    SVM(basis = 25, candidates = nsamples, scale = scale, sampler = GridSample()),
    ops,
    exact,
)

p = plot(
    convergence(svm_halton)...;
    label = "SVM (Halton)",
    linewidth = 1.5,
    xlabel = "solver iteration",
    ylabel = "E₀ (Ha)",
    title = "Hydrogen solver convergence",
    legend = :topright,
)
for (label, sol) in [
        "SVM (Sobol)" => svm_sobol,
        "SVM (Golden)" => svm_golden,
        "SVM (LatticeRuleSample)" => svm_lattice,
        "SVM (Latin Hypercube)" => svm_hypercube,
        "SVM (Grid sample)" => svm_grid,
    ]
    plot!(p, convergence(sol)...; label, linewidth = 1.5)
end
hline!(p, [exact]; label = "exact", color = :black, linestyle = :dash)
p
