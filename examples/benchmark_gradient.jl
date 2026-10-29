using FewBodyECG
import Antique
using OptimKit: ConjugateGradient, GradientDescent, LBFGS
using Plots

ops = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])

exact = Antique.energy(Antique.HydrogenAtom(Z = 1), n = 1)

function run_method(label, alg)
    sol = solve(ops, alg)
    println(label, ": E0 = ", sol.E₀, " Ha, Δ = ", sol.E₀ - exact)
    return sol
end

basis = 10
methods = [
    "GVM + gradient descent" => GVM(
        basis;
        optimizer = GradientDescent(; maxiter = 100, gradtol = 1.0e-6),
    ),
    "GVM + conjugate gradient" => GVM(
        basis;
        optimizer = ConjugateGradient(; maxiter = 100, gradtol = 1.0e-6),
    ),
    "GVM + L-BFGS" => GVM(
        basis;
        optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-6),
    ),
    "DynamicGVM + conjugate gradient" => DynamicGVM(
        basis;
        candidates = 10,
        optimizer = ConjugateGradient(; maxiter = 50, gradtol = 1.0e-6),
    ),
    "SVM → GVM + L-BFGS" =>
        SVM(basis; candidates = 10) →
        GVM(optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-6)),
]
solutions = [label => run_method(label, alg) for (label, alg) in methods]

label, sol = first(solutions)
p = plot(
    convergence(sol)...;
    label,
    linewidth = 1.5,
    xlabel = "solver iteration",
    ylabel = "E₀ (Ha)",
    title = "Hydrogen gradient-solver convergence",
    legend = :topright,
)
for (label, sol) in Iterators.drop(solutions, 1)
    plot!(p, convergence(sol)...; label, linewidth = 1.5)
end
hline!(p, [exact]; label = "exact", color = :black, linestyle = :dash)
p
