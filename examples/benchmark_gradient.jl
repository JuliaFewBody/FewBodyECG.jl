using FewBodyECG
import Antique
using OptimKit: ConjugateGradient, GradientDescent, LBFGS
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

exact = Antique.E(Antique.HydrogenAtom(Z = 1), n = 1)

function run_method(label, alg)
    sol = solve(ops, alg)
    println(label, ": E0 = ", sol.E₀, " Ha, Δ = ", sol.E₀ - exact)
    return sol
end

basis = 10
scale = 1.0
methods = [
    "GVM + gradient descent" => GVM(
        basis;
        scale,
        optimizer = GradientDescent(; maxiter = 100, gradtol = 1.0e-6),
    ),
    "GVM + conjugate gradient" => GVM(
        basis;
        scale,
        optimizer = ConjugateGradient(; maxiter = 100, gradtol = 1.0e-6),
    ),
    "GVM + L-BFGS" => GVM(
        basis;
        scale,
        optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-6),
    ),
    "DynamicGVM + conjugate gradient" => DynamicGVM(
        basis;
        candidates = 10,
        scale,
        optimizer = ConjugateGradient(; maxiter = 50, gradtol = 1.0e-6),
    ),
    "SVM → GVM + L-BFGS" =>
        SVM(basis; candidates = 10, scale) →
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
