using FewBodyECG
import Antique
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

exact = Antique.E(Antique.HydrogenAtom(Z = 1), n = 1)

function run_method(label, alg, ops, exact)
    sol = solve(ops, alg)
    println(label, ": E0 = ", sol.E₀, " Ha, Δ = ", sol.E₀ - exact)
    return sol
end

basis = 10
scale = 1.0
cold_gvm = run_method(
    "GVM (cold start)", GVM(basis = basis, scale = scale, maxiter = 100), ops, exact
)
dynamic_gvm = run_method(
    "DynamicGVM",
    DynamicGVM(
        basis = basis, candidates = 10, scale = scale, maxiter_step = 50
    ),
    ops,
    exact,
)
warm_gvm = run_method(
    "SVM → GVM",
    SVM(basis = basis, candidates = 10, scale = scale) → GVM(maxiter = 100),
    ops,
    exact,
)

p = plot(
    convergence(cold_gvm)...;
    label = "GVM (cold start)",
    linewidth = 1.5,
    xlabel = "solver iteration",
    ylabel = "E₀ (Ha)",
    title = "Hydrogen gradient-solver convergence",
    legend = :topright,
)
for (label, sol) in ["DynamicGVM" => dynamic_gvm, "SVM → GVM" => warm_gvm]
    plot!(p, convergence(sol)...; label, linewidth = 1.5)
end
hline!(p, [exact]; label = "exact", color = :black, linestyle = :dash)
p
