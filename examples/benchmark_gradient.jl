# Hydrogen with the gradient methods and several OptimKit optimizers.
using FewBodyECG
import Antique
using OptimKit: ConjugateGradient, GradientDescent, LBFGS
using Plots

ops = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])
exact = Antique.energy(Antique.HydrogenAtom(Z = 1), n = 1)

methods = [
    "GVM + gradient descent" => GVM(10; optimizer = GradientDescent(; maxiter = 100, gradtol = 1.0e-6)),
    "GVM + conjugate gradient" => GVM(10; optimizer = ConjugateGradient(; maxiter = 100, gradtol = 1.0e-6)),
    "GVM + L-BFGS" => GVM(10; optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-6)),
    "DynamicGVM + conjugate gradient" =>
        DynamicGVM(10; candidates = 10, optimizer = ConjugateGradient(; maxiter = 50, gradtol = 1.0e-6)),
    "SVM → GVM + L-BFGS" => SVM(10; candidates = 10) → GVM(optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-6)),
]

p = plot(; xlabel = "step", ylabel = "E₀ (Ha)", title = "Hydrogen gradient-solver convergence")
for (label, method) in methods
    sol = solve(ops, method)
    @info label E₀ = sol.E₀ Δ = sol.E₀ - exact
    plot!(p, convergence(sol); label)
end
hline!(p, [exact]; label = "exact", color = :black, linestyle = :dash)
