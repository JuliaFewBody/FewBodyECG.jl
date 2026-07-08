# # Stochastic to variational workflow
#
# A practical workflow is to explore cheaply with SVM, refine the sampled basis,
# then optimize all Gaussian parameters jointly.

using FewBodyECG
using Plots

mₚ = 1836.15267343
ops = Operators([mₚ, mₚ, 1.0], [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

workflow = SVM(basis = 12, candidates = 15, scale = 1.0) →
    Refine(sweeps = 1, candidates = 15, scale = 1.0) →
    Variational(basis = 12, maxiter = 30)

sol = solve(ops, workflow)
sol

for (i, stage) in enumerate(sol.stages)
    println("stage ", i, ": ", stage.method, " => ", last(stage.energies), " Ha")
end

plot(sol, -0.597139)
