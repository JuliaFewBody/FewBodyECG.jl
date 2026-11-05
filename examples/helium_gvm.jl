# Helium S/P levels: Suzuki–Varga, Table 8.1, p. 153, in atomic units.
# The reference values are the book's finite-basis SVM results.
using FewBodyECG
using OptimKit: LBFGS
using Plots

n_gaussians = 24
masses = [1.0e15, 1.0, 1.0]
H = Kinetic(masses) + Coulomb([2.0, -1.0, -1.0])
levels = [
    ("¹Sᵉ", 0, 1, -2.9037243769),
    ("³Sᵉ", 0, -1, -2.1752291),
    ("¹Pᵒ", 1, 1, -2.1238423),
    ("³Pᵒ", 1, -1, -2.1331635),
]
solutions = map(levels) do (label, rank, sign, reference)
    symmetry = PermutationSymmetry(masses, [[1, 3, 2]]; signs = [sign])
    sol = solve(
        H, SVM(n_gaussians; rank, candidates = 30, widths = :wide) →
            GVM(; optimizer = LBFGS(; maxiter = 150, gradtol = 1.0e-7, verbosity = 0, ls_verbosity = 0));
        symmetry
    )
    println(label, ": E = ", round(energy(sol); digits = 9), " Ha; book SVM = ", reference)
    sol
end

panels = map(levels, solutions) do (label, _, _, reference), sol
    low = min(reference, minimum(energy_history(sol, 2)))
    high = max(reference, last(sol.stages[1].history))
    margin = max(1.0e-8, 0.05 * (high - low))
    p = plot(
        convergence(sol); title = "$label: GVM after SVM", xlabel = "step", ylabel = "E (Ha)", label = "ECG", linewidth = 2,
        xlims = (length(sol.stages[1].history), length(energy_history(sol))), ylims = (low - margin, high + margin)
    )
    hline!(p, [reference]; label = "book SVM", linestyle = :dash)
    vline!(p, [length(sol.stages[1].history)]; label = "GVM starts", linestyle = :dot)
end
plot(panels...; layout = (2, 2), size = (950, 650), legend = :topright)
