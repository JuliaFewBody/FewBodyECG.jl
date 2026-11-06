# # Helium S and P levels with gradient optimization
#
# Each level starts from an SVM basis with the right rank and electron-exchange
# symmetry; GVM then optimizes all widths, and for P states the global vectors,
# together. The references are the book's SVM energies (Suzuki–Varga,
# Table 8.1, p. 153) with K = 600 for ¹Sᵉ and K = 200 otherwise, against 24
# functions here; energies in atomic units.

using FewBodyECG
using OptimKit: LBFGS
using Plots

masses = [1.0e15, 1.0, 1.0]
H = Kinetic(masses) + Coulomb([2.0, -1.0, -1.0])
optimizer = LBFGS(; maxiter = 150, gradtol = 1.0e-7, verbosity = 0, ls_verbosity = 0)

## (level, rank, exchange sign, book energy)
levels = [
    ("¹Sᵉ", 0, 1, -2.9037243769),
    ("³Sᵉ", 0, -1, -2.1752291),
    ("¹Pᵒ", 1, 1, -2.1238423),
    ("³Pᵒ", 1, -1, -2.1331635),
]
panels = map(levels) do (label, rank, sign, reference)
    symmetry = PermutationSymmetry(masses, [[1, 3, 2]]; signs = [sign])
    sol = solve(H, SVM(24; rank, candidates = 30, widths = :wide) → GVM(; optimizer); symmetry)
    @info label E = energy(sol) reference Δ = energy(sol) - reference
    p = plot(energy_history(sol, 2); label = "GVM", xlabel = "iteration", ylabel = "E (Ha)", title = label)
    hline!(p, [reference]; label = "book SVM", linestyle = :dash)
end
plot(panels...; layout = (2, 2), size = (950, 650))
