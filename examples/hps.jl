# # Positronium hydride (HPs⁺)
#
# A four-body Coulomb system consisting of a fixed proton, two electrons, and
# a positron.  The particle order is proton, electron, electron, positron.
# This rank-0 example targets the symmetric spatial ground sector of the two
# identical electrons.  The reference is the infinite-proton-mass SVM value
# from Suzuki–Varga Table 8.6 (K = 1200); this run uses a smaller basis as a
# practical example rather than claiming to reproduce K = 1200.

using FewBodyECG
using Plots

masses = [1.0e15, 1.0, 1.0, 1.0]
charges = [+1.0, -1.0, -1.0, +1.0]

ops = Operators(masses, charges)
ops += "Kinetic"
ops += "Coulomb"

hps_ref = -0.7891964
sol = solve(ops, GrowVariational(basis = 50, candidates = 20, scale = 1.0))

println("HPs⁺ E₀ = ", sol.E₀, " Ha")
println("reference (SVM, K = 1200) = ", hps_ref, " Ha   Δ = ", sol.E₀ - hps_ref)
println("variational upper bound respected: ", sol.E₀ ≥ hps_ref)

plot(sol, hps_ref; title = "Positronium-hydride convergence")
