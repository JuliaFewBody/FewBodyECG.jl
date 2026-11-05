# # Leading relativistic corrections to helium 2¹P
# Stanke et al., [Phys. Rev. A 97, 012513 (2018)](https://doi.org/10.1103/PhysRevA.97.012513), Table I.
# Infinite nuclear mass; corrections are coefficients of α² in atomic units.

using FewBodyECG
using LinearAlgebra: Diagonal, diag, dot
using Plots

n_gaussians = 150
masses, charges = [1.0e15, 1.0, 1.0], [2.0, -1.0, -1.0]
H = Kinetic(masses) + Coulomb(charges)
symmetry = PermutationSymmetry(masses, [[1, 3, 2]])
sol = solve(H, SVM(n_gaussians; rank = 1, candidates = 50, widths = :wide) → Refine(4; candidates = 25, widths = :wide); symmetry)
S = FewBodyECG.build_overlap_matrix(sol.basis)
H₀ = FewBodyECG.build_hamiltonian_matrix(sol.basis, H)
D = Diagonal(1 ./ sqrt.(diag(S)))
E, C = FewBodyECG.solve_generalized_eigenproblem(D * H₀ * D, D * S * D)
C = D * C
c = C[:, 1]
matrix(op) = FewBodyECG.build_operator_matrix(sol.basis, op)

_, U = FewBodyECG.jacobi_transform(masses)
darwin = sum(matrix(ContactDensityOperator(U[i, :] - U[1, :]; coefficient = π)) for i in 2:3) -
    matrix(ContactDensityOperator(U[2, :] - U[3, :]; coefficient = π))
terms = [matrix(MassVelocityOperator(masses)), darwin, matrix(OrbitOrbitOperator(masses, charges))]
V = sum(terms)
corrections = [real(dot(c, M * c)) for M in terms]
labels = ["mass–velocity", "Darwin", "orbit–orbit"]

# These are separate perturbative contributions; spin–spin is not included.
# Direct contact and p⁴ estimates converge more slowly than the energy.
println("He 2¹P: E = ", round(E[1]; digits = 6), " Ha (reference −2.123843)")
println("ΔE/α² (Ha): ", join(("$name = $(round(value; digits = 6))" for (name, value) in zip(labels, corrections)), "; "))

# First-order state response in the same basis; α is the paper's convention.
α = 7.2973525698e-3
mixing = (C[:, 2:end]' * V * c) ./ (E[1] .- E[2:end])
δc = C[:, 2:end] * mixing
c_rel = c + α^2 * δc
c_rel /= sqrt(real(dot(c_rel, S * c_rel)))
ψ₀, ψ_rel = Wavefunction(sol.basis, c), Wavefunction(sol.basis, c_rel)

# Jacobi slice x₁=(1,0,0), x₂=(0,1,z), away from pair coalescence.
z = range(-12.0, 12.0; length = 601)
slice(ψ, z) = real(ψ([1.0 0.0 0.0; 0.0 1.0 z]))
amplitude₀, amplitude_rel = slice.(Ref(ψ₀), z), slice.(Ref(ψ_rel), z)

energy_plot = plot(convergence(sol); xlabel = "step", ylabel = "E (Ha)", label = "ECG", title = "Helium 2¹P")
hline!(energy_plot, [-2.123843086498]; label = "reference", linestyle = :dash)
wave_plot = plot(z, amplitude₀; label = "nonrelativistic", legend = :bottomright, ylabel = "ψ (a.u.)", title = "Jacobi wavefunction slice", linewidth = 2)
plot!(wave_plot, z, amplitude_rel; label = "with α² corrections", linestyle = :dash, linewidth = 2)
difference_plot = plot(z, 1.0e4 .* (amplitude_rel .- amplitude₀); label = false, ylabel = "10⁴(ψ_rel − ψ₀)", xlabel = "z (a.u.); x₁=(1,0,0), x₂=(0,1,z)", title = "Magnified change", linewidth = 2)
physics_plot = plot(
    energy_plot, wave_plot, difference_plot; layout = @layout([a [b; c]]), size = (1100, 650), left_margin = 6Plots.mm, bottom_margin = 7Plots.mm,
    titlefontsize = 12, guidefontsize = 10, legendfontsize = 9
)
physics_plot
