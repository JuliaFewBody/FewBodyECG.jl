# # Leading relativistic corrections to helium 2¹P
#
# Stanke et al., [Phys. Rev. A 97, 012513 (2018)](https://doi.org/10.1103/PhysRevA.97.012513),
# Table I. The nucleus is infinitely heavy, and the corrections are the
# coefficients of α² in atomic units, evaluated perturbatively on the
# nonrelativistic state.

using FewBodyECG
using FewBodyECG: build_operator_matrix, build_overlap_matrix, jacobi_transform
using LinearAlgebra
using Plots

masses, charges = [1.0e15, 1.0, 1.0], [2.0, -1.0, -1.0]
H = Kinetic(masses) + Coulomb(charges)

## 2¹P: Rank1 (L = 1) Gaussians, even under electron exchange.
method = SVM(150; rank = 1, candidates = 50, widths = :wide) → Refine(4; candidates = 25, widths = :wide)
sol = solve(H, method; symmetry = PermutationSymmetry(masses, [[1, 3, 2]]))
reference = -2.123843086498

E, C = sol.E, sol.coefficients
c = C[:, 1]
matrix(op) = build_operator_matrix(sol.basis, op)

## Darwin term π δ³(r₁ₙ) + π δ³(r₂ₙ) − π δ³(r₁₂).
_, U = jacobi_transform(masses)
contact(i, j) = matrix(ContactDensityOperator(U[i, :] - U[j, :]; coefficient = π))
corrections = (
    mass_velocity = matrix(MassVelocityOperator(masses)),
    darwin = contact(2, 1) + contact(3, 1) - contact(2, 3),
    orbit_orbit = matrix(OrbitOrbitOperator(masses, charges)),
)
## Separate perturbative contributions; spin–spin is not included.  Contact and
## p⁴ estimates converge more slowly than the energy.
ΔE = map(M -> dot(c, M * c), corrections)
@info "He 2¹P; corrections in units of α² Ha" E = E[1] reference ΔE...

## First-order change of the state in the same basis, with α as in the paper.
α = 7.2973525698e-3
V = sum(corrections)
δc = C[:, 2:end] * ((C[:, 2:end]' * V * c) ./ (E[1] .- E[2:end]))
c_rel = c + α^2 * δc
c_rel /= sqrt(dot(c_rel, build_overlap_matrix(sol.basis) * c_rel))

## Slice through Jacobi space: x₁ = (1, 0, 0), x₂ = (0, 1, z), away from coalescence.
z = range(-12.0, 12.0; length = 601)
slice(coefficients) = [Wavefunction(sol.basis, coefficients)([1.0 0.0 0.0; 0.0 1.0 zᵢ]) for zᵢ in z]
ψ₀, ψ_rel = slice(c), slice(c_rel)

energy_plot = plot(sol, reference; title = "Helium 2¹P")
wave_plot = plot(
    z, [ψ₀ ψ_rel]; label = ["nonrelativistic" "with α² corrections"], linestyle = [:solid :dash],
    ylabel = "ψ (a.u.)", title = "Jacobi wavefunction slice",
)
difference_plot = plot(
    z, 1.0e4 .* (ψ_rel .- ψ₀); label = false,
    xlabel = "z (a.u.)", ylabel = "10⁴(ψ_rel − ψ₀)", title = "Magnified change",
)
plot(energy_plot, wave_plot, difference_plot; layout = @layout([a [b; c]]), size = (1100, 650))
