# # Why the deuteron binds: spin and isospin exchange
#
# The Minnesota potential (Suzuki–Varga, Table 11.1, p. 217) is a sum of
# Gaussians with Wigner, Majorana, Bartlett and Heisenberg exchange parts,
# `V = Σₖ Vₖ exp(-γₖr²) (Wₖ + Mₖ Pʳ + Bₖ Pσ - Hₖ Pτ)`. In an S-wave it binds
# the S = 1, T = 0 channel (the deuteron, −2.202 MeV in Table 11.9, p. 237) but
# not S = 0, T = 1.

using FewBodyECG
using FewBodyECG: build_operator_matrix, build_overlap_matrix, solve_generalized_eigenproblem, jacobi_transform
using FewBodyECG: up, down, proton, neutron
using LinearAlgebra
using Plots

masses = fill(1 / 41.47, 2)   # ℏ²/m = 41.47 MeV fm²
_, U = jacobi_transform(masses)
w = U' * [1.0, -1.0]          # Jacobi weight of the proton–neutron separation

## Radial Gaussians in the pair distance, each times the four product states
## with M_S = M_T = 0.  The innermost loop runs over isospin.
widths = exp10.(range(-4, 2, length = 24))
orbitals = [Rank0Gaussian(α * (w * w'), zeros(1, 3)) for α in widths]
spins = (SpinState([up, down]), SpinState([down, up]))
isospins = (IsospinState([proton, neutron]), IsospinState([neutron, proton]))
basis = BasisSet([SpinGaussian(g, s; isospin = t) for g in orbitals for s in spins for t in isospins])

## Minnesota rows (Vₖ [MeV], γₖ [fm⁻²], Wₖ, Mₖ, Bₖ, Hₖ) with exchange parameter u = 1.
minnesota = [
    (200.0, 1.487, 0.5, 0.5, 0.0, 0.0),
    (-178.0, 0.639, 0.25, 0.25, 0.25, 0.25),
    (-91.85, 0.465, 0.25, 0.25, -0.25, -0.25),
]
Pʳ, Pσ, Pτ = PermutationOperator(masses, [2, 1]), SpinExchangeOperator(1, 2), IsospinExchangeOperator(1, 2)
matrix(op) = build_operator_matrix(basis, op)
potential = sum(minnesota) do (V, γ, W, M, B, H)
    radial = GaussianOperator(V, γ, w)
    W * matrix(radial) + M * matrix(radial * Pʳ) + B * matrix(radial * Pσ) - H * matrix(radial * Pτ)
end
hamiltonian = matrix(KineticOperator(masses)) + potential
overlap = build_overlap_matrix(basis)

## Coupled channel (|↑↓⟩ ± |↓↑⟩)/√2 ⊗ (|pn⟩ ± |np⟩)/√2 on the first k radial
## functions: + for S = 1 or T = 1, − for S = 0 or T = 0.
channel(S, T, k) = kron(Matrix(1.0I, length(orbitals), k), kron([1, 2S - 1], [1, 2T - 1]) / 2)

## Lowest channel energy and its radial coefficients.  Rescaling to a unit
## diagonal keeps the widely spread widths well conditioned.
function ground_state(S, T, k)
    C = channel(S, T, k)
    Hc, Sc = C' * hamiltonian * C, C' * overlap * C
    D = Diagonal(1 ./ sqrt.(diag(Sc)))
    cond(D * Sc * D) < 1.0e12 || throw(ArgumentError("radial basis with $k functions is too ill-conditioned"))
    E, c = solve_generalized_eigenproblem(D * Hc * D, D * Sc * D)
    return first(E), D * c[:, 1]
end

sizes = 4:2:24
triplet = [first(ground_state(1, 0, k)) for k in sizes]
singlet = [first(ground_state(0, 1, k)) for k in sizes]   # positive levels approximate the continuum
E_d, c_d = ground_state(1, 0, length(orbitals))

## The spin-isospin states are normalized, so `c_d` also normalizes the radial part.
radial_basis = BasisSet(orbitals)
rms = sqrt(dot(c_d, build_operator_matrix(radial_basis, OscillatorOperator(1.0, w)) * c_d))
@info "Deuteron (Minnesota)" E = E_d rms_separation_fm = rms reference = -2.202

## u(r) = √(4π) r ψ(r) in physical fm; the Jacobi coordinate is x = |w| r.
ψ = Wavefunction(radial_basis, c_d)
radii = range(0, 20, length = 600)
scale = abs(only(w))
u = [sqrt(4π) * r * ψ([r / scale]) / scale^(3 / 2) for r in radii]
u .*= sign(u[argmax(abs.(u))])   # fix the arbitrary sign of the eigenvector

triplet_potential(r) = 200exp(-1.487r^2) - 178exp(-0.639r^2)
singlet_potential(r) = 200exp(-1.487r^2) - 91.85exp(-0.465r^2)
r = range(0, 6, length = 400)
potentials = plot(
    r, [triplet_potential.(r) singlet_potential.(r)]; label = ["S=1, T=0" "S=0, T=1"],
    xlabel = "r (fm)", ylabel = "V(r) (MeV)", title = "Channel potentials",
)
energies = plot(
    sizes, [triplet singlet]; label = ["triplet" "singlet"], marker = :circle,
    xlabel = "radial Gaussians", ylabel = "E (MeV)", title = "Basis convergence",
)
hline!(energies, [0.0]; label = "p + n threshold", color = :gray, linestyle = :dash)
hline!(energies, [-2.202]; label = "book", color = :black, linestyle = :dot)
probability = plot(
    radii, abs2.(u); label = "deuteron", fillrange = 0, fillalpha = 0.15,
    xlabel = "r (fm)", ylabel = "u(r)² (fm⁻¹)", title = "Separation probability",
)
vline!(probability, [rms]; label = "rms = $(round(rms; digits = 2)) fm", color = :black, linestyle = :dash)
plot(potentials, energies, probability; layout = (1, 3), size = (1200, 380))
