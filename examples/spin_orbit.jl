# # Spin-orbit doublet — the complex Hermitian path
#
# Spin-dependent interactions make the Hamiltonian **complex Hermitian**, and
# FewBodyECG assembles and diagonalises it directly.  Here two spin-½ particles
# share a p-wave-like spatial manifold — Gaussians shifted along `x`, `y`, `z`,
# each dressed with the spin-triplet state `|↑↑⟩` — bound by a central Gaussian
# attraction.  A [`SpinOrbit`](@ref) term couples the relative orbital motion to
# the total spin `S₁ + S₂`.
#
# With the coupling off, the two upper orbital states are degenerate.  Turning it
# on lifts the degeneracy: the `m_ℓ = ±1` combinations shift in opposite
# directions, giving a doublet whose splitting grows **linearly** with the
# coupling strength — a model fine-structure splitting.

using FewBodyECG
using LinearAlgebra
using Plots
using FewBodyECG: build_hamiltonian_matrix, build_overlap_matrix, solve_generalized_eigenproblem, up

masses = [1.0, 1.0]
central = Kinetic(masses) - 4.0 * Gaussian(1, 2; exponent = 0.4)   # central binding well
H(λ) = central + λ * SpinOrbit(1, 2; exponent = 0.4)                # spin-orbit strength λ

## p-wave-like manifold: unit-shifted Gaussians along x, y, z with |↑↑⟩ spin
shift(dir) = (v = zeros(1, 3); v[1, dir] = 0.8; v)
basis = BasisSet(
    [
        SpinGaussian(Rank0Gaussian([0.4;;], shift(d)), SpinState([up, up])) for d in 1:3
    ]
)

H₀ = build_hamiltonian_matrix(basis, central)                     # real, SO-free
H₁ = build_hamiltonian_matrix(basis, H(1.0))
S = build_overlap_matrix(basis)

println("H(λ = 0) eltype = ", eltype(H₀), "  (spin-orbit-free, real)")
println("H(λ = 1) eltype = ", eltype(H₁), "  Hermitian? ", isapprox(H₁, H₁'))

## Sweep the spin-orbit strength λ and collect the (real) spectrum
λs = range(0, 2, length = 21)
levels = map(λs) do λ
    E, _ = solve_generalized_eigenproblem(build_hamiltonian_matrix(basis, H(λ)), S)
    sort(real.(E))
end
spectrum = reduce(hcat, levels)'          # rows = λ, columns = levels

## The upper two levels form the spin-orbit doublet; its gap is linear in λ.
gap = spectrum[:, 3] .- spectrum[:, 2]
println("doublet gap at λ = 0, 1, 2: ", round.(gap[[1, 11, 21]], digits = 4))

plot(
    λs, spectrum;
    xlabel = "spin-orbit coupling λ", ylabel = "E (Ha)",
    label = ["ground" "doublet⁻" "doublet⁺"], linewidth = 2,
    title = "Spin-orbit doublet splitting",
)

# ## Convergence of the manifold
#
# The three-function manifold above is the minimal one.  Enriching each Cartesian
# direction with a geometric ladder of widths gives a sequence of **nested**
# bases, so the ground-state energy (here at fixed coupling `λ = 1`) descends
# monotonically toward the variational limit.

αs = exp10.(range(-0.8, 0.6, length = 8))     # fixed ladder ⇒ nested bases
sg(α, d) = SpinGaussian(Rank0Gaussian([α;;], shift(d)), SpinState([up, up]))
λ = 1.0

sizes = Int[]
ground = Float64[]
for k in 1:length(αs)
    b = BasisSet([sg(α, d) for α in αs[1:k] for d in 1:3])
    E, _ = solve_generalized_eigenproblem(build_hamiltonian_matrix(b, H(λ)), build_overlap_matrix(b))
    push!(sizes, 3k)
    push!(ground, minimum(real.(E)))
end
println("converged ground-state energy: ", ground[end], " Ha")

plot(
    sizes, ground;
    xlabel = "basis size", ylabel = "E₀ (Ha)", label = "ground state (λ = 1)",
    marker = :circle, linewidth = 2, title = "Basis-size convergence",
)
