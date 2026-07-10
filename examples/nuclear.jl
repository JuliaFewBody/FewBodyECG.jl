# # Nuclear few-body with Gaussian potentials
#
# Realistic soft-core nucleon–nucleon potentials are **sums of Gaussians**, so
# [`GaussianOperator`](@ref) represents them exactly — opening the nuclear domain
# with no new machinery.  We work in nuclear units (energies in MeV, lengths in
# fm); a particle's package "mass" is `mc² / (ħc)²` so that `ħ²/2m → (ħc)²/2mc²`
# with `ħc = 197.327 MeV·fm` (giving `ħ²/2mₙ ≈ 20.7 MeV·fm²`).
#
# Two classic central-force benchmarks:
# * the **deuteron** with the Minnesota potential (which reproduces the two-body
#   binding), and
# * the **triton** with the Volkov V1 potential (a standard three-body central
#   benchmark).

using FewBodyECG

const ħc = 197.3269804
mpkg(mc²) = mc² / ħc^2
mp, mn = mpkg(938.272), mpkg(939.565)

## Deuteron — Minnesota triplet-even central potential
## V(r) = 200 e^{-1.487 r²} − 178 e^{-0.639 r²}  (MeV, r in fm)
deut = Operators([mp, mn])
deut += "Kinetic"
deut += ("Gaussian", 1, 2, 200.0, 1.487)     # repulsive core
deut += ("Gaussian", 1, 2, -178.0, 0.639)    # triplet attraction

sol_d = solve(deut, SVM(basis = 40, candidates = 25, scale = 3.0))
Ed_ref = -2.202
println("deuteron  E = ", round(sol_d.E₀, digits = 4), " MeV   (Minnesota ", Ed_ref, ")")

## Triton — Volkov V1 central potential on all three pairs
## V(r) = 144.86 e^{-(r/0.82)²} − 83.34 e^{-(r/1.60)²}  (MeV, r in fm)
γR, γA = 1 / 0.82^2, 1 / 1.6^2
trit = Operators([mn, mn, mp])
trit += "Kinetic"
trit += ("Gaussian", 1, 2, 144.86, γR); trit += ("Gaussian", 1, 2, -83.34, γA)
trit += ("Gaussian", 1, 3, 144.86, γR); trit += ("Gaussian", 1, 3, -83.34, γA)
trit += ("Gaussian", 2, 3, 144.86, γR); trit += ("Gaussian", 2, 3, -83.34, γA)

sol_t = solve(trit, SVM(basis = 120, candidates = 30, scale = 3.5))
Et_ref = -8.46
println("triton    E = ", round(sol_t.E₀, digits = 4), " MeV   (Volkov V1 ", Et_ref, ")")

# The triton's central-force binding is ~4× the deuteron's — the classic
# few-body sequence.  (The physical triton also needs spin-isospin and tensor
# forces, which a purely central model omits.)
