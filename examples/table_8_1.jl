# # Suzuki–Varga Table 8.1
#
# Three-body S and P states, each solved with K = 24 functions as
# SVM → Refine → GVM and compared with the book's SVM energies (Table 8.1,
# p. 153), which use K = 200–600. The remaining differences are mostly basis
# size; they are largest for tdμ, which has no exchange symmetry to exploit.
# Energies are in Hartree and masses in electron masses; an infinitely heavy
# nucleus is approximated by 10¹⁵ mₑ.

using FewBodyECG
using OptimKit: LBFGS

mμ, md, mt = 206.7686, 3670.481, 5496.918
ps = (masses = ones(3), charges = [1.0, -1.0, -1.0], exchange = [1, 3, 2])
atom(Z) = (masses = [1.0e15, 1.0, 1.0], charges = [Z, -1.0, -1.0], exchange = [1, 3, 2])
ttμ = (masses = [mt, mt, mμ], charges = [1.0, 1.0, -1.0], exchange = [2, 1, 3])
tdμ(m_t, m_d, m_μ) = (masses = [m_t, m_d, m_μ], charges = [1.0, 1.0, -1.0], exchange = nothing)

## (system, particles, state, rank, exchange sign, book energy, book K)
rows = [
    ("Ps⁻", ps, "¹Sᵉ", 0, 1, -0.262005070226, 600),
    ("∞H⁻", atom(1.0), "¹Sᵉ", 0, 1, -0.527710163, 600),   # as printed in the book
    ("ttμ", ttμ, "¹Sᵉ", 0, 1, -112.973, 200),
    ("ttμ", ttμ, "¹Pᵒ", 1, -1, -110.2621, 200),
    ("tdμ", tdμ(mt, md, mμ), "¹Sᵉ", 0, 1, -111.36444, 200),
    ("tdμ", tdμ(5496.92158, 3670.483014, 206.768262), "¹Pᵒ", 1, 1, -108.17923, 200),   # the book's mass set for this row
    ("∞He", atom(2.0), "¹Sᵉ", 0, 1, -2.9037243769, 600),
    ("∞He", atom(2.0), "³Sᵉ", 0, -1, -2.1752291, 200),
    ("∞He", atom(2.0), "¹Pᵒ", 1, 1, -2.1238423, 200),
    ("∞He", atom(2.0), "³Pᵒ", 1, -1, -2.1331635, 200),
    ("∞Li⁺", atom(3.0), "¹Sᵉ", 0, 1, -7.279913, 300),
]
K = 24

optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-8, verbosity = 0, ls_verbosity = 0)
for (system, particles, state, rank, sign, book, book_K) in rows
    (; masses, charges, exchange) = particles
    symmetry = exchange === nothing ? nothing : PermutationSymmetry(masses, [exchange]; signs = [sign])
    method = SVM(K; rank, candidates = 40) → Refine(2; candidates = 30) → GVM(; optimizer)
    sol = solve(Kinetic(masses) + Coulomb(charges), method; symmetry)
    @info "$system $state" E = energy(sol) K book book_K Δ = energy(sol) - book
end
