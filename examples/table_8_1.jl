# Suzuki–Varga, Table 8.1, p. 153. Energies are in Hartree, masses in mₑ.
# Book energies are copied from the SVM column, including ∞H⁻ as printed.
using FewBodyECG
using OptimKit: LBFGS
using Printf

n_gaussians = 24

set_a = (mu = 206.7686, d = 3670.481, t = 5496.918)
set_b = (mu = 206.768262, d = 3670.483014, t = 5496.92158)
swap_electrons, swap_nuclei = [1, 3, 2], [2, 1, 3]

# Electronic atoms use infinite nuclear mass, approximated by 10¹⁵ mₑ.
# Each row gives system, state, masses, charges, rank, exchange, sign, book energy.
systems = [
    ("Ps⁻", "¹Sᵉ", [1.0, 1.0, 1.0], [1.0, -1.0, -1.0], 0, swap_electrons, 1, -0.262005070226),
    ("∞H⁻", "¹Sᵉ", [1.0e15, 1.0, 1.0], [1.0, -1.0, -1.0], 0, swap_electrons, 1, -0.527710163),
    ("ttμ", "¹Sᵉ", [set_a.t, set_a.t, set_a.mu], [1.0, 1.0, -1.0], 0, swap_nuclei, 1, -112.973),
    ("ttμ", "¹Pᵒ", [set_a.t, set_a.t, set_a.mu], [1.0, 1.0, -1.0], 1, swap_nuclei, -1, -110.2621),
    ("tdμ", "¹Sᵉ", [set_a.t, set_a.d, set_a.mu], [1.0, 1.0, -1.0], 0, nothing, 1, -111.36444),
    ("tdμ", "¹Pᵒ", [set_b.t, set_b.d, set_b.mu], [1.0, 1.0, -1.0], 1, nothing, 1, -108.17923),
    ("∞He", "¹Sᵉ", [1.0e15, 1.0, 1.0], [2.0, -1.0, -1.0], 0, swap_electrons, 1, -2.9037243769),
    ("∞He", "³Sᵉ", [1.0e15, 1.0, 1.0], [2.0, -1.0, -1.0], 0, swap_electrons, -1, -2.1752291),
    ("∞He", "¹Pᵒ", [1.0e15, 1.0, 1.0], [2.0, -1.0, -1.0], 1, swap_electrons, 1, -2.1238423),
    ("∞He", "³Pᵒ", [1.0e15, 1.0, 1.0], [2.0, -1.0, -1.0], 1, swap_electrons, -1, -2.1331635),
    ("∞Li⁺", "¹Sᵉ", [1.0e15, 1.0, 1.0], [3.0, -1.0, -1.0], 0, swap_electrons, 1, -7.279913),
]

@printf("%-8s %-6s %5s %19s %19s %13s\n", "System", "State", "N", "E (Ha)", "Book SVM (Ha)", "E − book (Ha)")
flush(stdout)
for (system, state, masses, charges, rank, exchange, sign, reference) in systems
    symmetry = exchange === nothing ? nothing : PermutationSymmetry(masses, [exchange]; signs = [sign])
    H = Kinetic(masses) + Coulomb(charges)
    sol = solve(
        H, SVM(n_gaussians; rank, candidates = 40) →
            Refine(2; candidates = 30) →
            GVM(; optimizer = LBFGS(; maxiter = 100, gradtol = 1.0e-8, verbosity = 0, ls_verbosity = 0));
        symmetry
    )
    @printf("%-8s %-6s %5d %19.12f %19.12f %+13.5e\n", system, state, length(sol.basis), energy(sol), reference, energy(sol) - reference)
    flush(stdout)
end
