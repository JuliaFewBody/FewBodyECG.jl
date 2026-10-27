# # Muonic molecular ions
#
# In a muonic molecular ion such as tdμ, a negative muon (mass ≈ 207 mₑ) binds
# two hydrogen nuclei.  All three particles have comparable masses, so none can
# be treated as fixed, and the system is about 200 times smaller than an
# ordinary molecule, about 0.01 bohr across.  Masses are in electron masses and
# energies in Hartree (atomic units).

using FewBodyECG
using FewBodyDB: db

mμ, md, mt = 206.7686, 3670.481, 5496.918

# The K = 200 SVM energies are from Suzuki & Varga, Table 8.1; the reference
# energies are the more precise values from the same book, stored in FewBodyDB.
#
# `SVM` grows the basis to 200 functions; `Refine` then revisits each function
# and replaces it whenever a fresh candidate lowers the energy, keeping the
# basis size at 200 so the comparison with the K = 200 values stays fair.
systems = [
    ("tdμ", [md, mt, mμ], -111.36444),
    ("ttμ", [mt, mt, mμ], -112.973),
]

for (name, masses, svm200) in systems
    ref = db(:Suzuki2003Jul, name, :energy, "¹Sᵉ").value

    ops = Operators(masses, [+1.0, +1.0, -1.0])
    ops += "Kinetic"
    ops += "Coulomb"

    sol = solve(
        ops,
        SVM(basis = 200, candidates = 40) →
            Refine(2; candidates = 20);
        tol = 1.0e-4,
        window = 15,
    )

    println(name)
    println("  ECG E₀              = ", sol.E₀, " Ha")
    println("  Δ vs SVM K=200      = ", sol.E₀ - svm200, " Ha")
    println("  Δ vs FewBodyDB      = ", sol.E₀ - ref, " Ha")
end
