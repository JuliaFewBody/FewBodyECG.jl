# # Muonic molecular ions (ddμ, dtμ, ttμ)
#
# A negative muon binds two hydrogen-isotope nuclei into a tiny molecular ion —
# the mechanism behind muon-catalyzed fusion.  Each is a three-body Coulomb
# problem `Operators(masses, charges)` away, differing only in the nuclear masses
# (electron-mass atomic units).  The heavy, deeply-bound, two-scale nature makes
# these among the hardest ECG problems; the Suzuki–Varga K = 200 benchmarks are
# reproduced here to a few mHa.

using FewBodyECG

mμ, md, mt = 206.7686, 3670.481, 5496.918
systems = [
    ("ddμ", [md, md, mμ], NaN),
    ("dtμ", [md, mt, mμ], -111.36444),        # Suzuki–Varga Table 8.1
    ("ttμ", [mt, mt, mμ], -112.973),        # Suzuki–Varga Table 8.1
]

for (name, masses, ref) in systems
    ops = Operators(masses, [+1.0, +1.0, -1.0])
    ops += "Kinetic"
    ops += "Coulomb"
    sol = solve(ops, SVM(basis = 300, candidates = 40, scale = 0.02); tol = 1.0e-4, window = 15)
    msg = isnan(ref) ? "(prediction)" : "ref $ref   Δ = $(round(sol.E₀ - ref, digits = 5))"
    println(rpad(name, 5), " E₀ = ", round(sol.E₀, digits = 5), " Ha   ", msg)
end

# Heavier nuclei localise the muon more tightly, so the binding deepens along
# ddμ → dtμ → ttμ.
