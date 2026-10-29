using FewBodyECG
using FewBodyDB: db

helium = Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2.0, -1.0, -1.0])

he_ref = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value
he = solve(helium, SVM(basis = 50, candidates = 25))

f_nucleus(r) = -2.0 / r
f_electron(r) = 1.0 / r
helium_num = Kinetic([1.0e15, 1.0, 1.0]) +
    Custom(f_nucleus, 1, 2) + Custom(f_nucleus, 1, 3) + Custom(f_electron, 2, 3)

he_num = solve(helium_num, SVM(basis = 50, candidates = 25))

println("analytic  E0 = ", he.E₀, " Ha  (Δ = ", he.E₀ - he_ref, ")")
println("numerical E0 = ", he_num.E₀, " Ha  (Δ = ", he_num.E₀ - he_ref, ")")
