# Helium with the Coulomb terms given as plain functions of the distance, whose
# matrix elements are evaluated by numerical quadrature.
using FewBodyECG
using FewBodyDB: db

reference = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value

analytic = solve(Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2.0, -1.0, -1.0]), SVM(basis = 50, candidates = 25))

nucleus(r) = -2.0 / r
electron(r) = 1.0 / r
numerical_ops = Kinetic([1.0e15, 1.0, 1.0]) + Custom(nucleus, 1, 2) + Custom(nucleus, 1, 3) + Custom(electron, 2, 3)
numerical = solve(numerical_ops, SVM(basis = 50, candidates = 25))

@info "He" analytic = analytic.E₀ numerical = numerical.E₀ reference
