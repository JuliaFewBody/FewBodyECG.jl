# HD⁺ ground state (J = 0, v = 0) against two high-precision references.
using FewBodyECG
using FewBodyDB: db
using Plots

mₚ, m_d = 1836.15267343, 3670.48296788
ops = Kinetic([mₚ, m_d, 1.0]) + Coulomb([+1.0, +1.0, -1.0])
sol = solve(ops, SVM(basis = 150, candidates = 25))

reference(source) = db(source, "HD⁺", :energy, (J = 0, v = 0)).value
for source in (:Bubin2005Jan, :Karr2006Apr)
    @info "HD⁺ vs $source" E₀ = sol.E₀ reference = reference(source) Δ = sol.E₀ - reference(source)
end

plot(sol, reference(:Bubin2005Jan))
