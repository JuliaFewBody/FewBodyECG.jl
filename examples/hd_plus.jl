using FewBodyECG
using FewBodyDB: db
using Plots

mₚ, m_d = 1836.15267343, 3670.48296788
ops = Operators([mₚ, m_d, 1.0], [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 150, candidates = 25, scale = 1.0))
sol

state = (J = 0, v = 0)
for reference in (:Bubin2005Jan, :Karr2006Apr)
    ref = db(reference, "HD⁺", :energy, state).value
    println(reference, ": ", ref, " Ha   Δ = ", sol.E₀ - ref)
end

plot(sol, db(:Bubin2005Jan, "HD⁺", :energy, state).value)
