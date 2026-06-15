using FewBodyECG
using Plots

function make_ops(Z::Int)
    os = Operators([1.0e15, 1.0, 1.0], [+Z, -1, -1])
    os += "Kinetic"
    os += "Coulomb"
    return os
end

E_ref_He  = -2.903724   # SVM K=200  (Suzuki & Varga 1998, Table 8.1)
E_ref_Li  = -7.279913   # SVM K=300

println("He  (Z=2): sequential ECG  K=50...")
result_he = solve_ECG_sequential(make_ops(2), 50;
    scale = 1.0, n_candidates = 20, verbose = false)
println("  E = $(round(result_he.ground_state; digits = 7)) Ha  " *
    "(ref: $E_ref_He,  err: $(round(result_he.ground_state - E_ref_He; sigdigits=2)))")

println("Li⁺ (Z=3): sequential ECG  K=50...")
result_li = solve_ECG_sequential(make_ops(3), 50;
    scale = 0.5, n_candidates = 20, verbose = false)
println("  E = $(round(result_li.ground_state; digits = 7)) Ha  " *
    "(ref: $E_ref_Li,  err: $(round(result_li.ground_state - E_ref_Li; sigdigits=2)))")

n_conv, E_conv = convergence(result_li)

p1 = plot(n_conv, E_conv;
    xlabel = "Basis size", ylabel = "E (Ha)",
    label = "Li⁺ sequential ECG", lw = 2,
    title = "Li⁺ (Z=3) convergence  ¹Sᵉ")
hline!(p1, [E_ref_Li]; ls = :dash, color = :gray, label = "SVM ref  K=300")
display(p1)

r_he, ρ_he = correlation_function(result_he; rmax = 4.0, npoints = 300)
r_li, ρ_li = correlation_function(result_li; rmax = 4.0, npoints = 300)

p2 = plot(r_he, ρ_he;
    xlabel = "r (a₀)", ylabel = "r²|ψ(r)|²",
    label = "He  (Z=2)", lw = 2, ls = :dash,
    title = "Radial density: He vs Li⁺")
plot!(p2, r_li, ρ_li; lw = 2, label = "Li⁺ (Z=3)")
display(p2)

println("\nIsoelectronic series scan (greedy K=100 each)...")
Z_range = 2:5
E_ecg = Float64[]
for Z in Z_range
    sc = Z == 2 ? 1.0 : 0.5
    r = solve_ECG(make_ops(Z), 100; scale = sc, verbose = false)
    println("  Z=$Z: E = $(round(r.ground_state; digits = 5)) Ha")
    push!(E_ecg, r.ground_state)
end
E_ni = [-Float64(Z)^2 for Z in Z_range]   # non-interacting: −Z²

p3 = plot(collect(Z_range), E_ecg;
    xlabel = "Nuclear charge Z", ylabel = "E (Ha)",
    label = "ECG (interacting)", lw = 2, marker = :circle,
    title = "Helium-like isoelectronic series")
plot!(p3, collect(Z_range), E_ni;
    lw = 2, ls = :dash, label = "Non-interacting  (−Z²)")
scatter!(p3, [2, 3], [E_ref_He, E_ref_Li];
    label = "SVM ref (Suzuki & Varga)", marker = :diamond, ms = 6)
display(p3)
