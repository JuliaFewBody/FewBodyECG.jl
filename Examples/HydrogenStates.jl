using FewBodyECG
using LinearAlgebra
using Plots

masses = [1.0e15, 1.0]
Λmat   = Λ(masses)
_, U   = _jacobi_transform(masses)
w      = U' * Float64.([1, -1])   # electron-proton separation in Jacobi coords

ops = Operator[KineticOperator(Λmat); CoulombOperator(-1.0, w)]

sr_1s = solve_ECG_sequential(ops, 16;
    n_candidates = 8, scale = 1.0, max_iterations_step = 120, verbose = true)

E_1s = sr_1s.ground_state
println("\n  E(1s)       = $(round(E_1s; digits = 8)) Ha")
println("  E_exact(1s) = -0.50000000 Ha")
println("  |ΔE|        = $(round(abs(E_1s + 0.5); sigdigits = 2))\n")

a_p    = [1.0]
s_zero = [0.0]

# Log-spaced widths covering the spatial extent of the 2p orbital (peak ~4 a₀)
alphas_p = exp10.(range(log10(0.003), log10(3.0), length = 16))
basis_p  = GaussianBase[]
E_2p_conv = Float64[]
vecs_2p   = Matrix{Float64}(undef, 0, 0)

for (k, α) in enumerate(alphas_p)
    push!(basis_p, Rank1Gaussian([α;;], a_p, s_zero))
    bset = BasisSet(basis_p)
    H = build_hamiltonian_matrix(bset, ops)
    S = build_overlap_matrix(bset)
    vals, vecs = solve_generalized_eigenproblem(H, S)
    E_k = minimum(vals)
    push!(E_2p_conv, E_k)
    global vecs_2p = vecs
    println("  step $(lpad(k, 2))  α = $(rpad(round(α; digits = 4), 7))" *
            "  E = $(round(E_k; digits = 8))")
end

E_2p = minimum(E_2p_conv)
c_2p = vecs_2p[:, 1]
println("\n  E(2p)       = $(round(E_2p; digits = 8)) Ha")
println("  E_exact(2p) = -0.12500000 Ha")
println("  |ΔE|        = $(round(abs(E_2p + 0.125); sigdigits = 2))\n")

a_d      = [1.0]
# α range chosen for the 3d orbital spatial scale: peak of r⁶exp(−2αr²) is at
# r = √(3/α), so α ≈ 1/27 ≈ 0.037 for the 3d state (peak at r ≈ 9 a₀).
alphas_d = exp10.(range(log10(0.002), log10(0.8), length = 12))
basis_d   = GaussianBase[]
E_3d_conv = Float64[]
vecs_3d   = Matrix{Float64}(undef, 0, 0)

for (k, α) in enumerate(alphas_d)
    push!(basis_d, Rank2Gaussian([α;;], a_d, a_d, s_zero))
    bset = BasisSet(basis_d)
    H = build_hamiltonian_matrix(bset, ops)
    S = build_overlap_matrix(bset)
    vals, vecs = solve_generalized_eigenproblem(H, S)
    E_k = minimum(vals)
    push!(E_3d_conv, E_k)
    global vecs_3d = vecs
    note = E_k < -1 / 18 ? "  ← below −1/18" : ""
    println("  step $(lpad(k, 2))  α = $(rpad(round(α; digits = 4), 7))" *
            "  E = $(round(E_k; digits = 6))" * note)
end

E_rank2 = minimum(E_3d_conv)
c_rank2 = vecs_3d[:, 1]
println("\n  E(Rank2)      = $(round(E_rank2; digits = 6)) Ha")
println("  E_exact(3d)   = $(round(-1/18; digits = 6)) Ha")
println("  (energy lies below 3d due to L=0 mixing)\n")

function ψ_rank1(rval, c, bfs)
    r_vec = [rval]
    return sum(
        c[i] * dot(bfs[i].a, r_vec) *
            exp(-dot(r_vec, parent(bfs[i].A) * r_vec) + dot(bfs[i].s, r_vec))
            for i in eachindex(bfs)
    )
end

function ψ_rank2(rval, c, bfs)
    r_vec = [rval]
    return sum(
        c[i] * dot(bfs[i].a, r_vec) * dot(bfs[i].b, r_vec) *
            exp(-dot(r_vec, parent(bfs[i].A) * r_vec) + dot(bfs[i].s, r_vec))
            for i in eachindex(bfs)
    )
end

function normalise(ρ, grid)
    dr = step(grid)
    return ρ ./ (sum(ρ) * dr)
end

r_1s = range(0.01, 12.0, length = 600)
r_2p = range(0.01, 22.0, length = 600)
r_3d = range(0.01, 35.0, length = 600)

ρ_ecg_1s   = normalise([r^2 * abs2(ψ₀([r], sr_1s)) for r in r_1s], r_1s)
ρ_ecg_2p   = normalise([r^2 * abs2(ψ_rank1(r, c_2p, basis_p)) for r in r_2p], r_2p)
ρ_ecg_rank2 = normalise([r^2 * abs2(ψ_rank2(r, c_rank2, basis_d)) for r in r_3d], r_3d)

ρ_exact_1s = normalise([r^2 * exp(-2r)       for r in r_1s], r_1s)
ρ_exact_2p = normalise([r^4 * exp(-r)        for r in r_2p], r_2p)
ρ_exact_3d = normalise([r^6 * exp(-2r / 3)  for r in r_3d], r_3d)

p = plot(
    layout = (3, 1),
    size   = (720, 1000),
    left_margin   = 6Plots.mm,
    bottom_margin = 4Plots.mm,
    legend = :topright,
)

plot!(p[1], collect(r_1s), ρ_ecg_1s;
    label = "ECG  Rank0 (16 fn.)", lw = 2.5, color = :steelblue)
plot!(p[1], collect(r_1s), ρ_exact_1s;
    label = "Exact 1s", lw = 1.8, ls = :dash, color = :black)
xlabel!(p[1], "r (a.u.)")
ylabel!(p[1], "r²|ψ(r)|²")
title!(p[1], "1s  (L=0)  E = $(round(E_1s; digits=6)) Ha  |  exact = −0.5")

plot!(p[2], collect(r_2p), ρ_ecg_2p;
    label = "ECG  Rank1 (16 fn.)", lw = 2.5, color = :tomato)
plot!(p[2], collect(r_2p), ρ_exact_2p;
    label = "Exact 2p", lw = 1.8, ls = :dash, color = :black)
xlabel!(p[2], "r (a.u.)")
ylabel!(p[2], "r²|ψ(r)|²")
title!(p[2], "2p  (L=1)  E = $(round(E_2p; digits=6)) Ha  |  exact = −0.125")

plot!(p[3], collect(r_3d), ρ_ecg_rank2;
    label = "ECG  Rank2 (12 fn.)", lw = 2.5, color = :seagreen)
plot!(p[3], collect(r_3d), ρ_exact_3d;
    label = "Exact 3d shape  (r⁶ e^{−2r/3})", lw = 1.8, ls = :dash, color = :black)
xlabel!(p[3], "r (a.u.)")
ylabel!(p[3], "r²|ψ(r)|²")
title!(p[3], "d-wave-like  Rank2  E = $(round(E_rank2; digits=5)) Ha  |" *
             "  exact 3d = $(round(-1/18; digits=5))  (L=0/L=2 mixed)")

display(p)
