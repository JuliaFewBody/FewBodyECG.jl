using FewBodyECG
import Antique
using LinearAlgebra
using Plots

atom = Antique.HydrogenAtom()

masses = [1.0e15, 1.0]
Λmat   = Λ(masses)
_, U   = _jacobi_transform(masses)
w      = U' * Float64.([1, -1])   # electron-proton separation in Jacobi coords

ops = Operator[KineticOperator(Λmat); CoulombOperator(-1.0, w)]

sr_1s = solve_ECG_sequential(ops, 16;
    n_candidates = 8, scale = 1.0, max_iterations_step = 120, verbose = true)

E_1s = sr_1s.ground_state
println("\n  E(1s)       = $(round(E_1s; digits = 8)) Ha")
println("  E_exact(1s) = $(round(Antique.E(atom; n = 1); digits = 8)) Ha")
println("  |ΔE|        = $(round(abs(E_1s - Antique.E(atom; n = 1)); sigdigits = 2))\n")

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
println("  E_exact(2p) = $(round(Antique.E(atom; n = 2); digits = 8)) Ha")
println("  |ΔE|        = $(round(abs(E_2p - Antique.E(atom; n = 2)); sigdigits = 2))\n")

# Orthogonal polarization vectors define a pure d-wave channel.
a_d = reshape([1.0, 0.0, 0.0], 1, 3)
b_d = reshape([0.0, 0.0, 1.0], 1, 3)
alphas_d = exp10.(range(log10(0.002), log10(0.8), length = 24))
basis_d   = GaussianBase[]
E_3d_conv = Float64[]
vecs_3d   = Matrix{Float64}(undef, 0, 0)

for (k, α) in enumerate(alphas_d)
    push!(basis_d, Rank2Gaussian([α;;], a_d, b_d, s_zero))
    bset = BasisSet(basis_d)
    H = build_hamiltonian_matrix(bset, ops)
    S = build_overlap_matrix(bset)
    vals, vecs = solve_generalized_eigenproblem(H, S)
    E_k = minimum(vals)
    push!(E_3d_conv, E_k)
    global vecs_3d = vecs
    println("  step $(lpad(k, 2))  α = $(rpad(round(α; digits = 4), 7))" *
            "  E = $(round(E_k; digits = 8))")
end

E_rank2 = minimum(E_3d_conv)
c_rank2 = vecs_3d[:, 1]
println("\n  E(3d, Rank2 pure d) = $(round(E_rank2; digits = 8)) Ha")
println("  E_exact(3d)         = $(round(Antique.E(atom; n = 3); digits = 8)) Ha")
println("  |ΔE|                = $(round(abs(E_rank2 - Antique.E(atom; n = 3)); sigdigits = 2))\n")

function ψ_rank1(rval, c, bfs)
    r_vec = [rval]
    return sum(
        c[i] * dot(bfs[i].a, r_vec) *
            exp(-dot(r_vec, parent(bfs[i].A) * r_vec) + dot(bfs[i].s, r_vec))
            for i in eachindex(bfs)
    )
end

function ψ_rank2(rval, c, bfs)
    # Directional profile for the pure d-wave basis: r̂ = (x+z)/√2.
    θ_d, φ_d = π / 4, 0.0
    r_cart = rval .* [sin(θ_d) * cos(φ_d), sin(θ_d) * sin(φ_d), cos(θ_d)]
    r_vec = [rval]
    return sum(
        c[i] * (
            bfs[i].a isa AbstractMatrix ?
            dot(vec(bfs[i].a), r_cart) * dot(vec(bfs[i].b), r_cart) :
            dot(bfs[i].a, r_vec) * dot(bfs[i].b, r_vec)
        ) * exp(-dot(r_vec, parent(bfs[i].A) * r_vec) + dot(bfs[i].s, r_vec))
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

θ_p, φ_p = 0.0, 0.0
θ_d, φ_d = π / 4, 0.0

ρ_exact_1s = normalise(
    [r^2 * abs2(Antique.ψ(atom, r, 0.0, 0.0; n = 1, l = 0, m = 0)) for r in r_1s],
    r_1s,
)
ρ_exact_2p = normalise(
    [r^2 * abs2(Antique.ψ(atom, r, θ_p, φ_p; n = 2, l = 1, m = 0)) for r in r_2p],
    r_2p,
)
ρ_exact_3d = normalise(
    [r^2 * abs2(Antique.ψ(atom, r, θ_d, φ_d; n = 3, l = 2, m = 1)) for r in r_3d],
    r_3d,
)

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
    label = "Antique 1s", lw = 1.8, ls = :dash, color = :black)
xlabel!(p[1], "r (a.u.)")
ylabel!(p[1], "r²|ψ(r)|²")
title!(p[1], "1s  (L=0)  E = $(round(E_1s; digits=6)) Ha  |  exact = $(round(Antique.E(atom; n = 1); digits = 6))")

plot!(p[2], collect(r_2p), ρ_ecg_2p;
    label = "ECG  Rank1 (16 fn.)", lw = 2.5, color = :tomato)
plot!(p[2], collect(r_2p), ρ_exact_2p;
    label = "Antique 2p (θ=0)", lw = 1.8, ls = :dash, color = :black)
xlabel!(p[2], "r (a.u.)")
ylabel!(p[2], "r²|ψ(r)|²")
title!(p[2], "2p  (L=1)  E = $(round(E_2p; digits=6)) Ha  |  exact = $(round(Antique.E(atom; n = 2); digits = 6))")

plot!(p[3], collect(r_3d), ρ_ecg_rank2;
    label = "ECG  Rank2 (12 fn.)", lw = 2.5, color = :seagreen)
plot!(p[3], collect(r_3d), ρ_exact_3d;
    label = "Antique 3d (θ=π/4, m=1)", lw = 1.8, ls = :dash, color = :black)
xlabel!(p[3], "r (a.u.)")
ylabel!(p[3], "r²|ψ(r)|²")
title!(p[3], "pure d-wave  Rank2  E = $(round(E_rank2; digits=6)) Ha  |  exact 3d = $(round(Antique.E(atom; n = 3); digits = 6))")

display(p)
