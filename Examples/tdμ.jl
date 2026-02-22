using FewBodyECG, LinearAlgebra
using QuasiMonteCarlo
using Plots
using Random

import FewBodyECG: _generate_A_matrix

masses = [5496.918, 3670.481, 206.7686]

Λmat = Λ(masses)
kin = KineticOperator(Λmat)

J, U = _jacobi_transform(masses)

w_pairs = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]

w_jac = [U' * Float64.(w) for w in w_pairs]

coeffs = [+1.0, -1.0, -1.0]

ops = Operator[kin; [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_jac)]...]

E_exact = -111.36444

n_pairs = length(w_jac)
d = length(w_jac[1])
s_zero = zeros(d)

function make_points(n_max, n_pairs; method = :quasi)
    if method === :quasi
        # Halton sequence = Van der Corput with bases 2, 3, 5
        return [QuasiMonteCarlo.sample(i + 1, n_pairs, HaltonSample())[:, end]
                for i in 1:n_max]
    else
        Random.seed!(13)
        return [rand(n_pairs) for _ in 1:n_max]
    end
end

function compute_energy(points, n_basis, b₀)
    basis_fns = Rank0Gaussian[]
    for i in 1:n_basis
        bij = points[i] .* b₀
        A = _generate_A_matrix(bij, w_jac)
        push!(basis_fns, Rank0Gaussian(A, s_zero))
    end

    basis = BasisSet(basis_fns)
    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)

    try
        vals, vecs = solve_generalized_eigenproblem(H, S)
        return minimum(vals), vals, vecs
    catch e
        @warn "Eigenproblem failed" exception = e
        return NaN, Float64[], Matrix{Float64}(undef, 0, 0)
    end
end

n_basis = 180
b₀_values = range(0.02, 0.04, length = 15)

quasi_pts = make_points(n_basis, n_pairs; method = :quasi)
pseudo_pts = make_points(n_basis, n_pairs; method = :pseudo)

E_quasi_b₀ = Float64[]
E_pseudo_b₀ = Float64[]

for b₀ in b₀_values
    Eq, _, _ = compute_energy(quasi_pts, n_basis, b₀)
    Ep, _, _ = compute_energy(pseudo_pts, n_basis, b₀)
    push!(E_quasi_b₀, Eq)
    push!(E_pseudo_b₀, Ep)
    println("  b₀ = $(round(b₀; digits=4)):  quasi = $(round(Eq; digits=4)),  pseudo = $(round(Ep; digits=4))")
end

rel_quasi = @. (E_quasi_b₀ - E_exact) / abs(E_exact)
rel_pseudo = @. (E_pseudo_b₀ - E_exact) / abs(E_exact)

p1 = plot(b₀_values, rel_pseudo,
    label = "pseudo", marker = :square, ls = :dash, lw = 1.5,
    xlabel = "scale factor b₀", ylabel = "(E - Eₓ)/|Eₓ|",
    title = "E[tdμ], $n_basis Gaussians",
    yscale = :log10, ylims = (1e-4, 1e-1), legend = :topright)
plot!(p1, b₀_values, rel_quasi,
    label = "quasi", marker = :circle, ls = :solid, lw = 1.5)

display(p1)

b₀_opt_quasi = b₀_values[argmin(E_quasi_b₀)]
b₀_opt_pseudo = b₀_values[argmin(E_pseudo_b₀)]
println("\nOptimal b₀ (quasi):  $b₀_opt_quasi → E = $(minimum(E_quasi_b₀))")
println("Optimal b₀ (pseudo): $b₀_opt_pseudo → E = $(minimum(E_pseudo_b₀))")
println("Exact:                           Eₓ = $E_exact")

b₀_opt = b₀_opt_quasi
n_values = 100:10:200

n_max = maximum(n_values)
quasi_pts_big = make_points(n_max, n_pairs; method = :quasi)
pseudo_pts_big = make_points(n_max, n_pairs; method = :pseudo)

E_quasi_n = Float64[]
E_pseudo_n = Float64[]
best_vecs = Matrix{Float64}(undef, 0, 0)
best_n = 0

for n in n_values
    Eq, vq, vecq = compute_energy(quasi_pts_big, n, b₀_opt)
    Ep, _, _ = compute_energy(pseudo_pts_big, n, b₀_opt)
    push!(E_quasi_n, Eq)
    push!(E_pseudo_n, Ep)

    if Eq == minimum(E_quasi_n)
        global best_vecs = vecq
        global best_n = n
    end

    println("  n = $n:  quasi = $(round(Eq; digits=4)),  pseudo = $(round(Ep; digits=4))")
end

rel_quasi_n = @. (E_quasi_n - E_exact) / abs(E_exact)
rel_pseudo_n = @. (E_pseudo_n - E_exact) / abs(E_exact)

p2 = plot(collect(n_values), rel_pseudo_n,
    label = "pseudo", marker = :square, ls = :dash, lw = 1.5,
    xlabel = "basis size n", ylabel = "(E - Eₓ)/|Eₓ|",
    title = "E[tdμ]",
    yscale = :log10, ylims = (1e-4, 1e-1), legend = :topright)
plot!(p2, collect(n_values), rel_quasi_n,
    label = "quasi", marker = :circle, ls = :solid, lw = 1.5)

display(p2)

best_fns = [Rank0Gaussian(_generate_A_matrix(quasi_pts_big[i] .* b₀_opt, w_jac), s_zero)
             for i in 1:best_n]
c₀ = best_vecs[:, 1]

r_grid = range(0.001, 0.15, length = 400)
ρ_r = zeros(length(r_grid))

for (k, rval) in enumerate(r_grid)
    r_vec = zeros(d)
    r_vec[1] = rval
    ψ_val = ψ₀(r_vec, c₀, best_fns)
    ρ_r[k] = rval^2 * abs2(ψ_val)
end

# Normalize
dr = step(r_grid)
integral = sum(ρ_r) * dr
if integral > 0
    ρ_r ./= integral
end

p3 = plot(r_grid, ρ_r,
    xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²",
    label = "tdμ correlation", lw = 2,
    title = "tdμ radial correlation function")

display(p3)

println("\nFinal best energy (quasi): $(minimum(E_quasi_n))")
println("Exact:                     $E_exact")
println("Relative error:            $((minimum(E_quasi_n) - E_exact) / abs(E_exact))")
