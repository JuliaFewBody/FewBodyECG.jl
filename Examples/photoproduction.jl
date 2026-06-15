using FewBodyECG
using LinearAlgebra
using Optim
using Plots
using Random

const HC = 197.3269804          # hbar * c in MeV fm
const ALPHA_FS = 1 / 137.035999084
const E2 = ALPHA_FS * HC        # e^2 in MeV fm
const FM2_TO_MICROBARN = 1.0e4

const PROTON_MASS = 938.27208816
const PION0_MASS = 134.9768

struct PhotoProductionModel
    bw::Float64   # fm
    Sw::Float64   # MeV
end

reduced_mass(m1, m2) = m1 * m2 / (m1 + m2)

function overlap_element(alpha_i, alpha_j)
    a = alpha_i + alpha_j
    return (9 / 2) * (pi / a)^(3 / 2) / a
end

function source_element(alpha, kappa, amplitude)
    a = alpha + kappa
    return (9 / 2) * amplitude * (pi / a)^(3 / 2) / a
end

function kinetic_element(alpha_i, alpha_j, mu)
    a = alpha_i + alpha_j
    return 3 * (HC^2 / (2 * mu)) * (15 * alpha_i * alpha_j / a^2) * (pi / a)^(3 / 2)
end

function build_model_matrices(
        alphas::AbstractVector{<:Real},
        model::PhotoProductionModel;
        pion_mass::Float64 = PION0_MASS,
        bare_mass::Float64 = PROTON_MASS,
    )
    K = length(alphas)
    kappa = 1 / model.bw^2
    amplitude = model.Sw / model.bw
    mu = reduced_mass(bare_mass, pion_mass)

    H = zeros(Float64, K + 1, K + 1)
    N = zeros(Float64, K + 1, K + 1)
    N[1, 1] = 1.0

    for i in 1:K
        H[1, i + 1] = source_element(alphas[i], kappa, amplitude)
        H[i + 1, 1] = H[1, i + 1]
        for j in 1:i
            Nij = overlap_element(alphas[i], alphas[j])
            Hij = kinetic_element(alphas[i], alphas[j], mu) + pion_mass * Nij
            N[i + 1, j + 1] = Nij
            N[j + 1, i + 1] = Nij
            H[i + 1, j + 1] = Hij
            H[j + 1, i + 1] = Hij
        end
    end

    return H, N, mu
end

function solve_dressed_proton(
        alphas::AbstractVector{<:Real},
        model::PhotoProductionModel;
        proton_mass::Float64 = PROTON_MASS,
        pion_mass::Float64 = PION0_MASS,
        tol::Float64 = 1.0e-10,
        maxiters::Int = 60,
    )
    bare_mass = proton_mass
    best = nothing

    for _ in 1:maxiters
        H, N, mu = build_model_matrices(alphas, model; pion_mass, bare_mass)
        evals, evecs = solve_generalized_eigenproblem(H, N; regularization = 1.0e-12)
        idx = argmin(evals)
        E0 = evals[idx]
        coeffs = evecs[:, idx]
        bare_mass_next = proton_mass - E0
        best = (; E0, coeffs, bare_mass = bare_mass_next, reduced_mass = mu, H, N)
        abs(bare_mass_next - bare_mass) < tol && break
        bare_mass = 0.5 * (bare_mass + bare_mass_next)
    end

    return merge(best, (; alphas = collect(alphas),))
end

function decode_alphas(theta::AbstractVector{<:Real})
    if any(!isfinite, theta)
        error("non-finite optimizer parameters")
    end
    alphas = exp.(theta)
    sort!(alphas)
    return alphas
end

function objective(theta::AbstractVector{<:Real}, model::PhotoProductionModel)
    if any(x -> x < log(1.0e-3) || x > log(5.0), theta)
        return Inf
    end

    alphas = decode_alphas(theta)
    try
        result = solve_dressed_proton(alphas, model)
        return isfinite(result.E0) ? result.E0 : Inf
    catch
        return Inf
    end
end

function initial_log_alphas(model::PhotoProductionModel, K::Int)
    kappa = 1 / model.bw^2
    amin = max(kappa / 5, 1.0e-2)
    amax = max(12 * kappa, 0.9)
    return collect(range(log(amin), log(amax), length = K))
end

function optimize_exponents(
        model::PhotoProductionModel;
        K::Int = 5,
        nstarts::Int = 6,
        maxiters::Int = 600,
        seed::Int = 22,
        verbose::Bool = false,
    )
    Random.seed!(seed)
    base = initial_log_alphas(model, K)
    starts = Vector{Vector{Float64}}()
    push!(starts, base)
    for _ in 2:nstarts
        push!(starts, base .+ 0.7 .* randn(K))
    end

    best_opt = nothing
    best_value = Inf

    for (istart, start) in enumerate(starts)
        opt = optimize(
            theta -> objective(theta, model),
            start,
            NelderMead(),
            Optim.Options(
                iterations = maxiters,
                store_trace = false,
                show_trace = false,
                show_every = 50,
            ),
        )
        value = Optim.minimum(opt)
        verbose && println("  start $istart: E0 = $(round(value; digits = 6)) MeV")
        if value < best_value
            best_value = value
            best_opt = opt
        end
    end

    best_opt === nothing && error("optimization failed")
    alphas = decode_alphas(Optim.minimizer(best_opt))
    solution = solve_dressed_proton(alphas, model)
    return merge(solution, (; optimizer = best_opt, model))
end

function pion_sector_weight(solution)
    cpi = solution.coeffs[2:end]
    Npi = solution.N[2:end, 2:end]
    return dot(cpi, Npi * cpi)
end

neutral_pion_weight(solution) = pion_sector_weight(solution) / 3

function dressing_form_factor(s, solution)
    coeffs = solution.coeffs[2:end]
    total = 0.0
    for (cn, alpha) in zip(coeffs, solution.alphas)
        total += cn * exp(-(s^2) / (4 * alpha)) * (pi / alpha)^(3 / 2) / (2 * alpha)
    end
    return total
end

function final_relative_energy(Egamma; proton_mass = PROTON_MASS, pion_mass = PION0_MASS)
    return Egamma + proton_mass - sqrt((proton_mass + pion_mass)^2 + Egamma^2)
end

function final_relative_momentum_squared(Eq; proton_mass = PROTON_MASS, pion_mass = PION0_MASS)
    return Eq * (Eq + 2 * proton_mass) * (Eq + 2 * pion_mass) *
           (Eq + 2 * proton_mass + 2 * pion_mass) /
           (4 * (Eq + proton_mass + pion_mass)^2)
end

function d_momentum_sq_d_energy(Eq; proton_mass = PROTON_MASS, pion_mass = PION0_MASS)
    a = Eq^2 + 2 * Eq * proton_mass + 2 * proton_mass^2 + 2 * Eq * pion_mass + 2 * proton_mass * pion_mass
    b = Eq^2 + 2 * Eq * proton_mass + 2 * pion_mass^2 + 2 * Eq * pion_mass + 2 * proton_mass * pion_mass
    return a * b / (2 * (Eq + proton_mass + pion_mass)^3)
end

function differential_cross_section(
        Egamma::Real,
        theta::Real,
        solution;
        proton_mass::Float64 = PROTON_MASS,
        pion_mass::Float64 = PION0_MASS,
    )
    Eq = final_relative_energy(Egamma; proton_mass, pion_mass)
    Eq <= 0 && return 0.0

    qhc2 = final_relative_momentum_squared(Eq; proton_mass, pion_mass)
    qhc2 <= 0 && return 0.0

    q = sqrt(qhc2) / HC
    k = Egamma / HC
    beta = pion_mass / (proton_mass + pion_mass)
    s2 = q^2 + (beta * k)^2 + 2 * beta * q * k * cos(theta)
    s = sqrt(max(s2, 0.0))
    F = dressing_form_factor(s, solution)

    prefactor = E2 / (8 * pi * proton_mass^2 * k)
    return prefactor * q^3 * d_momentum_sq_d_energy(Eq; proton_mass, pion_mass) *
           sin(theta)^2 * s2 * F^2
end

function trapz(x::AbstractVector, y::AbstractVector)
    total = 0.0
    for i in 1:(length(x) - 1)
        total += 0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1])
    end
    return total
end

function total_cross_section(Egamma::Real, solution; ntheta::Int = 600)
    theta_grid = collect(range(0.0, pi, length = ntheta))
    integrand = [
        2 * pi * sin(theta) * differential_cross_section(Egamma, theta, solution)
        for theta in theta_grid
    ]
    return trapz(theta_grid, integrand) * FM2_TO_MICROBARN
end

models = [
    PhotoProductionModel(3.8, 79.7),
    PhotoProductionModel(3.9, 41.5),
    PhotoProductionModel(4.0, 29.4),
]

println("Optimizing Gaussian exponents with Optim.NelderMead...")
solutions = map(models) do model
    println("\nModel: bw = $(model.bw) fm, Sw = $(model.Sw) MeV")
    solution = optimize_exponents(model; K = 5, nstarts = 6, maxiters = 700, verbose = false)
    w_pion = pion_sector_weight(solution)
    w_pi0 = neutral_pion_weight(solution)
    println("  E0                = $(round(solution.E0; digits = 3)) MeV")
    println("  bare proton mass  = $(round(solution.bare_mass; digits = 3)) MeV")
    println("  total pion weight = $(round(100 * w_pion; digits = 2)) %")
    println("  pi0 weight        = $(round(100 * w_pi0; digits = 2)) %")
    println("  best alphas       = $(join(round.(solution.alphas; digits = 4), ", ")) fm^-2")
    solution
end

Eth = ((PROTON_MASS + PION0_MASS)^2 - PROTON_MASS^2) / (2 * PROTON_MASS)
Egamma_grid = collect(range(Eth + 0.1, 172.0, length = 180))

curves = map(solutions) do solution
    [total_cross_section(Egamma, solution) for Egamma in Egamma_grid]
end

p = plot(
    Egamma_grid,
    curves[1];
    xlabel = "Egamma (MeV)",
    ylabel = "sigma (microbarn)",
    lw = 2,
    label = "bw = 3.8 fm, Sw = 79.7 MeV",
    title = "p(gamma, pi0)p near-threshold photoproduction",
)
plot!(p, Egamma_grid, curves[2]; lw = 2, label = "bw = 3.9 fm, Sw = 41.5 MeV")
plot!(p, Egamma_grid, curves[3]; lw = 2, label = "bw = 4.0 fm, Sw = 29.4 MeV")
vline!(p, [Eth]; ls = :dash, color = :gray, label = "threshold")
display(p)

fav = solutions[2]
println("\nPreferred paper parameter set (bw = 3.9 fm, Sw = 41.5 MeV):")
println("  total pion weight = $(round(100 * pion_sector_weight(fav); digits = 2)) %")
println("  pi0 weight        = $(round(100 * neutral_pion_weight(fav); digits = 2)) %")
println("  mass shift        = $(round(fav.E0; digits = 3)) MeV")
