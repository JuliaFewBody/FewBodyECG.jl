struct SolverResults
    basis_functions::Vector{GaussianBase}
    n_basis::Int
    operators::Vector{FewBodyHamiltonians.Operator}
    method::Symbol
    sampler::QuasiMonteCarlo.DeterministicSamplingAlgorithm
    length_scale::Float64
    ground_state::Float64
    energies::Vector{Float64}
    eigenvectors::Vector{Matrix{Float64}}
end

function ψ₀(r::AbstractVector, c::AbstractVector, basis_fns::Vector{<:GaussianBase})
    return sum(
        c[i] * exp(-r' * basis_fns[i].A * r + basis_fns[i].s' * r)
            for i in eachindex(basis_fns)
    )
end

function ψ₀(r::AbstractVector, sr::SolverResults; state::Int = 1)
    c = sr.eigenvectors[end][:, state]
    return ψ₀(r, c, sr.basis_functions)
end

function convergence(sr::SolverResults)
    return 1:sr.n_basis, sr.energies
end

function correlation_function(
        sr::SolverResults;
        rmin::Real = 0.01,
        rmax::Real = 10.0,
        npoints::Int = 400,
        coord_index::Int = 1,
        normalize::Bool = true
    )

    d = length(sr.basis_functions[1].s)
    1 <= coord_index <= d || throw(ArgumentError("coord_index must be in 1:$d"))

    r_grid = range(rmin, rmax, length = npoints)
    ρ_r = zeros(npoints)

    for (i, rval) in enumerate(r_grid)
        r_vec = zeros(d)
        r_vec[coord_index] = rval

        ψ_val = ψ₀(r_vec, sr)
        ρ_r[i] = rval^2 * abs2(ψ_val)
    end

    if normalize
        integral = sum(
            (ρ_r[i] + ρ_r[i + 1]) / 2 * (r_grid[i + 1] - r_grid[i])
                for i in 1:(npoints - 1)
        )
        if integral > 0
            ρ_r ./= integral
        end
    end

    return collect(r_grid), ρ_r
end

function ψ(sr::SolverResults)
    a, b = correlation_function(sr::SolverResults; normalize = true)
    return plot(a, b, xlabel = "r (a.u.)", ylabel = "r²|ψ(r)|²", label = "Correlation", lw = 2)
end
