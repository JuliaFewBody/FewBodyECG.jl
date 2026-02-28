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
    fg_history::Vector{Float64}
end

"""
    ψ₀(r, c, basis_fns)
    ψ₀(r, sr; state=1)

Evaluate the ground-state wavefunction at Jacobi-coordinate point `r`.

The wavefunction is the linear combination
``\\psi_0(\\mathbf{r}) = \\sum_i c_i \\exp(-\\mathbf{r}^T A_i \\mathbf{r} + \\mathbf{s}_i^T \\mathbf{r})``.

When called with a [`SolverResults`](@ref), the eigenvector for the requested
`state` (default 1, i.e. the ground state) is used automatically.
"""
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

"""
    convergence(sr::SolverResults) -> (indices, energies)

Return the greedy build-up convergence curve from a stochastic [`solve_ECG`](@ref) run.

Returns `(1:n_basis, sr.energies)`: the energy after each basis function was
added.  For variational results `energies` contains only one entry
`[ground_state]`; use [`convergence_history`](@ref) instead.
"""
function convergence(sr::SolverResults)
    return 1:sr.n_basis, sr.energies
end

"""
    convergence_history(sr::SolverResults) -> (indices, energies)

Return the per-objective-call convergence history.

For [`solve_ECG_variational`](@ref) results this is the cumulative-minimum
energy at every primal `fg` evaluation, giving a monotone non-increasing curve
suitable for plotting optimisation progress.  The x-axis is the fg-call index.

For [`solve_ECG`](@ref) results this mirrors `convergence`.
"""
function convergence_history(sr::SolverResults)
    return 1:length(sr.fg_history), sr.fg_history
end

"""
    correlation_function(sr; rmin=0.01, rmax=10.0, npoints=400,
                         coord_index=1, normalize=true)

Compute the one-body radial density ``\\rho(r) = r^2 |\\psi_0(r)|^2`` along
a single Jacobi coordinate.

# Arguments
- `sr`           : [`SolverResults`](@ref) from either solver.
- `rmin`, `rmax` : radial grid range (a.u.).
- `npoints`      : number of grid points.
- `coord_index`  : which Jacobi coordinate to scan (default 1).
- `normalize`    : if `true`, normalise so that ``\\int \\rho(r)\\,dr = 1``.

Returns `(r_grid, ρ)` as plain `Vector{Float64}`.
"""
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
