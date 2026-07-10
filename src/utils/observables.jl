function correlation_function(
        sr::SolverResults;
        rmin::Real = 0.01,
        rmax::Real = 10.0,
        npoints::Int = 400,
        coord_index::Int = 1,
        normalize::Bool = true,
    )
    orbital = first(sr.basis_functions)
    orbital isa Rank0Gaussian ||
        throw(ArgumentError("correlation_function requires Rank0Gaussian basis functions"))
    d = size(orbital.s, 1)
    1 <= coord_index <= d || throw(ArgumentError("coord_index must be in 1:$d"))
    npoints > 1 || throw(ArgumentError("npoints must be greater than one"))

    r_grid = range(rmin, rmax, length = npoints)
    density = zeros(npoints)
    for (i, radius) in enumerate(r_grid)
        coordinates = zeros(d, 3)
        coordinates[coord_index, 3] = radius
        density[i] = radius^2 * abs2(ψ₀(coordinates, sr))
    end

    if normalize
        integral = sum(
            (density[i] + density[i + 1]) * (r_grid[i + 1] - r_grid[i]) / 2
                for i in 1:(npoints - 1)
        )
        iszero(integral) || (density ./= integral)
    end

    return collect(r_grid), density
end
