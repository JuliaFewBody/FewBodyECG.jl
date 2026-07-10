function plot_correlation(
        plotter,
        result::SolverResults;
        rmin::Real = 0.01,
        rmax::Real = 10.0,
        npoints::Int = 400,
        coord_index::Int = 1,
        normalize::Bool = true,
        kwargs...,
    )
    r, density = correlation_function(result; rmin, rmax, npoints, coord_index, normalize)
    return plotter(r, density; kwargs...)
end
