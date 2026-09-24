"""
    radial_profile(ψ::Wavefunction; coord = 1, direction = (0, 0, 1),
                   rmax = 10.0, npoints = 400, normalize = true)

Sample the radial density `r²|ψ|²` along Jacobi coordinate `coord` on the
physical half-line `r ≥ 0`, placing that coordinate at `r * direction` (the
Cartesian `direction` is normalized to unit length) and holding the other
coordinates at zero. Returns `(r, density)`.  When `normalize = true` the
density is scaled so that its trapezoidal integral over `[0, rmax]` equals 1.

Polarized basis functions are not spherically symmetric: for example an
`xy`-polarized `Rank2Gaussian` vanishes along the `z` axis, so choose a
`direction` such as `(1, 1, 0)` for it.

Because `r²|ψ|²` is defined only for non-negative radial distance, no mirrored
negative-`r` branch is produced.
"""
function radial_profile(
        ψ::Wavefunction;
        coord::Int = 1, direction = (0, 0, 1), rmax::Real = 10.0, npoints::Int = 400,
        normalize::Bool = true
    )
    d = size(first(ψ.basis.functions).A, 1)
    1 ≤ coord ≤ d || throw(ArgumentError("coord must be in 1:$d"))
    length(direction) == 3 ||
        throw(ArgumentError("direction must have 3 Cartesian components, got $(length(direction))"))
    n̂ = collect(float.(direction))
    norm(n̂) > 0 || throw(ArgumentError("direction must be nonzero"))
    n̂ ./= norm(n̂)
    r = collect(range(0.0, float(rmax), length = npoints))
    density = similar(r)
    v = zeros(d, 3)
    for k in eachindex(r)
        v[coord, :] .= r[k] .* n̂
        density[k] = r[k]^2 * abs2(ψ(v))
    end
    if normalize
        area = zero(eltype(density))
        for i in 1:(npoints - 1)
            area += (density[i] + density[i + 1]) * (r[i + 1] - r[i]) / 2
        end
        area > 0 && (density ./= area)
    end
    return r, density
end
