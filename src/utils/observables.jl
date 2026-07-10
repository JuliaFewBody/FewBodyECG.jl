"""
    radial_profile(ψ::Wavefunction; coord = 1, rmax = 10.0, npoints = 400, normalize = true)

Sample the radial density `r²|ψ(r)|²` along Jacobi coordinate `coord` on the
physical half-line `r ≥ 0` (the other coordinates held at zero), returning
`(r, density)`.  When `normalize = true` the density is scaled so that its
trapezoidal integral over `[0, rmax]` equals 1.

Because `r²|ψ|²` is defined only for non-negative radial distance, no mirrored
negative-`r` branch is produced.
"""
function radial_profile(
        ψ::Wavefunction;
        coord::Int = 1, rmax::Real = 10.0, npoints::Int = 400, normalize::Bool = true
    )
    d = size(first(ψ.basis.functions).A, 1)
    1 ≤ coord ≤ d || throw(ArgumentError("coord must be in 1:$d"))
    r = collect(range(0.0, float(rmax), length = npoints))
    density = similar(r)
    v = zeros(d)
    for k in eachindex(r)
        fill!(v, 0.0)
        v[coord] = r[k]
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
