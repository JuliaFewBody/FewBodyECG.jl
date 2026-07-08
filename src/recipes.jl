using RecipesBase

# plot(sol):        per-stage energy curves vs cumulative step
# plot(sol, E_ref): same, plus a reference-energy hline
@recipe function f(sol::Solution, reference::Union{Nothing, Real} = nothing)
    xguide --> "step"
    yguide --> "E (Ha)"
    legend --> :topright
    offset = 0
    for st in getfield(sol, :stages)
        xs = offset .+ (1:length(st.energies))
        offset += length(st.energies)
        @series begin
            label --> sprint(show, st.method)
            seriestype --> :path
            linewidth --> 2
            xs, st.energies
        end
    end
    if reference !== nothing
        @series begin
            label --> "reference"
            seriestype --> :hline
            linestyle --> :dash
            [float(reference)]
        end
    end
end

# plot(ψ; coord = 1, rmax = 10.0, npoints = 400): radial profile r²|ψ|²
# along one Jacobi coordinate (others fixed at 0).
@recipe function f(ψ::Wavefunction; coord = 1, rmax = 10.0, npoints = 400)
    d = length(first(ψ.basis.functions).s)
    1 ≤ coord ≤ d || throw(ArgumentError("coord must be in 1:$d"))
    rs = range(1.0e-3, rmax, length = npoints)
    ys = map(rs) do r
        v = zeros(d)
        v[coord] = r
        r^2 * abs2(ψ(v))
    end
    xguide --> "r (Jacobi coordinate $coord, mass-weighted)"
    yguide --> "r²|ψ(r)|²"
    label --> "|ψ|²"
    linewidth --> 2
    collect(rs), ys
end
