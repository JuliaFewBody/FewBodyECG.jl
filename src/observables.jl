"""
    Wavefunction

Callable variational wavefunction `ψ(r) = Σᵢ cᵢ gᵢ(r)` in **Jacobi
coordinates** (mass-weighted: the package's Jacobi transform normalises each
relative coordinate by √μ — see `jacobi_transform`).  Obtained from
[`wavefunction`](@ref); plot with `plot(ψ; coord = i)`.
"""
struct Wavefunction
    basis::BasisSet
    c::Vector{Float64}
end

_gauss(g, r) = exp(-(r' * g.A * r) + g.s' * r)
_eval(g::Rank0Gaussian, r) = _gauss(g, r)
_eval(g::Rank1Gaussian, r) = sum(_polar_projection(g.a, r)) * _gauss(g, r)
_eval(g::Rank2Gaussian, r) =
    dot(_polar_projection(g.a, r), _polar_projection(g.b, r)) * _gauss(g, r)

(ψ::Wavefunction)(r::AbstractVector) =
    sum(ψ.c[i] * _eval(ψ.basis.functions[i], r) for i in eachindex(ψ.c))

"""
    wavefunction(sol::Solution; state = sol.state) -> Wavefunction
"""
wavefunction(sol::Solution; state::Int = sol.state) =
    Wavefunction(getfield(sol, :basis), getfield(sol, :coefficients)[:, state])
