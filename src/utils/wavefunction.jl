"""
    Wavefunction

Callable variational wavefunction `ψ(r) = Σᵢ cᵢ gᵢ(r)` in **Jacobi
coordinates** (mass-weighted: the package's Jacobi transform normalises each
relative coordinate by √μ — see `jacobi_transform`).  Obtained from
[`wavefunction`](@ref); plot with `plot(ψ; coord = i)` or sample with
[`radial_profile`](@ref).
"""
struct Wavefunction
    basis::BasisSet
    c::AbstractVector{<:Number}
end

# `r` holds the (1D) amplitude of each Jacobi coordinate along the z axis; with
# an isotropic A this reproduces the radial Gaussian, and only the z component
# of the N×3 shift couples to it.
_shift_z(g::Rank0Gaussian) = @view parent(g.s)[:, 3]
_shift_z(g) = g.s
_gauss(g, r) = exp(-(r' * g.A * r) + _shift_z(g)' * r)
_eval(g::Rank0Gaussian, r) = _gauss(g, r)
_eval(g::Rank1Gaussian, r) = sum(_polar_projection(g.a, r)) * _gauss(g, r)
_eval(g::Rank2Gaussian, r) =
    dot(_polar_projection(g.a, r), _polar_projection(g.b, r)) * _gauss(g, r)

(ψ::Wavefunction)(r::AbstractVector) =
    sum(ψ.c[i] * _eval(ψ.basis.functions[i], r) for i in eachindex(ψ.c))

"""
    wavefunction(sol::Solution; state = sol.state) -> Wavefunction

Build the callable [`Wavefunction`](@ref) for the given `state` from a
[`Solution`](@ref)'s basis and generalized-eigenvector coefficients.
"""
wavefunction(sol::Solution; state::Int = sol.state) =
    Wavefunction(getfield(sol, :basis), getfield(sol, :coefficients)[:, state])
