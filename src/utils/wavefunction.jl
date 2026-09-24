"""
    Wavefunction

Callable variational wavefunction `ψ(r) = Σᵢ cᵢ gᵢ(r)` in **Jacobi
coordinates** (mass-weighted: the package's Jacobi transform normalises each
relative coordinate by √μ — see `FewBodyECG.jacobi_transform`).  Obtained from
[`wavefunction`](@ref); plot with `plot(ψ; coord = i)` or sample with
[`radial_profile`](@ref).

`ψ(r)` accepts either an `N × 3` matrix of Cartesian positions (row = Jacobi
coordinate, column = `x, y, z`) or a length-`N` vector, which places every
Jacobi coordinate on the `z` axis: `ψ(v) == ψ([0 0 v[1]; …])`.
"""
struct Wavefunction
    basis::BasisSet
    c::AbstractVector{<:Number}
end

# `r` is an N×3 supervector of Cartesian positions; `a·r = tr(aᵀr)` = dot(a, r).
_gauss(g, r) = exp(-_superdot(r, g.A, r) + dot(g.s, r))
_eval(g::Rank0Gaussian, r) = _gauss(g, r)
_eval(g::Rank1Gaussian, r) = dot(g.a, r) * _gauss(g, r)
_eval(g::Rank2Gaussian, r) = dot(g.a, r) * dot(g.b, r) * _gauss(g, r)

function (ψ::Wavefunction)(r::AbstractMatrix{<:Real})
    Base.require_one_based_indexing(r)
    _check_supervector(r, size(first(ψ.basis.functions).A, 1), "r")
    return sum(ψ.c[i] * _eval(ψ.basis.functions[i], r) for i in eachindex(ψ.c))
end

function (ψ::Wavefunction)(r::AbstractVector{<:Real})
    Base.require_one_based_indexing(r)
    return ψ(_supervector(r))
end

"""
    wavefunction(sol::Solution; state = sol.state) -> Wavefunction

Build the callable [`Wavefunction`](@ref) for the given `state` from a
[`Solution`](@ref)'s basis and generalized-eigenvector coefficients.
"""
wavefunction(sol::Solution; state::Int = sol.state) =
    Wavefunction(getfield(sol, :basis), getfield(sol, :coefficients)[:, state])
