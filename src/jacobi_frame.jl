# Transform of a `Hamiltonian` of physical terms to the Jacobi-frame operators
# the matrix elements and solvers act on.  A pair distance rᵢⱼ = |xᵢ - xⱼ| is
# |wᵀr| in Jacobi coordinates with w = U'eᵢⱼ, and a particle-space quadratic
# form xᵀGx is rᵀ(U'GU)r.

function _pair_weight(i, j, U, N)
    max(i, j) ≤ N || throw(ArgumentError("particle index $(max(i, j)) out of range 1:$N"))
    e = zeros(N)
    e[i], e[j] = 1.0, -1.0
    return U' * e
end

function _jacobi_quadratic(G, U)
    W = U' * G * U
    return (W + W') / 2
end

_jacobi_operator(t::Kinetic, _, _) = KineticOperator(t.masses)
_jacobi_operator(t::Coulomb, U, N) = CoulombOperator(t.coefficient, _pair_weight(t.i, t.j, U, N))
_jacobi_operator(t::Oscillator, U, N) = OscillatorOperator(t.coefficient, _pair_weight(t.i, t.j, U, N))
function _jacobi_operator(t::Gaussian, U, N)
    if length(t.factors) == 1
        i, j, γ = only(t.factors)
        return GaussianOperator(t.coefficient, γ, _pair_weight(i, j, U, N))
    end
    return ManyBodyGaussianOperator(t.coefficient, _jacobi_quadratic(exponent_matrix(t, N), U))
end
function _jacobi_operator(t::ManyBodyGaussian, U, N)
    n = size(t.G, 1)
    n == N || throw(
        ArgumentError("ManyBodyGaussian G is $n×$n, but Kinetic has $N masses")
    )
    return ManyBodyGaussianOperator(t.coefficient, _jacobi_quadratic(t.G, U))
end
function _jacobi_operator(t::Custom, U, N)
    f, c = t.f, t.coefficient
    g = isone(c) ? f : (r -> c * f(r))
    return NumericalPotential(g, _pair_weight(t.i, t.j, U, N); rtol = t.rtol, atol = t.atol, maxevals = t.maxevals)
end
_jacobi_operator(t::Tensor, U, N) =
    GaussianTensorOperator(t.coefficient, t.exponent, _pair_weight(t.i, t.j, U, N), t.i, t.j; traceless = t.traceless)
_jacobi_operator(t::SpinOrbit, U, N) =
    GaussianSpinOrbitOperator(t.coefficient, t.exponent, _pair_weight(t.i, t.j, U, N), t.i, t.j)

# (operators, masses, units) of `H`: its Jacobi-frame operators in the order
# the terms were written, the masses of its `Kinetic` term, and their units.
function _to_jacobi(H::Hamiltonian)
    K = kinetic(validate(H))
    N = length(K.masses)
    _, U = jacobi_transform(K.masses)
    ops = FewBodyHamiltonians.Operator[_jacobi_operator(t, U, N) for t in H.terms]
    return ops, K.masses, K.units
end

build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, H::Hamiltonian) =
    build_hamiltonian_matrix(basis, first(_to_jacobi(H)))

"""
    FewBodyECG.coulomb_weights(H::Hamiltonian) -> Vector{Vector{Float64}}

Return the Jacobi-frame weight vectors of the Coulomb terms of `H`, in the
order they were written. Useful for manual basis construction: with pair
widths `b`, the correlation matrix ``A = \\sum_k w_k w_k^T / b_k^2`` gives a
`Rank0Gaussian` whose exponent is ``-\\sum_k r_k^2 / b_k^2`` in the pair
distances ``r_k``:

```julia
using FewBodyECG: coulomb_weights
w = coulomb_weights(Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2, -1, -1]))
b = [1.0, 2.0, 1.5]                       # one width per pair
A = sum(w[k] * w[k]' / b[k]^2 for k in eachindex(w, b))
g = Rank0Gaussian(A, zeros(length(w[1])))
```
"""
coulomb_weights(H::Hamiltonian) = [op.w for op in first(_to_jacobi(H)) if op isa CoulombOperator]
