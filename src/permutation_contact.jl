"""
    PermutationOperator(masses, p)

Spatial particle permutation with `(Pψ)(x) = ψ(x[p, :])`, where each row of
`x` is a particle position and `p` contains each particle index exactly once.
`masses` defines the same mass-scaled Jacobi frame as [`jacobi_transform`](@ref).
Spin and isospin states are not permuted.

The relative-coordinate action is `T = J * U[p, :]`. Its Rank0 matrix element
is an overlap with ket data `T' * A * T` and `T' * s`; polarized elements follow
by shift differentiation (Suzuki & Varga, Table 7.1 and Appendix A.3).

For a Jacobi-frame operator, `op * P` applies the operator after the spatial
permutation, preserving the ket's spin and isospin labels. For example,
`GaussianOperator(V, γ, w) * P` is a Gaussian radial potential times spatial
exchange, as used for a Majorana term. A cycle or an operator times a
permutation need not be Hermitian; [`build_operator_matrix`](@ref) preserves
every matrix entry. For a transposition, `(S ± M) / 2` are the overlap matrices
of the even/odd projected spatial functions, with `S` the overlap matrix and
`M` the permutation matrix.
"""
struct PermutationOperator <: FewBodyHamiltonians.Operator
    T::Matrix{Float64}
    function PermutationOperator(masses::AbstractVector{<:Real}, p::AbstractVector{<:Integer})
        Base.require_one_based_indexing(masses, p)
        n = length(masses)
        n ≥ 2 || throw(ArgumentError("a permutation needs at least two particles"))
        length(p) == n && sort(collect(p)) == collect(1:n) ||
            throw(ArgumentError("p must contain each particle index in 1:$n exactly once"))
        m = Float64.(masses)
        all(x -> isfinite(x) && x > 0, m) ||
            throw(ArgumentError("masses must be finite and positive"))
        J, U = jacobi_transform(m)
        return new(J * U[p, :])
    end
end

"""
    ContactDensityOperator(w, r=zeros(3); coefficient=1)

Three-dimensional point density `coefficient * δ³(w' * x - r)` in Jacobi
coordinates. `w` selects the oriented pair vector, and `r` is a Cartesian
three-vector. For particles `i, j`, construct `w = U' * (eᵢ - eⱼ)` with
[`jacobi_transform`](@ref). `r = zeros(3)` gives the contact density.

The normalized expectation has units of inverse volume. Varying `r` gives the
spatial pair density; its integral over `d³r` equals one for `coefficient=1`.
A radial distribution requires an angular integral: `4πr²` times one direction
is valid only for an isotropic state. This is a direct delta expectation;
ECG contact estimates can converge more slowly than variational energies.

The Rank0 element is `coefficient * M * (β/π)^(3/2) * exp(-β*|r-μ|²)`,
where `M` is the overlap, `β = 1 / (w' * inv(A+B) * w)` and
`μ = w' * inv(A+B) * (s+t) / 2`. All Rank1/Rank2 combinations follow by
shift differentiation (Suzuki & Varga, Table 7.1 and Eq. (A.44)).
"""
struct ContactDensityOperator{T <: Real} <: FewBodyHamiltonians.PotentialTerm
    coefficient::T
    w::Vector{T}
    r::Vector{T}
    function ContactDensityOperator(
            w::AbstractVector{<:Real}, r::AbstractVector{<:Real} = zeros(3);
            coefficient::Real = 1
        )
        Base.require_one_based_indexing(w, r)
        !isempty(w) && all(isfinite, w) && any(!iszero, w) ||
            throw(ArgumentError("w must be a nonempty, finite, nonzero weight vector"))
        length(r) == 3 && all(isfinite, r) ||
            throw(ArgumentError("r must be a finite Cartesian three-vector"))
        isfinite(coefficient) || throw(ArgumentError("coefficient must be finite"))
        T = promote_type(Float64, eltype(w), eltype(r), typeof(coefficient))
        return new{T}(T(coefficient), Vector{T}(w), Vector{T}(r))
    end
end

struct _PermutedOperator{O <: FewBodyHamiltonians.Operator, P <: FewBodyHamiltonians.Operator} <: FewBodyHamiltonians.Operator
    operator::O
    permutation::P
end

Base.:*(op::FewBodyHamiltonians.Operator, P::PermutationOperator) = _PermutedOperator(op, P)

function _permuted_ket(ket::Rank0Gaussian, P::PermutationOperator)
    size(P.T) == size(ket.A) ||
        throw(DimensionMismatch("permutation and Gaussian must use the same Jacobi dimension"))
    return Rank0Gaussian(P.T' * ket.A * P.T, P.T' * ket.s)
end

_compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, P::PermutationOperator) =
    _compute_matrix_element(bra, _permuted_ket(ket, P))

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::_PermutedOperator)
    op.permutation isa PermutationOperator || return _requires_spin_gaussian(op.permutation)
    return _compute_matrix_element(bra, _permuted_ket(ket, op.permutation), op.operator)
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::ContactDensityOperator)
    length(op.w) == size(bra.A, 1) == size(ket.A, 1) ||
        throw(DimensionMismatch("contact-density weight length must equal Gaussian dimension"))
    S = bra.A + ket.A
    R = inv(S)
    variance = dot(op.w, R * op.w)
    variance > 0 || throw(ArgumentError("contact-density weight has nonpositive Gaussian variance"))
    v = bra.s + ket.s
    source = transpose(op.w) * R * v
    # Remove the constrained source before exponentiation to avoid Inf * 0
    # and cancellation between large shift-squared terms at finite contact.
    transverse = v - op.w * (source / variance)
    exponent = 0.25 * _superdot(transverse, R, transverse) +
        (dot(vec(source), op.r) - sum(abs2, op.r)) / variance
    n = size(S, 1)
    normalization = (π^(n - 1) / (det(S) * variance))^(3 / 2)
    return op.coefficient * normalization * exp(exponent)
end
