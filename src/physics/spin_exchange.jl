"""
    SpinExchangeOperator(i, j)

Swap the spin projections on particles `i` and `j`, leaving coordinates and
isospin unchanged. For spin ½, ``P^σ_{ij} = (1 + σ_i⋅σ_j)/2``:
its eigenvalue is +1 in a triplet and −1 in a singlet. No fermion sign is
included; antisymmetrization applies to the full spatial-spin-isospin state.

Use a [`SpinGaussian`](@ref) basis. `V * SpinExchangeOperator(i, j)` multiplies
a Jacobi-frame radial operator by spin exchange. Products act right-to-left,
including products with [`IsospinExchangeOperator`](@ref) and
[`PermutationOperator`](@ref). These are the central-force components in
Suzuki–Varga §11.1, Eqs. (11.3)–(11.5).
"""
struct SpinExchangeOperator <: FewBodyHamiltonians.Operator
    i::Int
    j::Int
    function SpinExchangeOperator(i::Integer, j::Integer)
        _check_pair(i, j)
        return new(Int(i), Int(j))
    end
end

"""
    IsospinExchangeOperator(i, j)

Swap the nucleon isospin projections on particles `i` and `j`, leaving
coordinates and spin unchanged. For isospin ½,
``P^τ_{ij} = (1 + τ_i⋅τ_j)/2``. Its eigenvalues are +1 for
`T=1` and −1 for `T=0`.

Both [`SpinGaussian`](@ref) functions must carry an [`IsospinState`](@ref).
`V * IsospinExchangeOperator(i, j)` is a radial operator times isospin exchange;
it composes with spin and spatial exchanges in the same way.
"""
struct IsospinExchangeOperator <: FewBodyHamiltonians.Operator
    i::Int
    j::Int
    function IsospinExchangeOperator(i::Integer, j::Integer)
        _check_pair(i, j)
        return new(Int(i), Int(j))
    end
end

const _InternalExchange = Union{SpinExchangeOperator, IsospinExchangeOperator}
Base.:*(op::FewBodyHamiltonians.Operator, P::_InternalExchange) = _PermutedOperator(op, P)

_requires_spin_gaussian(op) =
    throw(ArgumentError("$(nameof(typeof(op))) requires SpinGaussian functions with the corresponding internal states"))

_compute_matrix_element(::Rank0Gaussian, ::Rank0Gaussian, op::_InternalExchange) =
    _requires_spin_gaussian(op)
_compute_matrix_element(::Rank0Gaussian, ::Rank0Gaussian, op::_PermutedOperator{<:Any, <:_InternalExchange}) =
    _requires_spin_gaussian(op.permutation)

function _check_exchange_sites(op::_InternalExchange, ket::SpinGaussian)
    n = length(ket.spin.projections)
    max(op.i, op.j) ≤ n || throw(ArgumentError("exchange particle indices must lie in 1:$n"))
    return nothing
end

function _exchange_ket(ket::SpinGaussian, op::SpinExchangeOperator)
    _check_exchange_sites(op, ket)
    projections = copy(ket.spin.projections)
    projections[op.i], projections[op.j] = projections[op.j], projections[op.i]
    return SpinGaussian(ket.orbital, SpinState(projections); isospin = ket.isospin)
end

function _exchange_ket(ket::SpinGaussian, op::IsospinExchangeOperator)
    _check_exchange_sites(op, ket)
    ket.isospin === nothing && throw(ArgumentError("IsospinExchangeOperator requires an IsospinState on both SpinGaussian functions"))
    projections = copy(ket.isospin.projections)
    projections[op.i], projections[op.j] = projections[op.j], projections[op.i]
    return SpinGaussian(ket.orbital, ket.spin; isospin = IsospinState(projections))
end

_exchange_ket(ket::SpinGaussian, P::PermutationOperator) =
    SpinGaussian(_permuted_ket(ket.orbital, P), ket.spin; isospin = ket.isospin)

_compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian, op::_InternalExchange) =
    _compute_matrix_element(bra, _exchange_ket(ket, op))

_compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian, op::_PermutedOperator) =
    _compute_matrix_element(bra, _exchange_ket(ket, op.permutation), op.operator)

# Spin/isospin labels are fixed while only the orbital shifts are differentiated.
function _polarized_spin_element(bra::SpinGaussian, ket::SpinGaussian, op)
    element(g, h) = _compute_matrix_element(
        SpinGaussian(g, bra.spin; isospin = bra.isospin),
        SpinGaussian(h, ket.spin; isospin = ket.isospin), op
    )
    return _polarized_element(element, bra.orbital, ket.orbital)
end
