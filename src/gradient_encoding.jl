# Flat parameter vectors of the gradient methods.
#
# Each basis function contributes one block:
#   Rank0: [log-Cholesky of A]
#   Rank1: [log-Cholesky of A | u]  (u is the z column of the polarization)
# The log-Cholesky encoding keeps A positive definite for any parameter values.
# Every function is unshifted.  At zero shifts the energy is even in each
# function's shift, so a shift parameter would have zero gradient.  Rank1
# functions are polarized along z, which makes each one a pure L = 1, M = 0
# function; u is normalized on decode, so only its direction matters.  With a
# symmetry, every decoded orbital is wrapped in its projection.

struct _GradientLayout
    rank::Int
    n_dim::Int
    symmetry::Union{Nothing, PermutationSymmetry}
end

_n_chol(n_dim) = n_dim * (n_dim + 1) ÷ 2
_n_params(layout::_GradientLayout) =
    _n_chol(layout.n_dim) + (layout.rank == 0 ? 0 : layout.n_dim)

function _chol_to_params(L::AbstractMatrix)
    n = size(L, 1)
    params = Float64[]
    for j in 1:n
        for i in j:n
            push!(params, i == j ? log(L[i, j]) : L[i, j])
        end
    end
    return params
end

function _params_to_matrix(θ::AbstractVector, n::Int)
    T = eltype(θ)
    L = zeros(T, n, n)
    idx = 1
    for j in 1:n
        for i in j:n
            L[i, j] = (i == j) ? exp(θ[idx]) : θ[idx]
            idx += 1
        end
    end
    return Symmetric(L * L')
end

"""
    _gradient_layout(rank, symmetry, n_dim; init = nothing)

Parameter layout for a gradient optimization.  `rank = nothing` takes the rank
of `init`'s basis, or 0 without `init`.  Every function of `init` must be
representable in the layout.
"""
function _gradient_layout(rank, symmetry, n_dim::Integer; init = nothing)
    init_rank = init === nothing ? nothing : _stochastic_rank(getfield(init, :basis))
    rank = something(rank, init_rank, 0)
    rank in (0, 1) || throw(ArgumentError("gradient methods support rank 0 or 1, got $rank"))
    init_rank === nothing || rank == init_rank ||
        throw(ArgumentError("rank=$rank does not match the rank-$init_rank initial basis"))
    if init !== nothing
        check = rank == 0 ? _check_rank0_encodable : _check_rank1_encodable
        foreach(g -> check(_unprojected(g)), getfield(init, :basis).functions)
    end
    return _GradientLayout(rank, n_dim, symmetry)
end

function _check_rank0_encodable(g::Rank0Gaussian)
    iszero(g.s) || throw(ArgumentError("Rank0 functions must be unshifted for gradient methods"))
    return nothing
end

function _check_rank1_encodable(g::Rank1Gaussian)
    iszero(g.s) || throw(ArgumentError("Rank1 functions must be unshifted for gradient methods"))
    iszero(view(g.a, :, 1:2)) ||
        throw(ArgumentError("Rank1 functions must be polarized along z for gradient methods"))
    return nothing
end

function _orbital_params(g::Rank0Gaussian)
    _check_rank0_encodable(g)
    return Float64[]
end
_orbital_params(g::Rank1Gaussian) = g.a[:, 3]

function _encode_basis(layout::_GradientLayout, basis::BasisSet)
    params = Float64[]
    for g in basis.functions
        orbital = _unprojected(g)
        orbital isa (layout.rank == 0 ? Rank0Gaussian : Rank1Gaussian) ||
            throw(ArgumentError("basis function $(typeof(orbital)) does not match a rank-$(layout.rank) layout"))
        C = cholesky(Symmetric(Matrix(orbital.A)))
        append!(params, _chol_to_params(Matrix(C.L)))
        append!(params, Float64.(_orbital_params(orbital)))
    end
    return params
end

function _decode_orbital(layout::_GradientLayout, θ::AbstractVector)
    d = layout.n_dim
    T = eltype(θ)
    A = Matrix(_params_to_matrix(view(θ, 1:_n_chol(d)), d))
    rest = θ[(_n_chol(d) + 1):end]
    layout.rank == 0 && return Rank0Gaussian(A, zeros(T, d, 3))
    return Rank1Gaussian(A, rest / norm(rest), zeros(T, d, 3))
end

_wrap(::Nothing, orbital) = orbital
_wrap(symmetry::PermutationSymmetry, orbital) = ProjectedGaussian(orbital, symmetry)

function _decode_basis(layout::_GradientLayout, θ::AbstractVector, n_basis::Int)
    n_per = _n_params(layout)
    length(θ) == n_basis * n_per ||
        throw(DimensionMismatch("expected $(n_basis * n_per) parameters, got $(length(θ))"))
    return BasisSet(
        [
            _wrap(layout.symmetry, _decode_orbital(layout, view(θ, ((i - 1) * n_per + 1):(i * n_per))))
                for i in 1:n_basis
        ]
    )
end
