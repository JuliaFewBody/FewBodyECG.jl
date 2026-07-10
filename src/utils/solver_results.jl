struct SolverResults{G <: GaussianBase, O <: FewBodyHamiltonians.Operator, S, T <: Number}
    basis_functions::Vector{G}
    n_basis::Int
    operators::Vector{O}
    method::Symbol
    sampler::S
    length_scale::Float64
    ground_state::Float64
    energies::Vector{Float64}
    eigenvectors::Vector{Matrix{T}}
end

function _coordinate_matrix(r::AbstractVector{<:Real})
    return _shift_matrix(r)
end

function _coordinate_matrix(r::AbstractMatrix{<:Real})
    size(r, 2) == 3 || throw(ArgumentError("r must have three Cartesian columns"))
    return r
end

function ψ₀(r, c::AbstractVector, basis_fns::AbstractVector{<:GaussianBase})
    coordinates = _coordinate_matrix(r)
    length(c) == length(basis_fns) || throw(ArgumentError("coefficient and basis lengths must match"))
    return sum(c[i] * _rank0_value(coordinates, basis_fns[i]) for i in eachindex(basis_fns))
end

function _rank0_value(r::AbstractMatrix, basis_fn::Rank0Gaussian)
    size(r) == size(basis_fn.s) || throw(ArgumentError("r and Gaussian shift sizes must match"))
    return exp(-tr(r' * basis_fn.A * r) + tr(basis_fn.s' * r))
end

_rank0_value(::AbstractMatrix, ::GaussianBase) =
    throw(ArgumentError("ψ₀ currently supports Rank0Gaussian basis functions only"))

function ψ₀(r, sr::SolverResults; state::Int = 1)
    state in axes(sr.eigenvectors[end], 2) || throw(ArgumentError("invalid state index $state"))
    return ψ₀(r, sr.eigenvectors[end][:, state], sr.basis_functions)
end

convergence(sr::SolverResults) = (1:sr.n_basis, sr.energies)
