"""
    DipoleOperator(w; component=3, coefficient=1)
    DipoleOperator(masses, charges; component=3, coefficient=1)

Cartesian electric dipole component in Jacobi coordinates:
`coefficient * sum(w[k] * x[k, component])`. Components `1,2,3` are `x,y,z`.
The physical constructor uses `D = Σᵢ qᵢ(rᵢ - Rcm)`, with the mass-weighted
center of mass removed even for a charged system. Use the same masses as the
Hamiltonian; charges and lengths set the units of the returned moment.

Use [`build_operator_matrix`](@ref) with separate bra and ket bases for
transition moments. These are Cartesian matrix elements, without spherical
harmonic or reduced-matrix-element factors. Rank1/Rank2 elements follow from
the Rank0 Gaussian mean by shift differentiation.
"""
struct DipoleOperator{T <: Real} <: FewBodyHamiltonians.Operator
    w::Vector{T}
    component::Int
    coefficient::T
    function DipoleOperator(w::AbstractVector{<:Real}; component::Integer = 3, coefficient::Real = 1)
        Base.require_one_based_indexing(w)
        !isempty(w) && all(isfinite, w) || throw(ArgumentError("w must be a nonempty finite vector"))
        component in 1:3 || throw(ArgumentError("component must be 1, 2, or 3 (x, y, z)"))
        isfinite(coefficient) || throw(ArgumentError("coefficient must be finite"))
        T = promote_type(Float64, eltype(w), typeof(coefficient))
        return new{T}(Vector{T}(w), Int(component), T(coefficient))
    end
end

# U is a pseudoinverse, so it does not itself set the mass-weighted COM to zero.
function _multipole_frame(masses::AbstractVector{<:Real}, charges::AbstractVector{<:Real})
    Base.require_one_based_indexing(masses, charges)
    length(masses) == length(charges) || throw(DimensionMismatch("masses and charges must have the same length"))
    length(masses) ≥ 2 || throw(ArgumentError("multipoles need at least two particles"))
    all(m -> isfinite(m) && m > 0, masses) || throw(ArgumentError("masses must be finite and positive"))
    all(isfinite, charges) || throw(ArgumentError("charges must be finite"))
    m, q = Float64.(masses), collect(charges)
    _, U = jacobi_transform(m)
    return U, m / sum(m), q
end

function DipoleOperator(masses::AbstractVector{<:Real}, charges::AbstractVector{<:Real}; kwargs...)
    U, mass_fractions, q = _multipole_frame(masses, charges)
    return DipoleOperator(U' * (q - sum(q) * mass_fractions); kwargs...)
end

"""
    QuadrupoleOperator(G; components=(3,3), coefficient=1)
    QuadrupoleOperator(masses, charges; components=(3,3), coefficient=1)

Cartesian traceless electric quadrupole component. For `components=(a,b)`,
the physical constructor gives
`coefficient * Σᵢ qᵢ(3ρᵢₐρᵢᵦ - δₐᵦ|ρᵢ|²)`, where `ρᵢ = rᵢ - Rcm`.
Components `1,2,3` are `x,y,z`; there is no factor of `1/2` and no spherical
harmonic normalization. The units are charge times length squared.

In the Jacobi-frame constructor, `G` is a finite real symmetric matrix and
the operator is `coefficient * (3xₐ'G*xᵦ - δₐᵦΣ꜀ x꜀'G*x꜀)`.
The isotropic Gaussian covariance cancels from the traceless Rank0 element;
all polarized elements follow by shift differentiation.
"""
struct QuadrupoleOperator{T <: Real} <: FewBodyHamiltonians.Operator
    G::Matrix{T}
    components::Tuple{Int, Int}
    coefficient::T
    function QuadrupoleOperator(
            G::AbstractMatrix{<:Real}; components::Tuple{Integer, Integer} = (3, 3), coefficient::Real = 1
        )
        Base.require_one_based_indexing(G)
        !isempty(G) && issymmetric(G) && all(isfinite, G) ||
            throw(ArgumentError("G must be a nonempty finite symmetric matrix"))
        all(c -> c in 1:3, components) || throw(ArgumentError("components must be in 1:3 (x, y, z)"))
        isfinite(coefficient) || throw(ArgumentError("coefficient must be finite"))
        T = promote_type(Float64, eltype(G), typeof(coefficient))
        return new{T}(Matrix{T}(G), Int.(components), T(coefficient))
    end
end

function QuadrupoleOperator(masses::AbstractVector{<:Real}, charges::AbstractVector{<:Real}; kwargs...)
    U, mass_fractions, q = _multipole_frame(masses, charges)
    relative = U .- (mass_fractions' * U)
    G = relative' * Diagonal(q) * relative
    return QuadrupoleOperator((G + G') / 2; kwargs...)
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::DipoleOperator)
    length(op.w) == size(bra.A, 1) == size(ket.A, 1) ||
        throw(DimensionMismatch("dipole weight length must equal Gaussian dimension"))
    μ = inv(bra.A + ket.A) * (bra.s + ket.s) / 2
    return op.coefficient * dot(op.w, μ[:, op.component]) * _compute_matrix_element(bra, ket)
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::QuadrupoleOperator)
    size(op.G) == size(bra.A) == size(ket.A) ||
        throw(DimensionMismatch("quadrupole matrix must match Gaussian dimension"))
    μ = inv(bra.A + ket.A) * (bra.s + ket.s) / 2
    a, b = op.components
    moment = 3 * dot(μ[:, a], op.G * μ[:, b])
    a == b && (moment -= _superdot(μ, op.G, μ))
    return op.coefficient * moment * _compute_matrix_element(bra, ket)
end
