# Matrix elements of polarized Gaussians as shift derivatives of Rank0 elements.
#
# A polarization is a derivative with respect to the shift:
#     (a·r) exp(−rᵀAr + tr(sᵀr)) = ∂/∂ε exp(−rᵀAr + tr((s + εa)ᵀr)) at ε = 0,
# and (a·r)(b·r)·g is the mixed second derivative along a and b.  Any element
# ⟨bra|op|ket⟩ is therefore the mixed derivative of the Rank0 element between
# the shifted plain Gaussians, one derivative per polarization vector of the
# bra and the ket (Suzuki & Varga, Lect. Notes Phys. m54, Eq. (A.44)).

const _PolarizedGaussian = Union{Rank1Gaussian, Rank2Gaussian}
const _AnyRankGaussian = Union{Rank0Gaussian, Rank1Gaussian, Rank2Gaussian}

_polarizations(::Rank0Gaussian) = ()
_polarizations(g::Rank1Gaussian) = (g.a,)
_polarizations(g::Rank2Gaussian) = (g.a, g.b)

# Tag of the k-th shift-derivative variable.  Fixed tags make the dual-number
# types depend only on the derivative order, so compiled code is shared across
# operators and rank pairs.  A higher tag nests outside a lower one.
struct _ShiftTag{K} end
ForwardDiff.:≺(::Type{_ShiftTag{I}}, ::Type{_ShiftTag{J}}) where {I, J} = I < J
# Shift differentiation is inside any user derivative of widths or shifts.
ForwardDiff.:≺(::Type{<:ForwardDiff.Tag}, ::Type{<:_ShiftTag}) = true # COV_EXCL_LINE: constant-folded
ForwardDiff.:≺(::Type{<:_ShiftTag}, ::Type{<:ForwardDiff.Tag}) = false # COV_EXCL_LINE: constant-folded

# ∂ⁿf/∂ε₁⋯∂εₙ at ε = x0 for f taking an n-tuple of scalars: seed
# εₖ = x0 + 1·dεₖ and read off the coefficient of dε₁⋯dεₙ.  The tag-aware
# `partials` returns zero for any value that does not carry the tag, including a
# `Complex` of duals, so complex results are split into real and imaginary parts
# here.
function _mixed_derivative(f, ::Val{n}, x0) where {n}
    εs = ntuple(k -> ForwardDiff.Dual{_ShiftTag{k}}(x0, one(x0)), Val(n))
    y = f(εs)
    y isa Complex && return complex(_top_coefficient(real(y), Val(n)), _top_coefficient(imag(y), Val(n)))
    return _top_coefficient(y, Val(n))
end
_top_coefficient(y, ::Val{0}) = y
_top_coefficient(y, ::Val{k}) where {k} =
    _top_coefficient(ForwardDiff.partials(_ShiftTag{k}, y, 1), Val(k - 1))

# s + Σᵢ εᵢ Pᵢ for a tuple of polarizations P and matching scalars ε.
_displaced_shift(s, P::Tuple{}, ε::Tuple{}) = s
_displaced_shift(s, P::Tuple, ε::Tuple) = s + sum(map((p, e) -> e .* p, P, ε))

"""
    _polarized_element(element, bra, ket)

Matrix element between two Gaussians of any rank, computed as the mixed shift
derivative of `element(::Rank0Gaussian, ::Rank0Gaussian)`, the corresponding
Rank0 matrix element.  `element` must be smooth in the shifts at the shifts of
`bra` and `ket`.
"""
function _polarized_element(element, bra::_AnyRankGaussian, ket::_AnyRankGaussian)
    P, Q = _polarizations(bra), _polarizations(ket)
    nb, nk = length(P), length(Q)
    A, B = parent(bra.A), parent(ket.A)
    function shifted(εs)
        εb = ntuple(i -> εs[i], Val(nb))
        εk = ntuple(j -> εs[nb + j], Val(nk))
        return element(
            Rank0Gaussian(A, _displaced_shift(bra.s, P, εb)),
            Rank0Gaussian(B, _displaced_shift(ket.s, Q, εk)),
        )
    end
    x0 = zero(float(promote_type(eltype(bra.s), eltype(ket.s))))
    return _mixed_derivative(shifted, Val(nb + nk), x0)
end

_compute_matrix_element(bra::_PolarizedGaussian, ket::_AnyRankGaussian) =
    _polarized_element(_compute_matrix_element, bra, ket)
_compute_matrix_element(bra::Rank0Gaussian, ket::_PolarizedGaussian) =
    _polarized_element(_compute_matrix_element, bra, ket)

_compute_matrix_element(bra::_PolarizedGaussian, ket::_AnyRankGaussian, op::FewBodyHamiltonians.Operator) =
    _polarized_element((g, h) -> _compute_matrix_element(g, h, op), bra, ket)
_compute_matrix_element(bra::Rank0Gaussian, ket::_PolarizedGaussian, op::FewBodyHamiltonians.Operator) =
    _polarized_element((g, h) -> _compute_matrix_element(g, h, op), bra, ket)
