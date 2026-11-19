# Raw Jacobi-frame operators keep their backend-specific input checks here.
function _check_pair(i, j)
    min(i, j) ≥ 1 || throw(ArgumentError("particle indices start at 1, got ($i, $j)"))
    i != j || throw(ArgumentError("particle indices must differ, got i = j = $i"))
    return nothing
end

_check_coefficient(c) = isfinite(c) || throw(ArgumentError("coefficient must be finite, got $c"))

function _check_weight(w, name; nonzero = true)
    Base.require_one_based_indexing(w)
    !isempty(w) && all(isfinite, w) && (!nonzero || any(!iszero, w)) ||
        throw(ArgumentError("$name must be a nonempty, finite$(nonzero ? ", nonzero" : "") vector, got $w"))
    return nothing
end
