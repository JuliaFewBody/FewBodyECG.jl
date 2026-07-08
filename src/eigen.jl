# =============================================================================
# Incremental arrowhead eigensolver for the stochastic variational method.
#
# This is the computational heart of the SVM as described in Suzuki & Varga,
# *Stochastic Variational Approach to Quantum-Mechanical Few-Body Problems*
# (LNP m54, 1998), Chapter 3-4.  It replaces the per-candidate LAPACK
# generalised eigensolve with the incremental update of Theorem 3.5.
#
# Setup.  We solve the generalised eigenproblem H c = ε S c for a non-orthogonal
# Gaussian basis {ψ₁,…,ψ_k}.  We maintain the eigendecomposition in factored
# form: eigenvalues ε₁≤…≤ε_k and a matrix Q (columns = eigenvectors cᵢ in the
# Gaussian basis) such that
#
#       Qᵀ S Q = I            (S-orthonormal)
#       Qᵀ H Q = diag(ε).
#
# Adding ψ_{k+1}.  Let s_col, h_col be its S- and H-overlaps with ψ₁…ψ_k and
# s_diag, h_diag its self-overlaps.  Transform into the eigenbasis:
#
#       s̃ = Qᵀ s_col,   h̃ = Qᵀ h_col.
#
# Gram-Schmidt the new function against the S-orthonormal eigenvectors φᵢ:
#
#       g² = s_diag − s̃ᵀs̃        (squared S-norm of the orthogonal residual)
#
# If g² ≤ 0 the candidate is (numerically) linearly dependent — this is the
# *exact* independence test, replacing the overlap-ratio heuristic.  In the
# enlarged S-orthonormal basis {φ₁,…,φ_k,φ_{k+1}} the Hamiltonian becomes an
# arrowhead matrix
#
#       M = [ diag(ε)   b ]        b[i] = (h̃[i] − s̃[i] ε[i]) / g
#           [   bᵀ      α ]        α    = (h_diag − 2 s̃ᵀh̃ + Σ s̃[i]²ε[i]) / g²
#
# whose eigenvalues are the roots of the secular equation (Eq. 3.25)
#
#       f(λ) = α − λ − Σ_i b[i]² / (ε[i] − λ) = 0.
#
# For *scoring* a candidate we need only the smallest root (ground state),
# which lies below ε₁ on a strictly monotone branch — globally safe to bracket.
# For *committing* a candidate we solve the full arrowhead (all eigenpairs) and
# update (ε, Q).
# =============================================================================

"""
    SVMEigen

In-place container for the incrementally maintained generalised
eigendecomposition of an ECG basis, kept in *whitened* form for numerical
stability.  We store the upper-triangular Cholesky factor `R` of the overlap
(`S = RᵀR`), the basis Hamiltonian `H`, and the orthonormal eigenvectors `W` of
the whitened Hamiltonian `H̃ = R⁻ᵀHR⁻¹` with eigenvalues `ε`.  The
Gaussian-basis (generalised) eigenvectors are `R⁻¹W` — see [`coefficients`](@ref).

Whitening once via Cholesky and then maintaining `W` as a genuine orthogonal
matrix avoids the progressive loss of S-orthonormality that plagues a directly
S-orthonormal basis: products of orthogonal matrices stay orthogonal, whereas
repeated Gram-Schmidt against S does not.
"""
mutable struct SVMEigen
    R::Matrix{Float64}      # upper-triangular Cholesky factor, S = RᵀR
    H::Matrix{Float64}      # basis Hamiltonian
    W::Matrix{Float64}      # orthonormal eigenvectors of H̃ = R⁻ᵀHR⁻¹
    ε::Vector{Float64}      # eigenvalues (ascending)
    k::Int
end

SVMEigen() = SVMEigen(
    Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
    Matrix{Float64}(undef, 0, 0), Float64[], 0,
)

"""
    coefficients(eig::SVMEigen) -> Matrix

Generalised eigenvectors in the Gaussian basis, `c = R⁻¹W`, satisfying
`cᵀ S c = I` and `cᵀ H c = diag(ε)`.  Column `j` is the coefficient vector of
the `j`-th state.
"""
function coefficients(eig::SVMEigen)
    eig.k == 0 && return Matrix{Float64}(undef, 0, 0)
    return UpperTriangular(eig.R) \ eig.W
end

"""
    arrowhead_secular(ε, b, α, λ)

Value of the arrowhead secular function f(λ) = α − λ − Σᵢ b[i]²/(ε[i]−λ).
"""
@inline function arrowhead_secular(ε::AbstractVector, b::AbstractVector, α::Real, λ::Real)
    s = α - λ
    @inbounds for i in eachindex(ε)
        s -= b[i]^2 / (ε[i] - λ)
    end
    return s
end

"""
    arrowhead_secular_deriv(ε, b, λ)

Derivative f'(λ) = −1 − Σᵢ b[i]²/(ε[i]−λ)² (strictly negative).
"""
@inline function arrowhead_secular_deriv(ε::AbstractVector, b::AbstractVector, λ::Real)
    s = -1.0
    @inbounds for i in eachindex(ε)
        d = ε[i] - λ
        s -= b[i]^2 / (d * d)
    end
    return s
end

# Safeguarded Newton/bisection root finder for f on an open bracket (lo, hi)
# where f(lo) > 0 > f(hi).  f is strictly decreasing on every pole-free
# interval, so this is globally convergent.
function _solve_secular_bracket(
        ε, b, α, lo::Float64, hi::Float64;
        maxiter::Int = 200, tol::Float64 = 1.0e-14
    )
    # On entry f(lo) > 0 > f(hi); f is strictly decreasing between poles.
    λ = 0.5 * (lo + hi)
    for _ in 1:maxiter
        f = arrowhead_secular(ε, b, α, λ)
        if abs(f) < tol * (1 + abs(α) + abs(λ))
            return λ
        end
        f > 0 ? (lo = λ) : (hi = λ)
        fp = arrowhead_secular_deriv(ε, b, λ)
        λn = λ - f / fp                      # Newton step
        if !(lo < λn < hi)                   # fall back to bisection
            λn = 0.5 * (lo + hi)
        end
        abs(λn - λ) < tol * (1 + abs(λ)) && return λn
        λ = λn
    end
    return λ
end

"""
    smallest_arrowhead_eigval(ε, b, α; tol) -> Float64

Smallest eigenvalue of the arrowhead matrix `[diag(ε) b; bᵀ α]`, used to *score*
a candidate basis function.  Deflation (b[i] ≈ 0) is handled so the routine is
robust when a candidate is nearly S-orthogonal to an existing eigenvector.
`ε` must be sorted ascending.
"""
function smallest_arrowhead_eigval(
        ε::AbstractVector, b::AbstractVector, α::Real;
        tol::Float64 = 1.0e-12
    )
    k = length(ε)
    scale = max(1.0, maximum(abs, ε; init = 0.0), abs(α))
    # Active (coupled) indices; deflated diagonals are themselves eigenvalues.
    active = [i for i in 1:k if b[i]^2 > tol * scale^2]
    deflated_min = Inf
    @inbounds for i in 1:k
        if b[i]^2 <= tol * scale^2
            deflated_min = min(deflated_min, ε[i])
        end
    end
    if isempty(active)
        return min(isempty(ε) ? Inf : minimum(ε), α)
    end
    εa = @view ε[active]
    ba = @view b[active]
    hi = εa[1] - tol * scale                 # just below the lowest active pole
    lo = min(εa[1], α) - (norm(ba) + scale)  # f(lo) > 0 guaranteed by enlarging
    while arrowhead_secular(εa, ba, α, lo) < 0
        lo -= (norm(ba) + scale)
    end
    root = _solve_secular_bracket(εa, ba, α, lo, hi)
    return min(root, deflated_min)
end

"""
    _lowner_border(d, λ, b_src) -> b̂

Gu-Eisenstat / Löwner reconstruction of the arrowhead border.  Given the
diagonal poles `d` (ascending) and the *computed* eigenvalues `λ` of the
arrowhead, returns the border `b̂` for which `d` and `λ` are exactly poles and
roots:

    b̂[i]² = −∏_j (d[i]−λ[j]) / ∏_{l≠i} (d[i]−d[l])      (positive by interlacing)

Computed in log-space to avoid over/underflow, with the sign of `b_src[i]`
preserved.  Building eigenvectors from `b̂` (rather than the raw border) yields
vectors orthogonal to working precision even when eigenvalues nearly coincide —
the key to keeping `QᵀSQ = I` over many incremental steps.
"""
function _lowner_border(d::AbstractVector, λ::AbstractVector, b_src::AbstractVector)
    m = length(d)
    b̂ = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        logval = 0.0
        for j in eachindex(λ)
            logval += log(abs(d[i] - λ[j]))
        end
        for l in 1:m
            l == i && continue
            logval -= log(abs(d[i] - d[l]))
        end
        b̂[i] = flipsign(exp(0.5 * logval), b_src[i])
    end
    return b̂
end

"""
    full_arrowhead_eigen(ε, b, α) -> (λ, V)

Full eigendecomposition of the (k+1)×(k+1) arrowhead matrix `[diag(ε) b; bᵀ α]`.
Returns sorted eigenvalues `λ` and orthonormal eigenvectors `V` (columns).
`ε` must be sorted ascending.  Eigenvalues interlace the poles `ε`, so each root
is bracketed by consecutive poles and found by safeguarded bisection/Newton.
Eigenvectors use the Löwner-reconstructed border (see [`_lowner_border`](@ref))
so they stay orthogonal even for nearly coincident eigenvalues.  Deflated
coordinates (b[i] ≈ 0) contribute the unit eigenvector eᵢ.
"""
function full_arrowhead_eigen(
        ε::AbstractVector{Float64}, b::AbstractVector{Float64}, α::Float64;
        tol::Float64 = 1.0e-12
    )
    k = length(ε)
    n = k + 1
    scale = max(1.0, maximum(abs, ε; init = 0.0), abs(α))
    deflated = [i for i in 1:k if b[i]^2 <= tol * scale^2]
    active = [i for i in 1:k if b[i]^2 > tol * scale^2]

    λ = Vector{Float64}(undef, n)
    V = zeros(Float64, n, n)
    slot = 1

    # Deflated coordinates: eigenvalue ε[i], eigenvector eᵢ.
    for i in deflated
        λ[slot] = ε[i]
        V[i, slot] = 1.0
        slot += 1
    end

    if isempty(active)
        # Pure diagonal plus isolated corner.
        λ[slot] = α
        V[n, slot] = 1.0
    else
        εa = ε[active]
        ba = b[active]
        m = length(active)
        # Roots: one below εa[1], one in each (εa[j], εa[j+1]), one above εa[m].
        roots = Vector{Float64}(undef, m + 1)
        for j in 0:m
            lo = j == 0 ? εa[1] - (norm(ba) + scale) : εa[j] + tol * scale
            hi = j == m ? εa[m] + (norm(ba) + scale) : εa[j + 1] - tol * scale
            if j == 0
                while arrowhead_secular(εa, ba, α, lo) < 0
                    lo -= (norm(ba) + scale)
                end
            end
            if j == m
                while arrowhead_secular(εa, ba, α, hi) > 0
                    hi += (norm(ba) + scale)
                end
            end
            roots[j + 1] = _solve_secular_bracket(εa, ba, α, lo, hi)
        end
        # Löwner-stabilised eigenvectors: v[i] = b̂[i]/(εa[i]−λ), corner = −1.
        b̂ = _lowner_border(εa, roots, ba)
        for root in roots
            for (jj, i) in enumerate(active)
                V[i, slot] = b̂[jj] / (εa[jj] - root)
            end
            V[n, slot] = -1.0
            V[:, slot] ./= norm(@view V[:, slot])
            λ[slot] = root
            slot += 1
        end
    end

    # Sort ascending.
    p = sortperm(λ)
    return λ[p], V[:, p]
end

"""
    _whiten_candidate(eig, s_col, h_col, s_diag, h_diag) -> (β, ω, ρ², y, r)

Append a candidate to the *whitened* problem.  `r = R⁻ᵀ s_col` is the new
Cholesky column and `ρ² = s_diag − rᵀr` its squared diagonal (≤ 0 ⇒ linearly
dependent — the exact independence test).  In the eigenbasis of the current
whitened Hamiltonian the new row/column appears as an arrowhead with border `β`
and corner `ω`; these feed the secular solver.  `y = R⁻¹r` is reused by the
commit path.  All triangular solves are O(k²).
"""
function _whiten_candidate(
        eig::SVMEigen, s_col::AbstractVector, h_col::AbstractVector,
        s_diag::Real, h_diag::Real
    )
    k = eig.k
    if k == 0
        return (Float64[], h_diag / s_diag, float(s_diag), Float64[], Float64[])
    end
    Ru = UpperTriangular(eig.R)
    r = Ru' \ s_col                         # solve Rᵀ r = s_col   (lower-tri)
    ρ2 = s_diag - dot(r, r)
    if ρ2 <= 0
        return (nothing, 0.0, ρ2, nothing, nothing)
    end
    ρ = sqrt(ρ2)
    y = Ru \ r                              # R⁻¹ r
    Hy = eig.H * y
    ω = (dot(y, Hy) - 2 * dot(y, h_col) + h_diag) / ρ2
    z = (Ru' \ (h_col .- Hy)) ./ ρ          # R⁻ᵀ(h_col − Hy)/ρ
    β = eig.W' * z
    return (β, ω, ρ2, y, r)
end

"""
    score_candidate(eig, s_col, h_col, s_diag, h_diag; state=1, min_resid_ratio=0) -> Float64 or nothing

Cheap O(k²) ground-state (or `state`-th) energy if the candidate were appended,
without committing.  Returns `nothing` if the candidate is linearly dependent,
or if its Cholesky residual `ρ²` is below `min_resid_ratio · s_diag` (a
principled overlap-threshold independence cut; default 0 ⇒ only exact dependence
is rejected).  For `state == 1` this uses the monotone smallest-root branch.
"""
function score_candidate(
        eig::SVMEigen, s_col, h_col, s_diag, h_diag;
        state::Int = 1, min_resid_ratio::Real = 0.0
    )
    β, ω, ρ2, _, _ = _whiten_candidate(eig, s_col, h_col, s_diag, h_diag)
    ρ2 <= max(0.0, min_resid_ratio * s_diag) && return nothing
    if eig.k == 0
        return ω                       # 1×1 problem: ε = h_diag/s_diag
    end
    if state == 1
        return smallest_arrowhead_eigval(eig.ε, β, ω)
    else
        λ, _ = full_arrowhead_eigen(eig.ε, collect(β), ω)
        return λ[min(state, length(λ))]
    end
end

"""
    commit_candidate!(eig, s_col, h_col, s_diag, h_diag)

Append a candidate to the basis, updating `(R, H, W, ε)` in place via the
whitened arrowhead eigen-update.  Returns the new eigenvalues, or `nothing` if
linearly dependent.  `W` is updated as `blockdiag(W,1)·V` with `V` the orthogonal
arrowhead eigenvectors, so orthogonality is preserved to working precision.
"""
function commit_candidate!(eig::SVMEigen, s_col, h_col, s_diag, h_diag)
    k = eig.k
    if k == 0
        eig.R = reshape([sqrt(float(s_diag))], 1, 1)
        eig.H = reshape([float(h_diag)], 1, 1)
        eig.W = reshape([1.0], 1, 1)
        eig.ε = [h_diag / s_diag]
        eig.k = 1
        return eig.ε
    end
    β, ω, ρ2, _, r = _whiten_candidate(eig, s_col, h_col, s_diag, h_diag)
    ρ2 <= 0 && return nothing
    ρ = sqrt(ρ2)
    λ, V = full_arrowhead_eigen(eig.ε, collect(β), ω)

    # W_new = blockdiag(W, 1) · V  (orthogonal × orthogonal = orthogonal).
    Vtop = @view V[1:k, :]
    Vbot = @view V[k + 1, :]
    Wnew = Matrix{Float64}(undef, k + 1, k + 1)
    Wnew[1:k, :] = eig.W * Vtop
    Wnew[k + 1, :] = Vbot

    # R_new = [R r; 0 ρ],  H_new = [H h_col; h_colᵀ h_diag].
    Rnew = Matrix{Float64}(undef, k + 1, k + 1)
    Rnew[1:k, 1:k] = eig.R
    Rnew[1:k, k + 1] = r
    Rnew[k + 1, 1:k] .= 0.0
    Rnew[k + 1, k + 1] = ρ
    Hnew = Matrix{Float64}(undef, k + 1, k + 1)
    Hnew[1:k, 1:k] = eig.H
    Hnew[1:k, k + 1] = h_col
    Hnew[k + 1, 1:k] = h_col
    Hnew[k + 1, k + 1] = h_diag

    eig.R = Rnew
    eig.H = Hnew
    eig.W = Wnew
    eig.ε = λ
    eig.k = k + 1
    return eig.ε
end
