# Two-nucleon model construction for the book nuclear-exchange tests.

# Rows are (V [MeV], γ [fm⁻²], W, M, B, H), exactly as in Table 11.1.
minnesota_rows(u) = [
    (200.0, 1.487, u / 2, (2 - u) / 2, 0.0, 0.0),
    (-178.0, 0.639, u / 4, (2 - u) / 4, u / 4, (2 - u) / 4),
    (-91.85, 0.465, u / 4, (2 - u) / 4, -u / 4, (u - 2) / 4),
]
# M_S = M_T = 0 product states span all four two-nucleon (S,T) channels.
function nucleon_basis(orbitals)
    up, down = FewBodyECG.up, FewBodyECG.down
    p, n = FewBodyECG.proton, FewBodyECG.neutron
    spins = [SpinState([up, down]), SpinState([down, up])]
    isos = [IsospinState([p, n]), IsospinState([n, p])]
    return BasisSet([SpinGaussian(g, s; isospin = t) for g in orbitals for s in spins for t in isos])
end

# Coupled singlet/triplet vectors in the product-state ordering above.
function channel_map(n, S, T)
    spin = [1.0, S == 1 ? 1.0 : -1.0] / sqrt(2)
    iso = [1.0, T == 1 ? 1.0 : -1.0] / sqrt(2)
    return kron(Matrix{Float64}(I, n, n), kron(spin, iso))
end

function pair_orbitals(L, widths, w)
    return map(widths) do α
        A, s = α * (w * w'), zeros(length(w), 3)
        L == 0 && return Rank0Gaussian(A, s)
        L == 1 && return Rank1Gaussian(A, w, s)   # physical pair z component
        return Rank2Gaussian(A, w * [1 0 0], w * [0 1 0], s)  # xy, pure L=2
    end
end

function exchange_matrix(basis, rows, w, Pr; isospin_form = false)
    Ps, Pt = SpinExchangeOperator(1, 2), IsospinExchangeOperator(1, 2)
    result = zero(FewBodyECG.build_overlap_matrix(basis))
    for (V, γ, W, M, B, H) in rows
        radial = GaussianOperator(V, γ, w)
        W != 0 && (result .+= W .* FewBodyECG.build_operator_matrix(basis, radial))
        M != 0 && (result .+= M .* FewBodyECG.build_operator_matrix(basis, radial * Pr))
        B != 0 && (result .+= B .* FewBodyECG.build_operator_matrix(basis, radial * Ps))
        if H != 0
            op, coefficient = isospin_form ? (radial * Pt, -H) : (radial * Pr * Ps, H)
            result .+= coefficient .* FewBodyECG.build_operator_matrix(basis, op)
        end
    end
    return result
end

function channel_solution(H, S, orbitals)
    D = Diagonal(1 ./ sqrt.(diag(S)))
    cond(D * S * D) < 1.0e12 || throw(ArgumentError("radial basis is too ill-conditioned; use fewer or more separated widths"))
    energies, vectors = FewBodyECG.solve_generalized_eigenproblem(D * H * D, D * S * D)
    coefficients = D * vectors[:, 1]
    return (;
        energy = first(energies), coefficients,
        wavefunction = Wavefunction(BasisSet(orbitals), coefficients),
    )
end

# The 3D Jacobi-to-physical Jacobian is |w|⁻³, so u(r)=√(4π)rψ(r) has unit
# radial norm in physical fm without renormalizing a truncated plotting range.
function s_wave_radial_amplitude(ψ, radii, w)
    scale = abs(only(w))
    u = [sqrt(4π) * r * ψ([r / scale]) / scale^(3 / 2) for r in radii]
    return u / sign(u[argmax(abs.(u))])   # fix the arbitrary eigenvector phase
end
