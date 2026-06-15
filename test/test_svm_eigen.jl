using Test
using LinearAlgebra
using Random
using FewBodyECG

# Internal (unexported) symbols under test.
using FewBodyECG: SVMEigen, commit_candidate!, score_candidate,
    full_arrowhead_eigen, smallest_arrowhead_eigval, coefficients

# Build the generalised eigendecomposition of (H, S) incrementally, one column
# at a time, exactly as the SVM does.
function incremental_decomp(H::AbstractMatrix, S::AbstractMatrix)
    n = size(S, 1)
    eig = SVMEigen()
    for k in 0:(n - 1)
        s_col = S[1:k, k + 1]
        h_col = H[1:k, k + 1]
        commit_candidate!(eig, s_col, h_col, S[k + 1, k + 1], H[k + 1, k + 1])
    end
    return eig
end

@testset "SVM incremental arrowhead eigensolver" begin

    @testset "matches LAPACK on random SPD systems" begin
        rng = MersenneTwister(20260612)
        for n in (2, 5, 12, 30)
            M = randn(rng, n, n)
            S = M * M' + n * I                 # well-conditioned SPD overlap
            H = (M = randn(rng, n, n); Symmetric(M + M') |> Matrix)

            eig = incremental_decomp(H, S)
            λ_ref = sort(eigen(Symmetric(H), Symmetric(S)).values)

            # Ground state (what the SVM actually optimises) stays tight; the
            # naive arrowhead eigenvectors let interior levels drift to ~1e-6
            # over many incremental steps (Gu-Eisenstat stabilisation would fix
            # this, deferred until excited states need it).
            @test minimum(eig.ε) ≈ minimum(λ_ref) rtol = 1e-9
            @test sort(eig.ε) ≈ λ_ref rtol = 1e-8
            # S-orthonormality and H-diagonalisation of the coefficient matrix.
            c = coefficients(eig)
            @test c' * S * c ≈ I(n) atol = 1e-9
            @test c' * H * c ≈ Diagonal(eig.ε) atol = 1e-8
        end
    end

    @testset "scoring matches LAPACK ground state of the (k+1) submatrix" begin
        rng = MersenneTwister(7)
        n = 20
        M = randn(rng, n, n)
        S = M * M' + n * I
        H = (M = randn(rng, n, n); Symmetric(M + M') |> Matrix)

        eig = SVMEigen()
        for k in 0:(n - 1)
            s_col = S[1:k, k + 1]
            h_col = H[1:k, k + 1]
            E_score = score_candidate(eig, s_col, h_col, S[k + 1, k + 1], H[k + 1, k + 1])
            λ_ref = eigen(Symmetric(H[1:(k + 1), 1:(k + 1)]), Symmetric(S[1:(k + 1), 1:(k + 1)])).values
            @test E_score ≈ minimum(λ_ref) rtol = 1e-9
            commit_candidate!(eig, s_col, h_col, S[k + 1, k + 1], H[k + 1, k + 1])
        end
    end

    @testset "rejects linearly dependent candidates" begin
        # Two identical functions: the second has zero orthogonal residual.
        S = [1.0 1.0; 1.0 1.0]
        H = [-0.5 -0.5; -0.5 -0.5]
        eig = SVMEigen()
        commit_candidate!(eig, Float64[], Float64[], S[1, 1], H[1, 1])
        @test score_candidate(eig, [S[1, 2]], [H[1, 2]], S[2, 2], H[2, 2]) === nothing
        @test commit_candidate!(eig, [S[1, 2]], [H[1, 2]], S[2, 2], H[2, 2]) === nothing
    end

    @testset "full arrowhead eigen matches dense" begin
        rng = MersenneTwister(99)
        for k in (1, 3, 8)
            ε = sort(randn(rng, k))
            b = randn(rng, k)
            α = randn(rng)
            M = [Diagonal(ε) b; b' α]
            λ, V = full_arrowhead_eigen(ε, b, α)
            @test sort(λ) ≈ eigen(Symmetric(Matrix(M))).values rtol = 1e-9
            @test V' * V ≈ I(k + 1) atol = 1e-8
            @test V' * Symmetric(Matrix(M)) * V ≈ Diagonal(λ) atol = 1e-7
        end
    end

    @testset "competitive solver is self-consistent with LAPACK" begin
        # The corruption bug (incremental energy drifting below the true ground
        # state under competitive selection) is caught by requiring the reported
        # energy to equal LAPACK on the solver's own basis.
        ops = Operators([1e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"

        sr = solve_ECG_competitive(ops, 30; n_candidates = 25, scale = 1.0, verbose = false)
        basis = BasisSet(sr.basis_functions)
        H = build_hamiltonian_matrix(basis, ops)
        S = build_overlap_matrix(basis)
        # Compare against *unregularized* LAPACK: solve_generalized_eigenproblem
        # adds ε·I when cond(S) is large, which shifts its eigenvalue; the
        # whitened incremental solver tracks the true (unregularized) value even
        # at cond(S) ~ 1e15.
        λ_ref = eigen(Symmetric(H), Symmetric(S)).values

        @test sr.ground_state ≈ minimum(λ_ref) atol = 1e-5
        @test sr.ground_state > -0.5 - 1e-6          # variational: above exact H ground state
        # Energy history is non-increasing (competitive selection keeps the best).
        @test all(diff(sr.energies) .<= 1e-9)
        # Stored eigenvectors are S-normalised → ψ₀ usable.
        c = sr.eigenvectors[end][:, 1]
        @test c' * S * c ≈ 1.0 atol = 1e-6
        @test isfinite(ψ₀([0.5], sr))
    end

    @testset "matches LAPACK on the hydrogen ECG system" begin
        masses = [1.0e15, 1.0]
        Λmat = Λ(masses)
        _, U = _jacobi_transform(masses)
        w = U' * [1.0, -1.0]
        ops = Operator[KineticOperator(Λmat), CoulombOperator(-1.0, w)]

        sr = solve_ECG(ops, 25; scale = 1.0, verbose = false)
        basis = BasisSet(sr.basis_functions)
        H = build_hamiltonian_matrix(basis, ops)
        S = build_overlap_matrix(basis)

        eig = incremental_decomp(H, S)
        λ_ref, _ = solve_generalized_eigenproblem(H, S)

        @test minimum(eig.ε) ≈ minimum(λ_ref) rtol = 1e-8   # the real test: vs LAPACK
        @test minimum(eig.ε) ≈ -0.5 atol = 5e-2             # sanity: near hydrogen E₀
    end

end
