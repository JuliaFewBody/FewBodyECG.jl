using Test
using LinearAlgebra
using FewBodyECG
import FewBodyECG: _jacobi_transform, _encode_basis, _decode_basis, _chol_to_params, _params_to_matrix

# ---------------------------------------------------------------------------
# Shared 2-body (hydrogen) and 3-body (H⁻) operator fixtures
# ---------------------------------------------------------------------------

function _hydrogen_ops()
    masses = [1.0e15, 1.0]
    _, U = _jacobi_transform(masses)
    w = U' * [1.0, -1.0]
    ops = Operator[KineticOperator(Λ(masses)); CoulombOperator(-1.0, w)]
    return ops
end

function _hminus_ops()
    masses = [1.0e15, 1.0, 1.0]
    _, U = _jacobi_transform(masses)
    w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
    w_raw = [U' * Float64.(w) for w in w_list]
    coeffs = [-1.0, -1.0, +1.0]
    ops = Operator[KineticOperator(Λ(masses));
                   [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]...]
    return ops
end

# ---------------------------------------------------------------------------
# Cholesky parameterisation helpers
# ---------------------------------------------------------------------------

@testset "_chol_to_params / _params_to_matrix round-trip" begin
    for n in [1, 2, 3]
        L = LowerTriangular(tril(rand(n, n)) + 2I)   # positive diagonal
        params = _chol_to_params(Matrix(L))
        A_reconstructed = _params_to_matrix(params, n)
        A_original = Symmetric(L * L')
        @test A_reconstructed ≈ A_original rtol = 1.0e-10
    end
end

@testset "_encode_basis / _decode_basis round-trip" begin
    # 2-body (n_dim=1): 1×1 A matrices
    g1 = Rank0Gaussian([2.0;;], [0.0])
    g2 = Rank0Gaussian([5.0;;], [0.0])
    basis = BasisSet([g1, g2])

    θ = _encode_basis(basis)
    @test length(θ) == 2   # n_chol(1) = 1 per function, 2 functions

    basis2 = _decode_basis(θ, 2, 1)
    @test length(basis2.functions) == 2
    for (orig, recon) in zip(basis.functions, basis2.functions)
        @test Matrix(orig.A) ≈ Matrix(recon.A) rtol = 1.0e-10
    end
end

@testset "_encode_basis / _decode_basis round-trip (2D)" begin
    # 3-body (n_dim=2): 2×2 A matrices
    A = [3.0 0.5; 0.5 2.0]
    g = Rank0Gaussian(A, [0.0, 0.0])
    basis = BasisSet([g])

    θ = _encode_basis(basis)
    @test length(θ) == 3   # n_chol(2) = 3

    basis2 = _decode_basis(θ, 1, 2)
    @test Matrix(basis2.functions[1].A) ≈ A rtol = 1.0e-8
end

# ---------------------------------------------------------------------------
# solve_ECG_variational — argument validation
# ---------------------------------------------------------------------------

@testset "solve_ECG_variational argument validation" begin
    ops = _hydrogen_ops()

    @testset "Unknown loss_type throws" begin
        @test_throws ArgumentError solve_ECG_variational(
            ops, 3; loss_type = :bad, verbose = false
        )
    end

    @testset "Mismatched initial_basis size throws" begin
        sr = solve_ECG(ops, 5; verbose = false, scale = 1.0)
        basis5 = BasisSet(Rank0Gaussian[sr.basis_functions...])
        @test_throws ArgumentError solve_ECG_variational(
            ops, 3; initial_basis = basis5, verbose = false
        )
    end
end

# ---------------------------------------------------------------------------
# solve_ECG_variational — returned SolverResults structure
# ---------------------------------------------------------------------------

@testset "solve_ECG_variational returns valid SolverResults" begin
    ops = _hydrogen_ops()
    sr = solve_ECG_variational(ops, 5;
        scale = 1.0, max_iterations = 20, verbose = false
    )

    @test sr isa SolverResults
    @test sr.n_basis == 5
    @test length(sr.basis_functions) == 5
    @test isfinite(sr.ground_state)
    @test sr.ground_state < 0.0          # bound state
    @test sr.method === :variational
    @test length(sr.energies) == 1
    @test sr.energies[1] == sr.ground_state
    @test length(sr.eigenvectors) == 1
    @test size(sr.eigenvectors[1]) == (5, 5)
end

# ---------------------------------------------------------------------------
# solve_ECG_variational — both loss types run without error
# ---------------------------------------------------------------------------

@testset "solve_ECG_variational loss_type = :energy" begin
    ops = _hydrogen_ops()
    sr = solve_ECG_variational(ops, 5;
        loss_type = :energy, scale = 1.0, max_iterations = 20, verbose = false
    )
    @test isfinite(sr.ground_state)
    @test sr.ground_state < 0.0
end

@testset "solve_ECG_variational loss_type = :trace (warm start)" begin
    ops = _hydrogen_ops()
    # Warm-start from stochastic so the initial trace is already negative
    sr_s = solve_ECG(ops, 5; scale = 1.0, verbose = false)
    basis0 = BasisSet(Rank0Gaussian[sr_s.basis_functions...])
    sr = solve_ECG_variational(ops, 5;
        loss_type = :trace, initial_basis = basis0,
        max_iterations = 20, verbose = false
    )
    @test isfinite(sr.ground_state)
    @test sr.ground_state < 0.0
end

# ---------------------------------------------------------------------------
# solve_ECG_variational — variational principle
# ---------------------------------------------------------------------------

@testset "solve_ECG_variational respects variational bound (hydrogen)" begin
    ops = _hydrogen_ops()
    E_exact = -0.5   # hydrogen 1s

    sr = solve_ECG_variational(ops, 10;
        scale = 1.0, max_iterations = 100, verbose = false
    )

    # Variational principle: E₀ ≥ E_exact
    @test sr.ground_state >= E_exact - 1.0e-6
    # With 10 functions, should get within 0.01 Ha of exact
    @test sr.ground_state < E_exact + 0.01
end

@testset "solve_ECG_variational beats stochastic for hydrogen (same n)" begin
    ops = _hydrogen_ops()

    sr_stoch = solve_ECG(ops, 8; scale = 1.0, verbose = false)
    sr_var = solve_ECG_variational(ops, 8;
        scale = 1.0, max_iterations = 150, verbose = false
    )

    # Optimised basis should be at least as good as the stochastic one
    @test sr_var.ground_state <= sr_stoch.ground_state + 1.0e-6
end

# ---------------------------------------------------------------------------
# solve_ECG_variational — warm-start from stochastic result
# ---------------------------------------------------------------------------

@testset "solve_ECG_variational warm-start improves stochastic result" begin
    ops = _hminus_ops()

    sr_s = solve_ECG(ops, 8; scale = 1.0, verbose = false)
    basis0 = BasisSet(Rank0Gaussian[sr_s.basis_functions...])

    sr_v = solve_ECG_variational(ops, 8;
        initial_basis = basis0, max_iterations = 100, verbose = false
    )

    # Variational principle holds
    @test sr_v.ground_state >= -0.528 - 1.0e-4
    # Should not be worse than the starting point
    @test sr_v.ground_state <= sr_s.ground_state + 1.0e-6
end

# ---------------------------------------------------------------------------
# Compatibility with downstream utilities
# ---------------------------------------------------------------------------

@testset "ψ₀ works with variational SolverResults" begin
    ops = _hydrogen_ops()
    sr = solve_ECG_variational(ops, 5;
        scale = 1.0, max_iterations = 20, verbose = false
    )

    r_vec = [0.5]   # some point in Jacobi space
    psi = ψ₀(r_vec, sr; state = 1)
    @test isfinite(psi)
end

@testset "correlation_function works with variational SolverResults" begin
    ops = _hminus_ops()
    sr = solve_ECG_variational(ops, 5;
        scale = 1.0, max_iterations = 20, verbose = false
    )

    r_grid, rho = correlation_function(sr; npoints = 50)
    @test length(r_grid) == 50
    @test length(rho) == 50
    @test all(isfinite, rho)
    @test all(rho .>= 0.0)
end
