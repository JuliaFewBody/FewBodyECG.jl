using Test
using LinearAlgebra
using FewBodyECG
import FewBodyECG: _jacobi_transform, _encode_basis, _decode_basis, _chol_to_params, _params_to_matrix, convergence_history, solve_ECG_sequential

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
    # 2-body (n_dim=1): 1×1 A matrices, 1-D shift vectors
    # n_per = n_chol(1) + n_dim(1) = 2 params per function
    g1 = Rank0Gaussian([2.0;;], [0.0])
    g2 = Rank0Gaussian([5.0;;], [0.0])
    basis = BasisSet([g1, g2])

    θ = _encode_basis(basis)
    @test length(θ) == 4   # 2 functions × (n_chol=1 + n_dim=1)

    basis2 = _decode_basis(θ, 2, 1)
    @test length(basis2.functions) == 2
    for (orig, recon) in zip(basis.functions, basis2.functions)
        @test Matrix(orig.A) ≈ Matrix(recon.A) rtol = 1.0e-10
        @test recon.s ≈ orig.s rtol = 1.0e-10
    end
end

@testset "_encode_basis / _decode_basis round-trip (2D)" begin
    # 3-body (n_dim=2): 2×2 A matrices, 2-D shift vectors
    # n_per = n_chol(2) + n_dim(2) = 5 params per function
    A = [3.0 0.5; 0.5 2.0]
    g = Rank0Gaussian(A, [0.0, 0.0])
    basis = BasisSet([g])

    θ = _encode_basis(basis)
    @test length(θ) == 5   # 1 function × (n_chol=3 + n_dim=2)

    basis2 = _decode_basis(θ, 1, 2)
    @test Matrix(basis2.functions[1].A) ≈ A rtol = 1.0e-8
    @test basis2.functions[1].s ≈ g.s rtol = 1.0e-10
end

@testset "_encode_basis / _decode_basis round-trip with non-zero shifts" begin
    # Verify shift vectors are correctly preserved through the encode/decode cycle.
    A = [3.0 0.5; 0.5 2.0]
    s = [0.3, -0.1]
    g = Rank0Gaussian(A, s)
    basis = BasisSet([g])

    θ = _encode_basis(basis)
    @test length(θ) == 5

    basis2 = _decode_basis(θ, 1, 2)
    @test Matrix(basis2.functions[1].A) ≈ A rtol = 1.0e-8
    @test basis2.functions[1].s ≈ s rtol = 1.0e-10
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
    # fg_history records cumulative-minimum energies from primal evaluations.
    @test !isempty(sr.fg_history)
    @test last(sr.fg_history) <= sr.ground_state + 1.0e-8
    @test issorted(sr.fg_history; rev = true)   # monotone non-increasing
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

# ---------------------------------------------------------------------------
# convergence_history
# ---------------------------------------------------------------------------

@testset "convergence_history returns correct axes" begin
    ops = _hydrogen_ops()
    sr = solve_ECG_variational(ops, 5;
        scale = 1.0, max_iterations = 30, verbose = false
    )

    xs, ys = convergence_history(sr)
    @test length(xs) == length(ys)
    @test length(xs) == length(sr.fg_history)
    @test xs == 1:length(sr.fg_history)
    @test ys === sr.fg_history
    @test issorted(ys; rev = true)   # monotone non-increasing by construction
end

# ---------------------------------------------------------------------------
# Shift vectors are optimised (not pinned to zero)
# ---------------------------------------------------------------------------

@testset "shift vectors are included in optimised parameters" begin
    # _encode_basis should pack n_chol + n_dim params per Gaussian.
    # For a 1-D (hydrogen) basis: n_chol=1, n_dim=1 → 2 params per function.
    # The second param is the shift; _decode_basis should round-trip it.
    ops = _hydrogen_ops()
    sr = solve_ECG_variational(ops, 4;
        scale = 1.0, max_iterations = 50, verbose = false
    )
    # Each basis function has a 1-D shift vector stored in s.
    for g in sr.basis_functions
        @test length(g.s) == 1
        @test isfinite(g.s[1])
    end
end

# ---------------------------------------------------------------------------
# solve_ECG_sequential tests
# ---------------------------------------------------------------------------

@testset "solve_ECG_sequential argument validation" begin
    ops = _hydrogen_ops()
    @test_throws ArgumentError solve_ECG_sequential(
        ops, 3; loss_type = :bad, verbose = false
    )
end

@testset "solve_ECG_sequential returns valid SolverResults" begin
    ops = _hydrogen_ops()
    sr = solve_ECG_sequential(ops, 4;
        n_candidates = 3, scale = 1.0, max_iterations_step = 10, verbose = false
    )

    @test sr isa SolverResults
    @test sr.n_basis == 4
    @test length(sr.basis_functions) == 4
    @test isfinite(sr.ground_state)
    @test sr.ground_state < 0.0
    @test sr.method === :sequential
    # energies has one entry per sequential step
    @test length(sr.energies) == 4
    # eigenvectors: one final matrix
    @test length(sr.eigenvectors) == 1
    @test size(sr.eigenvectors[1]) == (4, 4)
    @test !isempty(sr.fg_history)
end

@testset "solve_ECG_sequential convergence is monotone" begin
    # By the variational principle, adding a linearly independent function
    # and then re-optimising cannot raise the ground-state energy.
    ops = _hydrogen_ops()
    sr = solve_ECG_sequential(ops, 6;
        n_candidates = 3, scale = 1.0, max_iterations_step = 20, verbose = false
    )
    for i in 2:length(sr.energies)
        @test sr.energies[i] <= sr.energies[i - 1] + 1.0e-8
    end
end

@testset "solve_ECG_sequential respects variational bound (hydrogen)" begin
    ops = _hydrogen_ops()
    E_exact = -0.5   # hydrogen 1s ground state

    sr = solve_ECG_sequential(ops, 6;
        n_candidates = 5, scale = 1.0, max_iterations_step = 50, verbose = false
    )

    @test sr.ground_state >= E_exact - 1.0e-6   # cannot go below exact
    @test sr.ground_state < E_exact + 0.01       # should be close with 6 functions
end

@testset "solve_ECG_sequential convergence_history is monotone" begin
    ops = _hydrogen_ops()
    sr = solve_ECG_sequential(ops, 4;
        n_candidates = 3, scale = 1.0, max_iterations_step = 15, verbose = false
    )
    xs, ys = convergence_history(sr)
    @test length(xs) == length(ys)
    @test issorted(ys; rev = true)
end

@testset "solve_ECG_sequential beats stochastic (hydrogen, same n)" begin
    ops = _hydrogen_ops()

    sr_stoch = solve_ECG(ops, 6; scale = 1.0, verbose = false)
    sr_seq   = solve_ECG_sequential(ops, 6;
        n_candidates = 5, scale = 1.0, max_iterations_step = 50, verbose = false
    )

    @test sr_seq.ground_state <= sr_stoch.ground_state + 1.0e-4
end

@testset "ψ₀ and correlation_function work with sequential SolverResults" begin
    ops = _hminus_ops()
    sr = solve_ECG_sequential(ops, 4;
        n_candidates = 3, scale = 1.0, max_iterations_step = 10, verbose = false
    )

    r_vec = [0.5, 0.3]
    psi = ψ₀(r_vec, sr; state = 1)
    @test isfinite(psi)

    r_grid, rho = correlation_function(sr; npoints = 30)
    @test length(r_grid) == 30
    @test all(isfinite, rho)
    @test all(rho .>= 0.0)
end
