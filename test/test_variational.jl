using Test
using LinearAlgebra
using FewBodyECG
import FewBodyECG: jacobi_transform, _encode_basis, _decode_basis, _chol_to_params, _params_to_matrix

# ---------------------------------------------------------------------------
# Shared 2-body (hydrogen) and 3-body (H⁻) operator fixtures
# ---------------------------------------------------------------------------

function _hydrogen_ops()
    masses = [1.0e15, 1.0]
    _, U = jacobi_transform(masses)
    w = U' * [1.0, -1.0]
    ops = Operator[KineticOperator(Λ(masses)); CoulombOperator(-1.0, w)]
    return ops
end

function _hminus_ops()
    masses = [1.0e15, 1.0, 1.0]
    _, U = jacobi_transform(masses)
    w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
    w_raw = [U' * Float64.(w) for w in w_list]
    coeffs = [-1.0, -1.0, +1.0]
    ops = Operator[
        KineticOperator(Λ(masses));
        [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]...
    ]
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
    @test length(θ) == 8   # 2 functions × (n_chol=1 + 3·n_dim=3)

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
    @test length(θ) == 9   # 1 function × (n_chol=3 + 3·n_dim=6)

    basis2 = _decode_basis(θ, 1, 2)
    @test Matrix(basis2.functions[1].A) ≈ A rtol = 1.0e-8
    @test basis2.functions[1].s ≈ g.s rtol = 1.0e-10
end

@testset "_encode_basis / _decode_basis round-trip with non-zero shifts" begin
    # Verify shift vectors are correctly preserved through the encode/decode cycle.
    A = [3.0 0.5; 0.5 2.0]
    s = [0.3 -0.2 0.5; -0.1 0.4 -0.3]   # full N×3 shift
    g = Rank0Gaussian(A, s)
    basis = BasisSet([g])

    θ = _encode_basis(basis)
    @test length(θ) == 9

    basis2 = _decode_basis(θ, 1, 2)
    @test Matrix(basis2.functions[1].A) ≈ A rtol = 1.0e-8
    @test basis2.functions[1].s ≈ s rtol = 1.0e-10
end

# ---------------------------------------------------------------------------
# Variational — argument validation
# ---------------------------------------------------------------------------

@testset "Variational argument validation" begin
    ops = _hydrogen_ops()

    @testset "Mismatched init size throws" begin
        sol5 = solve(ops, SVM(basis = 5, candidates = 1, scale = 1.0))
        @test_throws ArgumentError solve(ops, Variational(basis = 3, scale = 1.0); init = sol5)
    end
end

# ---------------------------------------------------------------------------
# Variational — returned Solution structure
# ---------------------------------------------------------------------------

@testset "Variational returns a valid Solution" begin
    ops = _hydrogen_ops()
    sol = solve(ops, Variational(basis = 5, scale = 1.0, maxiter = 20))

    @test sol isa Solution
    @test length(sol.basis.functions) == 5
    @test isfinite(sol.E₀)
    @test sol.E₀ < 0.0          # bound state
    @test sol.stages[1].method isa Variational
    @test size(sol.coefficients) == (5, 5)
    # energies(sol) records the cumulative-min energies from primal
    # evaluations along the LBFGS trajectory (formerly `fg_history`).
    @test !isempty(energies(sol))
    @test last(energies(sol)) <= sol.E₀ + 1.0e-8
    @test issorted(energies(sol); rev = true)   # monotone non-increasing
end

@testset "Variational tracks the requested state" begin
    sol = solve(_hydrogen_ops(), Variational(basis = 5, scale = 1.0, maxiter = 20); state = 2)

    @test sol.state == 2
    @test last(energies(sol)) > sol.E[1] + 1.0e-3
end

@testset "Gradient solvers initialise from pairwise geometry" begin
    for term in (("Oscillator", 1, 2, 0.5), ("Gaussian", 1, 2, -1.0, 1.0))
        ops = Operators([1.0e15, 1.0])
        ops += "Kinetic"
        ops += term

        for alg in (
                Variational(basis = 4, scale = 1.0, maxiter = 10),
                GrowVariational(basis = 4, candidates = 2, scale = 1.0, maxiter_step = 10),
            )
            sol = solve(ops, alg)
            @test isfinite(sol.E₀)
        end
    end
end

# ---------------------------------------------------------------------------
# Variational — variational principle
# ---------------------------------------------------------------------------

@testset "Variational respects variational bound (hydrogen)" begin
    ops = _hydrogen_ops()
    E_exact = -0.5   # hydrogen 1s

    sol = solve(ops, Variational(basis = 10, scale = 1.0, maxiter = 100))

    # Variational principle: E₀ ≥ E_exact
    @test sol.E₀ >= E_exact - 1.0e-6
    # With 10 functions, should get within 0.01 Ha of exact
    @test sol.E₀ < E_exact + 0.01
end

@testset "Variational beats stochastic for hydrogen (same n)" begin
    ops = _hydrogen_ops()

    sol_stoch = solve(ops, SVM(basis = 8, candidates = 1, scale = 1.0))
    sol_var = solve(ops, Variational(basis = 8, scale = 1.0, maxiter = 150))

    # Optimised basis should be at least as good as the stochastic one
    @test sol_var.E₀ <= sol_stoch.E₀ + 1.0e-6
end

# ---------------------------------------------------------------------------
# Variational — warm-start from stochastic result
# ---------------------------------------------------------------------------

@testset "Variational warm-start improves stochastic result" begin
    ops = _hminus_ops()

    sol_s = solve(ops, SVM(basis = 8, candidates = 1, scale = 1.0))

    sol_v = solve(ops, Variational(basis = 8, scale = 1.0, maxiter = 100); init = sol_s)

    # Variational principle holds
    @test sol_v.E₀ >= -0.528 - 1.0e-4
    # Should not be worse than the starting point
    @test sol_v.E₀ <= sol_s.E₀ + 1.0e-6
end

# ---------------------------------------------------------------------------
# Compatibility with downstream utilities
# ---------------------------------------------------------------------------

@testset "wavefunction works with a Variational Solution" begin
    ops = _hydrogen_ops()
    sol = solve(ops, Variational(basis = 5, scale = 1.0, maxiter = 20))

    r_vec = [0.5]   # some point in Jacobi space
    psi = wavefunction(sol; state = 1)(r_vec)
    @test isfinite(psi)
end

# ---------------------------------------------------------------------------
# energies(sol) as the per-iteration convergence trace
# ---------------------------------------------------------------------------

@testset "energies(sol) returns correct axes (Variational)" begin
    ops = _hydrogen_ops()
    sol = solve(ops, Variational(basis = 5, scale = 1.0, maxiter = 30))

    xs, ys = (1:length(energies(sol)), energies(sol))
    @test length(xs) == length(ys)
    @test xs == 1:length(energies(sol))
    @test ys === energies(sol)
    @test issorted(ys; rev = true)   # monotone non-increasing by construction
end

# ---------------------------------------------------------------------------
# Shift vectors are optimised (not pinned to zero)
# ---------------------------------------------------------------------------

@testset "shift vectors are included in optimised parameters" begin
    # _encode_basis packs n_chol + 3·n_dim params per Gaussian (N×3 shift).
    # For a 1-D (hydrogen) basis: n_chol=1, 3·n_dim=3 → 4 params per function.
    # The shift is optimised; _decode_basis round-trips it.
    ops = _hydrogen_ops()
    sol = solve(ops, Variational(basis = 4, scale = 1.0, maxiter = 50))
    # Each basis function has a 1×3 shift supervector stored in s.
    for g in sol.basis.functions
        @test size(g.s) == (1, 3)
        @test all(isfinite, g.s)
    end
end

# ---------------------------------------------------------------------------
# GrowVariational tests
# ---------------------------------------------------------------------------

@testset "GrowVariational returns a valid Solution" begin
    ops = _hydrogen_ops()
    sol = solve(
        ops, GrowVariational(basis = 4, candidates = 3, scale = 1.0, maxiter_step = 10)
    )

    @test sol isa Solution
    @test length(sol.basis.functions) == 4
    @test isfinite(sol.E₀)
    @test sol.E₀ < 0.0
    @test sol.stages[1].method isa GrowVariational
    # energies(sol) has one entry per sequential growth step
    @test length(energies(sol)) == 4
    # coefficients: one final matrix
    @test size(sol.coefficients) == (4, 4)
end

@testset "GrowVariational tracks the requested state" begin
    sol = solve(
        _hydrogen_ops(), GrowVariational(basis = 5, candidates = 3, scale = 1.0, maxiter_step = 20);
        state = 2,
    )

    @test sol.state == 2
    @test last(energies(sol)) > sol.E[1] + 1.0e-3
end

@testset "GrowVariational convergence is monotone" begin
    # By the variational principle, adding a linearly independent function
    # and then re-optimising cannot raise the ground-state energy.
    ops = _hydrogen_ops()
    sol = solve(
        ops, GrowVariational(basis = 6, candidates = 3, scale = 1.0, maxiter_step = 20)
    )
    ener = energies(sol)
    for i in 2:length(ener)
        @test ener[i] <= ener[i - 1] + 1.0e-8
    end
end

@testset "GrowVariational respects variational bound (hydrogen)" begin
    ops = _hydrogen_ops()
    E_exact = -0.5   # hydrogen 1s ground state

    sol = solve(
        ops, GrowVariational(basis = 6, candidates = 5, scale = 1.0, maxiter_step = 50)
    )

    @test sol.E₀ >= E_exact - 1.0e-6   # cannot go below exact
    @test sol.E₀ < E_exact + 0.01       # should be close with 6 functions
end

@testset "energies(sol) is monotone (GrowVariational)" begin
    ops = _hydrogen_ops()
    sol = solve(
        ops, GrowVariational(basis = 4, candidates = 3, scale = 1.0, maxiter_step = 15)
    )
    xs, ys = (1:length(energies(sol)), energies(sol))
    @test length(xs) == length(ys)
    @test issorted(ys; rev = true)
end

@testset "GrowVariational beats stochastic (hydrogen, same n)" begin
    ops = _hydrogen_ops()

    sol_stoch = solve(ops, SVM(basis = 6, candidates = 1, scale = 1.0))
    sol_seq = solve(
        ops, GrowVariational(basis = 6, candidates = 5, scale = 1.0, maxiter_step = 50)
    )

    @test sol_seq.E₀ <= sol_stoch.E₀ + 1.0e-4
end

@testset "wavefunction works with a GrowVariational Solution" begin
    ops = _hminus_ops()
    sol = solve(
        ops, GrowVariational(basis = 4, candidates = 3, scale = 1.0, maxiter_step = 10)
    )

    r_vec = [0.5, 0.3]
    psi = wavefunction(sol; state = 1)(r_vec)
    @test isfinite(psi)
end
