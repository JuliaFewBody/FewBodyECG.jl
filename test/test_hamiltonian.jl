using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
import FewBodyECG: _compute_overlap_element, _build_operator_matrix, _compute_matrix_element, normalized_overlap, is_linearly_independent
using QuasiMonteCarlo

@testset "build_overlap_matrix" begin

    @testset "Size and symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.5 0.0; 0.0 1.5], [0.0, 0.0])
        g3 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.1, 0.1])

        basis = BasisSet([g1, g2, g3])
        S = build_overlap_matrix(basis)

        @test size(S) == (3, 3)
        @test issymmetric(S)
        @test all(isfinite, S)
    end


    @testset "Consistency" begin
        g1 = Rank0Gaussian([1.0;;], [0.0])
        g2 = Rank0Gaussian([2.0;;], [0.0])

        basis = BasisSet([g1, g2])
        S = build_overlap_matrix(basis)

        S11 = _compute_matrix_element(g1, g1)
        S12 = _compute_matrix_element(g1, g2)
        S22 = _compute_matrix_element(g2, g2)

        @test S[1, 1] ≈ S11 rtol = 1.0e-12
        @test S[1, 2] ≈ S12 rtol = 1.0e-12
        @test S[2, 2] ≈ S22 rtol = 1.0e-12
    end
end

@testset "build_hamiltonian_matrix" begin

    @testset "Size and symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.0, 0.0])
        basis = BasisSet([g1, g2])

        K = KineticOperator([0.5 0.0; 0.0 0.5])
        H = build_hamiltonian_matrix(basis, [K])

        @test size(H) == (2, 2)
        @test issymmetric(H)
        @test all(isfinite, H)
    end

    @testset "Additivity" begin
        # H(K+V) should equal H(K) + H(V)
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])

        K = KineticOperator([0.5 0.0; 0.0 0.5])
        V = CoulombOperator(-1.0, [1.0, 0.0])

        H_K = build_hamiltonian_matrix(basis, [K])
        H_V = build_hamiltonian_matrix(basis, [V])
        H_both = build_hamiltonian_matrix(basis, [K, V])

        @test H_both ≈ H_K + H_V rtol = 1.0e-12
    end

    @testset "Empty operators" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])

        H = build_hamiltonian_matrix(basis, FewBodyHamiltonians.Operator[])

        @test H == zeros(1, 1)
    end
end


@testset "solve_generalized_eigenproblem" begin

    @testset "Basic solve" begin
        H = [2.0 0.5; 0.5 3.0]
        S = [1.0 0.1; 0.1 1.0]

        evals, evecs = solve_generalized_eigenproblem(H, S)

        @test length(evals) == 2
        @test size(evecs) == (2, 2)
        @test all(isfinite, evals)
        @test all(isfinite, evecs)
    end

    @testset "Eigenvalue equation H*v = λ*S*v" begin
        H = [3.0 0.5; 0.5 2.0]
        S = [1.0 0.2; 0.2 1.0]

        evals, evecs = solve_generalized_eigenproblem(H, S)

        # Check each eigenpair
        for i in 1:2
            λ = evals[i]
            v = evecs[:, i]
            residual = H * v - λ * S * v
            @test norm(residual) < 1.0e-9
        end
    end

    @testset "Requires positive definite S" begin
        H = [1.0 0.0; 0.0 1.0]
        S_bad = [-1.0 0.0; 0.0 1.0]

        @test_throws Exception solve_generalized_eigenproblem(H, S_bad)
    end
end


@testset "normalized_overlap" begin

    @testset "Symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.5, 0.5])

        overlap_12 = normalized_overlap(g1, g2)
        overlap_21 = normalized_overlap(g2, g1)

        @test overlap_12 ≈ overlap_21 rtol = 1.0e-12
    end

    @testset "Range [0, 1]" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.5, 0.5])
        g3 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [5.0, 5.0])

        overlap_12 = normalized_overlap(g1, g2)
        overlap_13 = normalized_overlap(g1, g3)

        @test 0.0 <= overlap_12 <= 1.0
        @test 0.0 <= overlap_13 <= 1.0
    end

    @testset "Identical Gaussians" begin
        A = [1.5 0.0; 0.0 1.5]
        s = [0.3, 0.4]
        g1 = Rank0Gaussian(A, s)
        g2 = Rank0Gaussian(A, s)

        overlap = normalized_overlap(g1, g2)
        @test overlap ≈ 1.0 rtol = 1.0e-10
    end
end

@testset "is_linearly_independent" begin

    @testset "Empty basis (always independent)" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        empty_basis = BasisSet(Rank0Gaussian[])

        @test is_linearly_independent(g, empty_basis; threshold = 0.95)
    end


    @testset "Well-separated (should accept)" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [5.0, 5.0])

        basis = BasisSet([g1])

        @test is_linearly_independent(g2, basis; threshold = 0.95)
    end

    @testset "Threshold behavior" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.3, 0.3])

        basis = BasisSet([g1])
        overlap = normalized_overlap(g2, g1)

        @test is_linearly_independent(g2, basis; threshold = overlap + 0.01)
    end

    @testset "Invalid threshold" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])

        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = 0.0)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = 1.0)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = -0.5)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = 1.5)
    end
end

using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
import FewBodyECG: _compute_overlap_element, _build_operator_matrix, _compute_matrix_element
import FewBodyECG: normalized_overlap, is_linearly_independent, default_scale
import FewBodyECG: _jacobi_transform, _generate_A_matrix, generate_bij, generate_shift
using QuasiMonteCarlo

# =============================================================================
# Existing tests (preserved)
# =============================================================================

@testset "build_overlap_matrix" begin

    @testset "Size and symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.5 0.0; 0.0 1.5], [0.0, 0.0])
        g3 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.1, 0.1])

        basis = BasisSet([g1, g2, g3])
        S = build_overlap_matrix(basis)

        @test size(S) == (3, 3)
        @test issymmetric(S)
        @test all(isfinite, S)
    end


    @testset "Consistency" begin
        g1 = Rank0Gaussian([1.0;;], [0.0])
        g2 = Rank0Gaussian([2.0;;], [0.0])

        basis = BasisSet([g1, g2])
        S = build_overlap_matrix(basis)

        S11 = _compute_matrix_element(g1, g1)
        S12 = _compute_matrix_element(g1, g2)
        S22 = _compute_matrix_element(g2, g2)

        @test S[1, 1] ≈ S11 rtol = 1.0e-12
        @test S[1, 2] ≈ S12 rtol = 1.0e-12
        @test S[2, 2] ≈ S22 rtol = 1.0e-12
    end
end

@testset "build_hamiltonian_matrix" begin

    @testset "Size and symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.0, 0.0])
        basis = BasisSet([g1, g2])

        K = KineticOperator([0.5 0.0; 0.0 0.5])
        H = build_hamiltonian_matrix(basis, [K])

        @test size(H) == (2, 2)
        @test issymmetric(H)
        @test all(isfinite, H)
    end

    @testset "Additivity" begin
        # H(K+V) should equal H(K) + H(V)
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])

        K = KineticOperator([0.5 0.0; 0.0 0.5])
        V = CoulombOperator(-1.0, [1.0, 0.0])

        H_K = build_hamiltonian_matrix(basis, [K])
        H_V = build_hamiltonian_matrix(basis, [V])
        H_both = build_hamiltonian_matrix(basis, [K, V])

        @test H_both ≈ H_K + H_V rtol = 1.0e-12
    end

    @testset "Empty operators" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])

        H = build_hamiltonian_matrix(basis, FewBodyHamiltonians.Operator[])

        @test H == zeros(1, 1)
    end
end


@testset "solve_generalized_eigenproblem" begin

    @testset "Basic solve" begin
        H = [2.0 0.5; 0.5 3.0]
        S = [1.0 0.1; 0.1 1.0]

        evals, evecs = solve_generalized_eigenproblem(H, S)

        @test length(evals) == 2
        @test size(evecs) == (2, 2)
        @test all(isfinite, evals)
        @test all(isfinite, evecs)
    end

    @testset "Eigenvalue equation H*v = λ*S*v" begin
        H = [3.0 0.5; 0.5 2.0]
        S = [1.0 0.2; 0.2 1.0]

        evals, evecs = solve_generalized_eigenproblem(H, S)

        # Check each eigenpair
        for i in 1:2
            λ = evals[i]
            v = evecs[:, i]
            residual = H * v - λ * S * v
            @test norm(residual) < 1.0e-9
        end
    end

    @testset "Requires positive definite S" begin
        H = [1.0 0.0; 0.0 1.0]
        S_bad = [-1.0 0.0; 0.0 1.0]

        @test_throws Exception solve_generalized_eigenproblem(H, S_bad)
    end
end


@testset "normalized_overlap" begin

    @testset "Symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.5, 0.5])

        overlap_12 = normalized_overlap(g1, g2)
        overlap_21 = normalized_overlap(g2, g1)

        @test overlap_12 ≈ overlap_21 rtol = 1.0e-12
    end

    @testset "Range [0, 1]" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.5, 0.5])
        g3 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [5.0, 5.0])

        overlap_12 = normalized_overlap(g1, g2)
        overlap_13 = normalized_overlap(g1, g3)

        @test 0.0 <= overlap_12 <= 1.0
        @test 0.0 <= overlap_13 <= 1.0
    end

    @testset "Identical Gaussians" begin
        A = [1.5 0.0; 0.0 1.5]
        s = [0.3, 0.4]
        g1 = Rank0Gaussian(A, s)
        g2 = Rank0Gaussian(A, s)

        overlap = normalized_overlap(g1, g2)
        @test overlap ≈ 1.0 rtol = 1.0e-10
    end
end

@testset "is_linearly_independent" begin

    @testset "Empty basis (always independent)" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        empty_basis = BasisSet(Rank0Gaussian[])

        @test is_linearly_independent(g, empty_basis; threshold = 0.95)
    end


    @testset "Well-separated (should accept)" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [5.0, 5.0])

        basis = BasisSet([g1])

        @test is_linearly_independent(g2, basis; threshold = 0.95)
    end

    @testset "Threshold behavior" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.3, 0.3])

        basis = BasisSet([g1])
        overlap = normalized_overlap(g2, g1)

        @test is_linearly_independent(g2, basis; threshold = overlap + 0.01)
    end

    @testset "Invalid threshold" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])

        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = 0.0)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = 1.0)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = -0.5)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold = 1.5)
    end
end

# =============================================================================
# New tests for default_scale
# =============================================================================

@testset "default_scale" begin

    @testset "Basic functionality" begin
        # Equal masses
        masses_equal = [1.0, 1.0, 1.0]
        scale = default_scale(masses_equal)
        @test scale ≈ 1.0 rtol = 1.0e-10
        @test scale > 0
    end

    @testset "Muonic systems (heavy nuclei)" begin
        # Muon mass ≈ 206.77 mₑ, so scale ~ 1/√207 ≈ 0.07
        masses_muonic = [5000.0, 5000.0, 206.77]
        scale = default_scale(masses_muonic)
        @test scale ≈ 1 / sqrt(206.77) rtol = 1.0e-6
        @test scale < 0.1  # Should be small for muonic systems
    end

    @testset "Ignores infinite masses" begin
        # Infinite mass (fixed nucleus) should be ignored
        masses_fixed = [1.0e15, 1.0, 1.0]
        scale = default_scale(masses_fixed)
        @test scale ≈ 1.0 rtol = 1.0e-10  # Should use minimum of finite masses
    end

    @testset "Single light particle" begin
        masses = [1000.0, 1000.0, 1.0]
        scale = default_scale(masses)
        @test scale ≈ 1.0 rtol = 1.0e-10
    end

    @testset "Hydrogen-like (electron mass = 1)" begin
        masses_hydrogen = [1836.15, 1.0]  # proton, electron
        scale = default_scale(masses_hydrogen)
        @test scale ≈ 1.0 rtol = 1.0e-10
    end
end

# =============================================================================
# Tests for generate_bij
# =============================================================================

@testset "generate_bij" begin

    @testset "Output range with bmin/bmax" begin
        b1 = 1.0
        bmin = 0.02 * b1
        bmax = 5.0 * b1
        
        for i in 1:100
            bij = generate_bij(:quasirandom, i, 3, b1; qmc_sampler = HaltonSample())
            @test all(bij .>= bmin - 1e-10)
            @test all(bij .<= bmax + 1e-10)
        end
    end

    @testset "No zero values (would cause singularity)" begin
        b1 = 1.0
        for i in 1:100
            bij = generate_bij(:quasirandom, i, 3, b1; qmc_sampler = HaltonSample())
            @test all(bij .> 0)  # Critical: b=0 would give A → ∞
        end
    end

    @testset "Reproducibility of quasirandom" begin
        b1 = 2.0
        bij1 = generate_bij(:quasirandom, 42, 3, b1; qmc_sampler = SobolSample())
        bij2 = generate_bij(:quasirandom, 42, 3, b1; qmc_sampler = SobolSample())
        @test bij1 ≈ bij2
    end

    @testset "Different indices give different values" begin
        b1 = 1.0
        bij1 = generate_bij(:quasirandom, 1, 3, b1; qmc_sampler = HaltonSample())
        bij2 = generate_bij(:quasirandom, 2, 3, b1; qmc_sampler = HaltonSample())
        @test bij1 != bij2
    end
end

@testset "generate_shift" begin

    @testset "Output range" begin
        scale = 2.0
        for i in 1:100
            s = generate_shift(:quasirandom, i, 3, scale; qmc_sampler = HaltonSample())
            @test all(abs.(s) .<= scale + 1e-10)
        end
    end

    @testset "Correct dimension" begin
        for dim in [1, 2, 3, 5]
            s = generate_shift(:quasirandom, 1, dim, 1.0; qmc_sampler = HaltonSample())
            @test length(s) == dim
        end
    end
end


@testset "_generate_A_matrix" begin

    @testset "Symmetry" begin
        w_list = [[1.0, -1.0], [1.0, 0.0]]
        bij = [1.0, 2.0]
        A = _generate_A_matrix(bij, w_list)
        @test issymmetric(A)
    end


    @testset "Correct size" begin
        w_list = [[1.0, -1.0], [1.0, 0.0]]
        bij = [1.0, 2.0]
        A = _generate_A_matrix(bij, w_list)
        @test size(A) == (2, 2)
    end

    @testset "Small bij gives large A elements" begin
        w_list = [[1.0, -1.0]]
        A_large_b = _generate_A_matrix([10.0], w_list)
        A_small_b = _generate_A_matrix([0.1], w_list)
        @test maximum(abs.(A_small_b)) > maximum(abs.(A_large_b))
    end
end


@testset "Physics: Variational principle" begin
    # Energy should decrease monotonically as basis size increases
    
    masses = [1.0e15, 1.0]
    Λmat = Λ(masses)
    kin = KineticOperator(Λmat)
    J, U = _jacobi_transform(masses)
    w_raw = [U' * [1.0, -1.0]]
    coulomb = CoulombOperator(-1.0, w_raw[1])
    ops = Operator[kin, coulomb]
    
    result = solve_ECG(ops, 20; scale = 1.5, verbose = false)
    
    # Check monotonic decrease (with some tolerance for numerical noise)
    for i in 2:length(result.energies)
        @test result.energies[i] <= result.energies[i-1] + 1e-10
    end
end

@testset "Physics: Scale sensitivity" begin
    # Demonstrate that scale matters for convergence
    
    masses = [1.0e15, 1.0]
    Λmat = Λ(masses)
    kin = KineticOperator(Λmat)
    J, U = _jacobi_transform(masses)
    w_raw = [U' * [1.0, -1.0]]
    coulomb = CoulombOperator(-1.0, w_raw[1])
    ops = Operator[kin, coulomb]
    
    # Good scale for hydrogen
    result_good = solve_ECG(ops, 15; scale = 1.5, verbose = false)
    
    # Bad scale (too small - Gaussians too narrow)
    result_bad = solve_ECG(ops, 15; scale = 0.05, verbose = false)
    
    # Good scale should give better (lower) energy
    @test result_good.ground_state < result_bad.ground_state
end


@testset "solve_ECG" begin

    @testset "Returns correct structure" begin
        masses = [1.0e15, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = _jacobi_transform(masses)
        w_raw = [U' * [1.0, -1.0]]
        coulomb = CoulombOperator(-1.0, w_raw[1])
        ops = Operator[kin, coulomb]
        
        result = solve_ECG(ops, 10; scale = 1.0, verbose = false)
        
        @test length(result.basis_functions) == result.n_basis
        @test length(result.energies) == result.n_basis
        @test result.ground_state == last(result.energies)
        @test result.ground_state == minimum(result.energies)
    end

    @testset "Respects max_attempts" begin
        masses = [1.0e15, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = _jacobi_transform(masses)
        w_raw = [U' * [1.0, -1.0]]
        coulomb = CoulombOperator(-1.0, w_raw[1])
        ops = Operator[kin, coulomb]
        
        # Request many basis functions but limit attempts
        result = solve_ECG(ops, 1000; scale = 1.0, max_attempts = 50, verbose = false)
        
        @test result.n_basis <= 50
    end

    @testset "Handles linear dependence rejection" begin
        masses = [1.0e15, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = _jacobi_transform(masses)
        w_raw = [U' * [1.0, -1.0]]
        coulomb = CoulombOperator(-1.0, w_raw[1])
        ops = Operator[kin, coulomb]
        
        # Very strict threshold should cause rejections
        result = solve_ECG(ops, 10; scale = 1.0, threshold = 0.5, verbose = false)
        
        # Should still produce valid results
        @test result.n_basis >= 1
        @test isfinite(result.ground_state)
    end
end
