using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
import FewBodyECG: _jacobi_transform, _generate_A_matrix, generate_bij, generate_shift
import FewBodyECG: ψ₀, convergence, correlation_function, SolverResults
using QuasiMonteCarlo

function create_mock_solver_results(;
        n_basis::Int = 5,
        dim::Int = 2,
        scale::Float64 = 1.0
    )
    # Create simple basis functions
    basis_fns = GaussianBase[]
    for i in 1:n_basis
        A = Diagonal(fill(0.5 * i, dim))
        s = zeros(dim)
        push!(basis_fns, Rank0Gaussian(Matrix(A), s))
    end

    # Create mock operators
    K = KineticOperator(Diagonal(fill(0.5, dim)))
    V = CoulombOperator(-1.0, [1.0; zeros(dim - 1)])
    operators = Operator[K, V]

    # Create mock energies (decreasing sequence)
    energies = [-0.1 * i for i in 1:n_basis]

    # Create mock eigenvectors
    eigenvectors = [randn(i, i) for i in 1:n_basis]
    # Normalize columns
    for i in 1:n_basis
        for j in 1:i
            eigenvectors[i][:, j] ./= norm(eigenvectors[i][:, j])
        end
    end

    return SolverResults(
        basis_fns,
        n_basis,
        operators,
        :quasirandom,
        HaltonSample(),
        scale,
        energies[end],
        energies,
        eigenvectors
    )
end

@testset "SolverResults" begin

    @testset "Construction" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 2)

        @test sr.n_basis == 3
        @test length(sr.basis_functions) == 3
        @test length(sr.energies) == 3
        @test length(sr.eigenvectors) == 3
        @test sr.method == :quasirandom
        @test sr.length_scale == 1.0
    end

    @testset "Ground state is last energy" begin
        sr = create_mock_solver_results(n_basis = 5)
        @test sr.ground_state == sr.energies[end]
    end

    @testset "Operators stored correctly" begin
        sr = create_mock_solver_results()
        @test length(sr.operators) == 2
        @test sr.operators[1] isa KineticOperator
        @test sr.operators[2] isa CoulombOperator
    end

    @testset "Different samplers" begin
        basis_fns = [Rank0Gaussian([1.0;;], [0.0])]
        ops = Operator[KineticOperator([0.5;;])]

        sr_halton = SolverResults(basis_fns, 1, ops, :quasirandom, HaltonSample(), 1.0, -0.5, [-0.5], [ones(1, 1)])
        sr_sobol = SolverResults(basis_fns, 1, ops, :quasirandom, SobolSample(), 1.0, -0.5, [-0.5], [ones(1, 1)])

        @test sr_halton.sampler isa HaltonSample
        @test sr_sobol.sampler isa SobolSample
    end
end

@testset "ψ₀" begin

    @testset "Basic evaluation with coefficients" begin
        # Single Gaussian: ψ = c * exp(-r'Ar + s'r)
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        c = [1.0]

        r = [0.0, 0.0]
        ψ_val = ψ₀(r, c, basis_fns)

        # At origin with s=0: exp(-0 + 0) = 1
        @test ψ_val ≈ 1.0 rtol = 1.0e-10
    end

    @testset "Gaussian decay" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        c = [1.0]

        ψ_origin = ψ₀([0.0, 0.0], c, basis_fns)
        ψ_far = ψ₀([3.0, 3.0], c, basis_fns)

        # Should decay away from origin
        @test abs(ψ_far) < abs(ψ_origin)
        @test ψ_far ≈ exp(-18.0) rtol = 1.0e-10  # exp(-(3² + 3²))
    end

    @testset "Shift vector effect" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [2.0, 0.0]  # Shift in x-direction
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        c = [1.0]

        # Maximum should be shifted
        ψ_origin = ψ₀([0.0, 0.0], c, basis_fns)
        ψ_shifted = ψ₀([1.0, 0.0], c, basis_fns)  # Closer to maximum

        @test abs(ψ_shifted) > abs(ψ_origin)
    end

    @testset "Linear combination" begin
        A1 = [1.0 0.0; 0.0 1.0]
        A2 = [2.0 0.0; 0.0 2.0]
        s = [0.0, 0.0]
        g1 = Rank0Gaussian(A1, s)
        g2 = Rank0Gaussian(A2, s)
        basis_fns = [g1, g2]

        c = [0.5, 0.5]
        r = [0.0, 0.0]

        ψ_val = ψ₀(r, c, basis_fns)

        # At origin: 0.5 * 1 + 0.5 * 1 = 1
        @test ψ_val ≈ 1.0 rtol = 1.0e-10
    end

    @testset "Negative coefficients" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]

        c_pos = [1.0]
        c_neg = [-1.0]
        r = [0.0, 0.0]

        @test ψ₀(r, c_pos, basis_fns) ≈ -ψ₀(r, c_neg, basis_fns) rtol = 1.0e-10
    end

    @testset "With SolverResults" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 2)

        # Should not throw
        r = [0.5, 0.5]
        ψ_val = ψ₀(r, sr; state = 1)

        @test isfinite(ψ_val)
    end

    @testset "Different states" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)
        r = [0.1, 0.1]

        # Different states should generally give different values
        ψ_1 = ψ₀(r, sr; state = 1)
        ψ_2 = ψ₀(r, sr; state = 2)

        @test isfinite(ψ_1)
        @test isfinite(ψ_2)
        # They might be equal by chance, but usually won't be
    end

    @testset "1D case" begin
        A = [2.0;;]
        s = [0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        c = [1.0]

        r = [1.0]
        ψ_val = ψ₀(r, c, basis_fns)

        @test ψ_val ≈ exp(-2.0) rtol = 1.0e-10
    end
end

@testset "convergence" begin

    @testset "Returns correct range and energies" begin
        sr = create_mock_solver_results(n_basis = 10)

        indices, energies = convergence(sr)

        @test indices == 1:10
        @test energies == sr.energies
        @test length(indices) == length(energies)
    end

    @testset "Single basis function" begin
        sr = create_mock_solver_results(n_basis = 1)

        indices, energies = convergence(sr)

        @test indices == 1:1
        @test length(energies) == 1
    end

    @testset "Energies are same object" begin
        sr = create_mock_solver_results(n_basis = 5)

        _, energies = convergence(sr)

        # Should be the same array (not a copy)
        @test energies === sr.energies
    end
end

@testset "correlation_function" begin

    @testset "Output dimensions" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)

        r_grid, ρ = correlation_function(sr; npoints = 100)

        @test length(r_grid) == 100
        @test length(ρ) == 100
    end

    @testset "Grid range" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)

        rmin, rmax = 0.5, 5.0
        r_grid, _ = correlation_function(sr; rmin = rmin, rmax = rmax, npoints = 50)

        @test first(r_grid) ≈ rmin
        @test last(r_grid) ≈ rmax
    end

    @testset "Non-negative density" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)

        _, ρ = correlation_function(sr; normalize = false)

        # r²|ψ|² should always be non-negative
        @test all(ρ .>= 0)
    end

    @testset "Normalization" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)

        r_grid, ρ_norm = correlation_function(sr; normalize = true, npoints = 500)
        _, ρ_unnorm = correlation_function(sr; normalize = false, npoints = 500)

        # Normalized should integrate to ~1
        dr = r_grid[2] - r_grid[1]
        integral_norm = sum(ρ_norm) * dr

        # Check that normalization changed something (unless already normalized)
        if sum(ρ_unnorm) * dr > 1.0e-10
            @test integral_norm ≈ 1.0 rtol = 0.1  # Rough due to trapezoidal rule
        end
    end

    @testset "coord_index selection" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 3)

        # Should work for all valid indices
        for idx in 1:3
            r_grid, ρ = correlation_function(sr; coord_index = idx)
            @test length(r_grid) > 0
            @test all(isfinite, ρ)
        end
    end

    @testset "Invalid coord_index" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 2)

        @test_throws ArgumentError correlation_function(sr; coord_index = 0)
        @test_throws ArgumentError correlation_function(sr; coord_index = 3)
        @test_throws ArgumentError correlation_function(sr; coord_index = -1)
    end

    @testset "Returns Vector not Range" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 2)

        r_grid, _ = correlation_function(sr)

        @test r_grid isa Vector
    end

    @testset "Finite values" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)

        r_grid, ρ = correlation_function(sr)

        @test all(isfinite, r_grid)
        @test all(isfinite, ρ)
    end

    @testset "Custom npoints" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 2)

        for np in [10, 100, 1000]
            r_grid, ρ = correlation_function(sr; npoints = np)
            @test length(r_grid) == np
            @test length(ρ) == np
        end
    end
end


@testset "Integration: Utils with real solver" begin

    @testset "Hydrogen atom utilities" begin
        # Set up hydrogen atom
        masses = [1.0e15, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = _jacobi_transform(masses)
        w_raw = [U' * [1.0, -1.0]]
        coulomb = CoulombOperator(-1.0, w_raw[1])
        ops = Operator[kin, coulomb]

        result = solve_ECG(ops, 15; scale = 1.5, verbose = false)

        # Test convergence
        indices, energies = convergence(result)
        @test length(indices) == result.n_basis
        @test energies[end] == result.ground_state

        # Test wavefunction evaluation
        r = [0.5]
        ψ_val = ψ₀(r, result; state = 1)
        @test isfinite(ψ_val)

        r_grid, ρ = correlation_function(result; npoints = 100)
        @test all(ρ .>= 0)
        @test length(r_grid) == 100
    end

    @testset "Three-body utilities" begin
        # Set up three-body system
        masses = [1000.0, 1000.0, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = _jacobi_transform(masses)

        w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
        w_raw = [U' * w for w in w_list]
        coeffs = [+1.0, -1.0, -1.0]
        coulomb_ops = [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]
        ops = Operator[kin; coulomb_ops...]

        result = solve_ECG(ops, 10; scale = 1.0, verbose = false)

        _, energies = convergence(result)
        for i in 2:length(energies)
            @test energies[i] <= energies[i - 1] + 1.0e-10
        end

        for coord_idx in 1:2
            r_grid, ρ = correlation_function(result; coord_index = coord_idx)
            @test all(isfinite, ρ)
        end
    end
end

@testset "Edge cases" begin

    @testset "Very small basis" begin
        basis_fns = [Rank0Gaussian([1.0;;], [0.0])]
        ops = Operator[KineticOperator([0.5;;])]
        eigvecs = [ones(1, 1)]

        sr = SolverResults(
            basis_fns, 1, ops, :quasirandom, HaltonSample(),
            1.0, -0.5, [-0.5], eigvecs
        )

        # All utilities should work
        @test ψ₀([0.0], sr) ≈ 1.0
        @test convergence(sr) == (1:1, [-0.5])

        r_grid, ρ = correlation_function(sr)
        @test length(r_grid) > 0
    end

    @testset "Large coordinates" begin
        sr = create_mock_solver_results(n_basis = 3, dim = 2)

        r_large = [100.0, 100.0]
        ψ_val = ψ₀(r_large, sr)

        @test isfinite(ψ_val)
        @test abs(ψ_val) < 1.0e-10
    end

    @testset "Zero at correlation function boundaries" begin
        sr = create_mock_solver_results(n_basis = 5, dim = 2)

        r_grid, ρ = correlation_function(sr; rmin = 1.0e-6, rmax = 1.0)
        @test ρ[1] ≈ 0 atol = 1.0e-10  # r²|ψ|² → 0 as r → 0
    end
end
