using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
using FewBodyECG: SATURATION_CAVEAT
import FewBodyECG: jacobi_transform, _generate_A_matrix, generate_bij, generate_shift
using QuasiMonteCarlo

function create_mock_solution(;
        n_basis::Int = 5,
        dim::Int = 2,
        scale::Float64 = 1.0,
        sampler = HaltonSample(),
        state::Int = 1
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

    # Mock per-step energy history (decreasing sequence)
    step_energies = [-0.1 * i for i in 1:n_basis]

    # Mock S-orthonormal-ish coefficient matrix (unit-norm columns)
    c = randn(n_basis, n_basis)
    for j in 1:n_basis
        c[:, j] ./= norm(c[:, j])
    end

    rep = ConvergenceReport(
        true, :saturation, 0.0, 1.0e-4, 20, nothing, 1.0, [SATURATION_CAVEAT]
    )
    method = SVM(basis = n_basis, scale = scale, sampler = sampler)
    stage = StageResult(method, step_energies, rep)
    E = sort(step_energies) # ascending; E[1] is the lowest (ground) energy
    return Solution(E, BasisSet(basis_fns), c, operators, state, [stage], rep)
end

@testset "Solution (mock construction)" begin

    @testset "Construction" begin
        sol = create_mock_solution(n_basis = 3, dim = 2)

        @test length(sol.basis.functions) == 3
        @test length(energies(sol)) == 3
        @test size(sol.coefficients, 2) == 3
        @test sol.stages[1].method.scale == 1.0
    end

    @testset "Ground state is last (best) step energy" begin
        sol = create_mock_solution(n_basis = 5)
        @test sol.E₀ == energies(sol)[end]
    end

    @testset "Operators stored correctly" begin
        sol = create_mock_solution()
        @test length(sol.operators) == 2
        @test sol.operators[1] isa KineticOperator
        @test sol.operators[2] isa CoulombOperator
    end

    @testset "Different samplers" begin
        sol_halton = create_mock_solution(n_basis = 1, sampler = HaltonSample())
        sol_sobol = create_mock_solution(n_basis = 1, sampler = SobolSample())

        @test sol_halton.stages[1].method.sampler isa HaltonSample
        @test sol_sobol.stages[1].method.sampler isa SobolSample
    end
end

@testset "wavefunction evaluation" begin

    @testset "Basic evaluation with coefficients" begin
        # Single Gaussian: ψ = c * exp(-r'Ar + s'r)
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        c = [1.0]

        r = [0.0, 0.0]
        ψ_val = Wavefunction(BasisSet(basis_fns), c)(r)

        # At origin with s=0: exp(-0 + 0) = 1
        @test ψ_val ≈ 1.0 rtol = 1.0e-10
    end

    @testset "Gaussian decay" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        c = [1.0]
        ψfn = Wavefunction(BasisSet(basis_fns), c)

        ψ_origin = ψfn([0.0, 0.0])
        ψ_far = ψfn([3.0, 3.0])

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
        ψfn = Wavefunction(BasisSet(basis_fns), c)

        # Maximum should be shifted
        ψ_origin = ψfn([0.0, 0.0])
        ψ_shifted = ψfn([1.0, 0.0])  # Closer to maximum

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

        ψ_val = Wavefunction(BasisSet(basis_fns), c)(r)

        # At origin: 0.5 * 1 + 0.5 * 1 = 1
        @test ψ_val ≈ 1.0 rtol = 1.0e-10
    end

    @testset "Negative coefficients" begin
        A = [1.0 0.0; 0.0 1.0]
        s = [0.0, 0.0]
        g = Rank0Gaussian(A, s)
        basis_fns = [g]
        r = [0.0, 0.0]

        ψ_pos = Wavefunction(BasisSet(basis_fns), [1.0])(r)
        ψ_neg = Wavefunction(BasisSet(basis_fns), [-1.0])(r)

        @test ψ_pos ≈ -ψ_neg rtol = 1.0e-10
    end

    @testset "With mock Solution" begin
        sol = create_mock_solution(n_basis = 3, dim = 2)

        # Should not throw
        r = [0.5, 0.5]
        ψ_val = wavefunction(sol; state = 1)(r)

        @test isfinite(ψ_val)
    end

    @testset "Different states" begin
        sol = create_mock_solution(n_basis = 5, dim = 2)
        r = [0.1, 0.1]

        # Different states should generally give different values
        ψ_1 = wavefunction(sol; state = 1)(r)
        ψ_2 = wavefunction(sol; state = 2)(r)

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
        ψ_val = Wavefunction(BasisSet(basis_fns), c)(r)

        @test ψ_val ≈ exp(-2.0) rtol = 1.0e-10
    end
end

@testset "energies helper (per-step history)" begin

    @testset "Returns correct range and energies" begin
        sol = create_mock_solution(n_basis = 10)

        idx, ener = (1:length(energies(sol)), energies(sol))

        @test idx == 1:10
        @test ener == energies(sol)
        @test length(idx) == length(ener)
    end

    @testset "Single basis function" begin
        sol = create_mock_solution(n_basis = 1)

        idx, ener = (1:length(energies(sol)), energies(sol))

        @test idx == 1:1
        @test length(ener) == 1
    end

    @testset "Energies are same object" begin
        sol = create_mock_solution(n_basis = 5)

        # Should be the same array (not a copy)
        @test energies(sol) === sol.stages[1].energies
    end
end

@testset "Integration: Utils with real solver" begin

    @testset "Hydrogen atom utilities" begin
        # Set up hydrogen atom
        masses = [1.0e15, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = jacobi_transform(masses)
        w_raw = [U' * [1.0, -1.0]]
        coulomb = CoulombOperator(-1.0, w_raw[1])
        ops = Operator[kin, coulomb]

        sol = solve(ops, SVM(basis = 15, candidates = 1, scale = 1.5))

        # Test per-step energy history
        idx, ener = (1:length(energies(sol)), energies(sol))
        @test length(idx) == length(sol.basis.functions)
        @test ener[end] == sol.E₀

        # Test wavefunction evaluation
        r = [0.5]
        ψ_val = wavefunction(sol; state = 1)(r)
        @test isfinite(ψ_val)
    end

    @testset "Three-body utilities" begin
        # Set up three-body system
        masses = [1000.0, 1000.0, 1.0]
        Λmat = Λ(masses)
        kin = KineticOperator(Λmat)
        J, U = jacobi_transform(masses)

        w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
        w_raw = [U' * w for w in w_list]
        coeffs = [+1.0, -1.0, -1.0]
        coulomb_ops = [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]
        ops = Operator[kin; coulomb_ops...]

        sol = solve(ops, SVM(basis = 10, candidates = 1, scale = 1.0))

        _, ener = (1:length(energies(sol)), energies(sol))
        for i in 2:length(ener)
            @test ener[i] <= ener[i - 1] + 1.0e-10
        end
    end
end

@testset "Edge cases" begin

    @testset "Very small basis" begin
        basis_fns = [Rank0Gaussian([1.0;;], [0.0])]
        ops = Operator[KineticOperator([0.5;;])]
        c = ones(1, 1)

        rep = ConvergenceReport(
            true, :saturation, 0.0, 1.0e-4, 20, nothing, 1.0, [SATURATION_CAVEAT]
        )
        stage = StageResult(SVM(1), [-0.5], rep)
        sol = Solution([-0.5], BasisSet(basis_fns), c, ops, 1, [stage], rep)

        # All utilities should work
        @test wavefunction(sol)([0.0]) ≈ 1.0
        @test (1:length(energies(sol)), energies(sol)) == (1:1, [-0.5])
    end

    @testset "Large coordinates" begin
        sol = create_mock_solution(n_basis = 3, dim = 2)

        r_large = [100.0, 100.0]
        ψ_val = wavefunction(sol)(r_large)

        @test isfinite(ψ_val)
        @test abs(ψ_val) < 1.0e-10
    end
end
