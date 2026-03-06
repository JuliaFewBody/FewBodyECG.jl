using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
import FewBodyECG: _jacobi_transform, Λ

@testset "Operators" begin

    @testset "Construction" begin

        @testset "Empty (system-unaware)" begin
            ops = Operators()
            @test length(ops) == 0
            @test ops.masses === nothing
            @test ops.charges === nothing
            @test ops._U === nothing
        end

        @testset "With masses only" begin
            masses = [1.0e15, 1.0]
            ops = Operators(masses)
            @test ops.masses ≈ Float64.(masses)
            @test ops.charges === nothing
            @test ops._U !== nothing
            @test size(ops._U) == (2, 1)   # N × (N-1) Jacobi matrix
            @test length(ops) == 0
        end

        @testset "With masses and charges" begin
            masses = [1.0e15, 1.0, 1.0]
            charges = [+1, -1, -1]
            ops = Operators(masses, charges)
            @test ops.masses ≈ Float64.(masses)
            @test ops.charges ≈ Float64.(charges)
            @test ops._U !== nothing
            @test size(ops._U) == (3, 2)   # N × (N-1) Jacobi matrix
        end

        @testset "Error: mismatched masses/charges" begin
            @test_throws ArgumentError Operators([1.0, 2.0], [1.0])
            @test_throws ArgumentError Operators([1.0, 2.0, 3.0], [1.0, -1.0])
        end
    end

    @testset "Adding raw operators" begin

        @testset "KineticOperator" begin
            ops = Operators()
            K = KineticOperator([0.5;;])
            ops += K
            @test length(ops) == 1
            @test ops[1] isa KineticOperator
        end

        @testset "CoulombOperator" begin
            ops = Operators()
            V = CoulombOperator(-1.0, [1.0])
            ops += V
            @test length(ops) == 1
            @test ops[1] isa CoulombOperator
            @test ops[1].coefficient ≈ -1.0
        end

        @testset "Multiple raw operators" begin
            ops = Operators()
            ops += KineticOperator([0.5;;])
            ops += CoulombOperator(-1.0, [1.0])
            ops += CoulombOperator(+0.5, [1.0])
            @test length(ops) == 3
        end
    end

    @testset "String: Kinetic" begin

        @testset "Adds correct KineticOperator" begin
            masses = [1.0e15, 1.0]
            ops = Operators(masses)
            ops += "Kinetic"
            @test length(ops) == 1
            @test ops[1] isa KineticOperator
        end

        @testset "Three-body kinetic" begin
            masses = [1.0e15, 1.0, 1.0]
            ops = Operators(masses)
            ops += "Kinetic"
            @test length(ops) == 1
            @test ops[1] isa KineticOperator
        end

        @testset "Error: no masses" begin
            ops = Operators()
            @test_throws ArgumentError ops += "Kinetic"
        end

        @testset "Error: unknown string" begin
            ops = Operators([1.0e15, 1.0])
            @test_throws ArgumentError ops += "BadOp"
        end
    end

    @testset "Tuple: explicit Coulomb pair" begin

        @testset "Adds correct CoulombOperator" begin
            masses = [1.0e15, 1.0, 1.0]
            ops = Operators(masses)
            ops += ("Coulomb", 1, 2, -1.0)
            @test length(ops) == 1
            @test ops[1] isa CoulombOperator
            @test ops[1].coefficient ≈ -1.0
        end

        @testset "Coefficient is stored correctly" begin
            masses = [1.0e15, 1.0, 1.0]
            ops = Operators(masses)
            ops += ("Coulomb", 1, 3, +2.5)
            @test ops[1].coefficient ≈ 2.5
        end

        @testset "Weight vector has correct dimension" begin
            masses = [1.0e15, 1.0, 1.0]   # 3-body → 2 Jacobi dims
            ops = Operators(masses)
            ops += ("Coulomb", 1, 2, -1.0)
            @test length(ops[1].w) == 2
        end

        @testset "Error: no masses" begin
            ops = Operators()
            @test_throws ArgumentError ops += ("Coulomb", 1, 2, -1.0)
        end

        @testset "Error: same index (i == j)" begin
            ops = Operators([1.0e15, 1.0, 1.0])
            @test_throws ArgumentError ops += ("Coulomb", 1, 1, -1.0)
            @test_throws ArgumentError ops += ("Coulomb", 2, 2, -1.0)
        end

        @testset "Error: index out of range" begin
            ops = Operators([1.0e15, 1.0, 1.0])
            @test_throws ArgumentError ops += ("Coulomb", 0, 1, -1.0)
            @test_throws ArgumentError ops += ("Coulomb", 1, 4, -1.0)
            @test_throws ArgumentError ops += ("Coulomb", -1, 2, -1.0)
        end

        @testset "Error: unknown operator name in tuple" begin
            ops = Operators([1.0e15, 1.0, 1.0])
            @test_throws ArgumentError ops += ("Kinetic", 1, 2, -1.0)
            @test_throws ArgumentError ops += ("Unknown", 1, 2, -1.0)
        end
    end

    @testset "String: auto all-pairs Coulomb" begin

        @testset "Two-body: one pair" begin
            ops = Operators([1.0e15, 1.0], [+1, -1])
            ops += "Coulomb"
            @test length(ops) == 1
            @test ops[1] isa CoulombOperator
            @test ops[1].coefficient ≈ -1.0   # (+1) * (-1)
        end

        @testset "Three-body: three pairs" begin
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops += "Coulomb"
            @test length(ops) == 3
            @test all(op isa CoulombOperator for op in ops)
        end

        @testset "Coefficients follow q_i * q_j" begin
            # proton (+1), e₁ (-1), e₂ (-1)
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops += "Coulomb"
            coeffs = [ops[i].coefficient for i in 1:3]
            # pairs (1,2), (1,3), (2,3) → -1, -1, +1
            @test coeffs[1] ≈ -1.0
            @test coeffs[2] ≈ -1.0
            @test coeffs[3] ≈ +1.0
        end

        @testset "Four-body: six pairs" begin
            ops = Operators([1.0e15, 1.0, 1.0, 1.0], [+1, -1, -1, -1])
            ops += "Coulomb"
            @test length(ops) == 6   # C(4,2)
        end

        @testset "Error: no charges" begin
            ops = Operators([1.0e15, 1.0, 1.0])
            @test_throws ArgumentError ops += "Coulomb"
        end
    end

    @testset "Collection interface" begin

        @testset "length" begin
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            @test length(ops) == 0
            ops += "Kinetic"
            @test length(ops) == 1
            ops += "Coulomb"
            @test length(ops) == 4   # 1 kinetic + 3 Coulomb
        end

        @testset "getindex" begin
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops += "Kinetic"
            ops += "Coulomb"
            @test ops[1] isa KineticOperator
            @test ops[2] isa CoulombOperator
            @test ops[4] isa CoulombOperator
        end

        @testset "iterate" begin
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops += "Kinetic"
            ops += "Coulomb"
            collected = collect(ops)
            @test collected[1] isa KineticOperator
            @test all(op isa CoulombOperator for op in collected[2:4])
        end

        @testset "eltype" begin
            @test eltype(Operators) == FewBodyHamiltonians.Operator
        end
    end

    @testset "show" begin

        @testset "Empty (no masses)" begin
            ops = Operators()
            str = sprint(show, ops)
            @test occursin("Operators", str)
            @test occursin("0 terms", str)
        end

        @testset "With masses (n-body header)" begin
            ops = Operators([1.0e15, 1.0])
            ops += "Kinetic"
            str = sprint(show, ops)
            @test occursin("2-body", str)
            @test occursin("Kinetic", str)
        end

        @testset "With masses and charges" begin
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops += "Kinetic"
            ops += "Coulomb"
            str = sprint(show, ops)
            @test occursin("charges", str)
            @test occursin("Kinetic", str)
            @test occursin("Coulomb", str)
        end

        @testset "Singular 'term' for 1 operator" begin
            ops = Operators()
            ops += KineticOperator([0.5;;])
            str = sprint(show, ops)
            @test occursin("1 term", str)
            @test !occursin("1 terms", str)
        end
    end

    @testset "coulomb_weights" begin

        @testset "Returns only Coulomb weights (skips Kinetic)" begin
            ops = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops += "Kinetic"
            ops += "Coulomb"
            ws = coulomb_weights(ops)
            @test length(ws) == 3   # 3 Coulomb pairs, not 4
        end

        @testset "Empty if no Coulomb operators" begin
            ops = Operators([1.0e15, 1.0])
            ops += "Kinetic"
            @test isempty(coulomb_weights(ops))
        end

        @testset "Correct Jacobi dimension" begin
            # 2-body → 1 Jacobi coordinate
            ops2 = Operators([1.0e15, 1.0], [+1, -1])
            ops2 += "Coulomb"
            @test all(length(w) == 1 for w in coulomb_weights(ops2))

            # 3-body → 2 Jacobi coordinates
            ops3 = Operators([1.0e15, 1.0, 1.0], [+1, -1, -1])
            ops3 += "Coulomb"
            @test all(length(w) == 2 for w in coulomb_weights(ops3))
        end

        @testset "Returns Vector{Vector{Float64}}" begin
            ops = Operators([1.0e15, 1.0], [+1, -1])
            ops += "Coulomb"
            ws = coulomb_weights(ops)
            @test ws isa Vector
            @test eltype(ws) == Vector{Float64}
        end
    end

    @testset "Equivalence with manual construction" begin

        @testset "Two-body hydrogen atom" begin
            masses = [1.0e15, 1.0]

            # New Operators interface
            ops_new = Operators(masses)
            ops_new += "Kinetic"
            ops_new += ("Coulomb", 1, 2, -1.0)

            # Manual interface
            Λmat = Λ(masses)
            _, U = _jacobi_transform(masses)
            w = U' * [1.0, -1.0]
            ops_old = Operator[KineticOperator(Λmat); CoulombOperator(-1.0, w)]

            g = Rank0Gaussian([1.0;;], [0.0])
            basis = BasisSet([g])

            H_new = build_hamiltonian_matrix(basis, ops_new)
            H_old = build_hamiltonian_matrix(basis, ops_old)
            @test H_new ≈ H_old rtol = 1.0e-12
        end

        @testset "Three-body H⁻ with auto-Coulomb" begin
            masses = [1.0e15, 1.0, 1.0]
            charges = [+1, -1, -1]

            ops_new = Operators(masses, charges)
            ops_new += "Kinetic"
            ops_new += "Coulomb"

            Λmat = Λ(masses)
            _, U = _jacobi_transform(masses)
            w_list = [U' * Float64.(w) for w in [[1, -1, 0], [1, 0, -1], [0, 1, -1]]]
            coeffs = [-1.0, -1.0, +1.0]
            ops_old = Operator[
                KineticOperator(Λmat);
                [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_list)]...
            ]

            g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
            basis = BasisSet([g])

            H_new = build_hamiltonian_matrix(basis, ops_new)
            H_old = build_hamiltonian_matrix(basis, ops_old)
            @test H_new ≈ H_old rtol = 1.0e-12
        end

        @testset "Explicit pair Coulomb matches manual" begin
            masses = [1.0e15, 1.0, 1.0]
            _, U = _jacobi_transform(masses)

            ops_new = Operators(masses)
            ops_new += ("Coulomb", 1, 2, -1.0)
            ops_new += ("Coulomb", 1, 3, -1.0)
            ops_new += ("Coulomb", 2, 3, +1.0)

            w12 = U' * [1.0, -1.0, 0.0]
            w13 = U' * [1.0, 0.0, -1.0]
            w23 = U' * [0.0, 1.0, -1.0]
            ops_old = Operator[
                CoulombOperator(-1.0, w12),
                CoulombOperator(-1.0, w13),
                CoulombOperator(+1.0, w23),
            ]

            g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
            basis = BasisSet([g])

            H_new = build_hamiltonian_matrix(basis, ops_new)
            H_old = build_hamiltonian_matrix(basis, ops_old)
            @test H_new ≈ H_old rtol = 1.0e-12
        end
    end

    @testset "Integration with solvers" begin

        @testset "solve_ECG: hydrogen atom ≈ -0.5 Ha" begin
            masses = [1.0e15, 1.0]
            ops = Operators(masses)
            ops += "Kinetic"
            ops += ("Coulomb", 1, 2, -1.0)
            sr = solve_ECG(ops, 25; scale = 1.0, verbose = false)
            @test sr.ground_state < -0.46   # converging toward -0.5 Ha
            @test sr.ground_state > -0.52
        end

        @testset "solve_ECG: H⁻ auto-Coulomb (bound state)" begin
            masses = [1.0e15, 1.0, 1.0]
            ops = Operators(masses, [+1, -1, -1])
            ops += "Kinetic"
            ops += "Coulomb"
            sr = solve_ECG(ops, 15; scale = 1.0, verbose = false)
            # H⁻ ground state ≈ -0.528 Ha; with a small basis just verify it is bound
            @test sr.ground_state < -0.3
        end

        @testset "solve_ECG via Operators matches ops.terms" begin
            masses = [1.0e15, 1.0]
            ops = Operators(masses)
            ops += "Kinetic"
            ops += ("Coulomb", 1, 2, -1.0)

            sr_ops = solve_ECG(ops, 25; scale = 1.0, verbose = false)
            sr_vec = solve_ECG(ops.terms, 25; scale = 1.0, verbose = false)

            @test sr_ops.ground_state < -0.46
            @test sr_vec.ground_state < -0.46
        end
    end
end
