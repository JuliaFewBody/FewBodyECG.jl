using OffsetArrays

@testset "generic axes" begin
    @testset "solve_generalized_eigenproblem" begin
        n = 4
        A = randn(n, n); H = A + A'
        B = randn(n, n); S = B * B' + 5I

        evals, vecs = solve_generalized_eigenproblem(H, S)

        Hv = view(H, :, :)
        Sv = view(S, :, :)
        evalsv, vecsv = solve_generalized_eigenproblem(Hv, Sv)
        @test evalsv ≈ evals
        @test vecsv ≈ vecs

        Ho = OffsetArray(H, -1, -1)
        So = OffsetArray(S, -1, -1)
        @test_throws ArgumentError solve_generalized_eigenproblem(Ho, So)
    end

    @testset "Wavefunction call operator" begin
        ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
        ops += "Kinetic"
        ops += "Coulomb"
        sol = solve(ops, SVM(basis = 10, candidates = 5, scale = 1.0))
        ψ = wavefunction(sol)

        r = [0.7]
        val = ψ(r)

        @test ψ(view(r, :)) ≈ val
        @test_throws ArgumentError ψ(OffsetVector(r, -1))
    end

    @testset "build_hamiltonian_matrix operators axis" begin
        ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
        ops += "Kinetic"
        ops += "Coulomb"
        sol = solve(ops, SVM(basis = 10, candidates = 5, scale = 1.0))

        Hplain = build_hamiltonian_matrix(sol.basis, ops.terms)
        Hoff = build_hamiltonian_matrix(sol.basis, OffsetVector(collect(ops.terms), -3))
        @test Hplain ≈ Hoff
    end
end
