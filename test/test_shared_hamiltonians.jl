import FewBodyHamiltonians

@testset "Hamiltonians constructed by the shared package" begin
    for name in (:Hamiltonian, :Kinetic, :Coulomb, :Operator)
        @test getfield(FewBodyECG, name) === getfield(FewBodyHamiltonians, name)
    end

    if FewBodyECG.Hamiltonian === FewBodyHamiltonians.Hamiltonian
        for name in (:Gaussian, :ManyBodyGaussian, :Oscillator, :Custom, :Tensor, :SpinOrbit, :UnitSystem)
            @test getfield(FewBodyECG, name) === getfield(FewBodyHamiltonians, name)
        end
        H = FewBodyHamiltonians.Kinetic([1.0e15, 1.0]) +
            FewBodyHamiltonians.Coulomb([1, -1])
        basis = BasisSet([Rank0Gaussian(reshape([1.0], 1, 1), zeros(1))])
        raw = Operator[KineticOperator([1.0e15, 1.0]), CoulombOperator(-1.0, [1.0])]
        @test FewBodyECG.build_hamiltonian_matrix(basis, H) ≈
            FewBodyECG.build_hamiltonian_matrix(basis, raw)
        solution = solve(H, SVM(8; candidates = 8, scale = 1.0))
        @test -0.500000000001 ≤ solution.E₀ < -0.45
        @test solution.hamiltonian === H
    end
end
