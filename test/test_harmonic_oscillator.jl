using Test
using FewBodyECG
using FewBodyHamiltonians

@testset "3D relative harmonic oscillator" begin
    exponents = 10 .^ range(-1, 1, length = 15)
    basis = BasisSet(
        Rank0Gaussian[
            Rank0Gaussian([α;;], zeros(1, 3)) for α in exponents
        ]
    )
    operators = Operator[
        KineticOperator([1 / 2;;]),
        OscillatorPotential(1 / 2, [1]),
    ]

    H = build_hamiltonian_matrix(basis, operators)
    S = build_overlap_matrix(basis)
    energies, _ = solve_generalized_eigenproblem(H, S)

    @test energies[1:3] ≈ [3 / 2, 7 / 2, 11 / 2] atol = 1.0e-5
end
