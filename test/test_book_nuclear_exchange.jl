module BookNuclearExchangeTests

using Test, LinearAlgebra, FewBodyECG
include(joinpath(@__DIR__, "..", "examples", "nuclear_exchange_model.jl"))

# Table 11.1, p. 217: ATS3 contains Wigner and Bartlett terms.
const ATS3_ROWS = [
    (1000.0, 3.0, 1.0, 0.0, 0.0, 0.0),
    (-326.7, 1.05, 0.5, 0.0, 0.5, 0.0),
    (-166.0, 0.8, 0.5, 0.0, -0.5, 0.0),
    (-43.0, 0.6, 0.5, 0.0, 0.5, 0.0),
    (-23.0, 0.4, 0.5, 0.0, -0.5, 0.0),
]

function channel_reference(orbitals, model, L, S, u, w)
    terms = if model == :Minnesota
        factor = iseven(L) ? 1.0 : u - 1
        [(200factor, 1.487), (S == 1 ? -178factor : 0.0, 0.639), (S == 0 ? -91.85factor : 0.0, 0.465)]
    else
        [
            (1000.0, 3.0), (S == 1 ? -326.7 : 0.0, 1.05), (S == 0 ? -166.0 : 0.0, 0.8),
            (S == 1 ? -43.0 : 0.0, 0.6), (S == 0 ? -23.0 : 0.0, 0.4),
        ]
    end
    return sum(FewBodyECG.build_operator_matrix(BasisSet(orbitals), GaussianOperator(V, γ, w)) for (V, γ) in terms)
end

@testset "Book nuclear exchange physics" begin
    masses = fill(1 / 41.47, 2)
    _, U = FewBodyECG.jacobi_transform(masses)
    w = U' * [1.0, -1.0]
    Pr = PermutationOperator(masses, [2, 1])
    @testset "Minnesota and ATS3 channels" begin
        for L in 0:2
            orbitals = pair_orbitals(L, [0.03, 0.08, 0.2, 0.6, 1.5, 4.0], w)
            basis = nucleon_basis(orbitals)
            overlap = FewBodyECG.build_overlap_matrix(BasisSet(orbitals))
            D = Diagonal(1 ./ sqrt.(diag(overlap)))
            Ps = FewBodyECG.build_operator_matrix(basis, SpinExchangeOperator(1, 2))
            Pt = FewBodyECG.build_operator_matrix(basis, IsospinExchangeOperator(1, 2))
            Pauli = FewBodyECG.build_operator_matrix(basis, Pr * SpinExchangeOperator(1, 2) * IsospinExchangeOperator(1, 2))
            for (model, u, rows) in (
                    (:Minnesota, 1.0, minnesota_rows(1.0)),
                    (:Minnesota, 1.5, minnesota_rows(1.5)), (:ATS3, 1.0, ATS3_ROWS),
                )
                Vspace = exchange_matrix(basis, rows, w, Pr)
                Viso = exchange_matrix(basis, rows, w, Pr; isospin_form = true)
                for S in 0:1
                    T = iseven(L) ? 1 - S : S
                    C = channel_map(length(orbitals), S, T)
                    reference = D * channel_reference(orbitals, model, L, S, u, w) * D
                    @test D * C' * Vspace * C * D ≈ reference rtol = 1.0e-10 atol = 1.0e-10
                    @test D * C' * Viso * C * D ≈ reference rtol = 1.0e-10 atol = 1.0e-10
                    @test D * C' * Ps * C * D ≈ (2S - 1) * D * overlap * D atol = 1.0e-10
                    @test D * C' * Pt * C * D ≈ (2T - 1) * D * overlap * D atol = 1.0e-10
                    @test D * C' * Pauli * C * D ≈ -D * overlap * D atol = 1.0e-10
                end
            end
        end
    end

    @testset "Deuteron solution and physical radial normalization" begin
        orbitals = pair_orbitals(0, exp10.(range(-4, 2, length = 24)), w)
        basis = nucleon_basis(orbitals)
        overlap = FewBodyECG.build_overlap_matrix(basis)
        H = FewBodyECG.build_operator_matrix(basis, KineticOperator(masses)) +
            exchange_matrix(basis, minnesota_rows(1.0), w, Pr; isospin_form = true)
        C = channel_map(length(orbitals), 1, 0)
        Ht, St = C' * H * C, C' * overlap * C
        states = [channel_solution(Ht[1:k, 1:k], St[1:k, 1:k], orbitals[1:k]) for k in (12, 18, 24)]
        final = last(states)
        @test final.energy ≈ -2.202 atol = 0.001
        @test all(diff([state.energy for state in states]) .≤ 1.0e-8)
        S_orb = FewBodyECG.build_overlap_matrix(BasisSet(orbitals))
        @test dot(final.coefficients, S_orb * final.coefficients) ≈ 1 atol = 1.0e-10
        r = range(0.0, 60.0, length = 3001)
        u = s_wave_radial_amplitude(final.wavefunction, r, w)
        P = abs2.(u)
        integral = step(r) * sum((P[1:(end - 1)] + P[2:end]) / 2)
        @test integral ≈ 1 atol = 1.0e-5
        R2 = FewBodyECG.build_operator_matrix(BasisSet(orbitals), OscillatorOperator(1.0, w))
        moment = dot(final.coefficients, R2 * final.coefficients) / dot(final.coefficients, S_orb * final.coefficients)
        sampled = collect(r) .^ 2 .* P
        radial_moment = step(r) * sum((sampled[1:(end - 1)] + sampled[2:end]) / 2)
        @test radial_moment ≈ moment rtol = 1.0e-5
        full_c = C * final.coefficients
        for (op, value) in ((SpinExchangeOperator(1, 2), 1), (IsospinExchangeOperator(1, 2), -1), (Pr, 1))
            M = FewBodyECG.build_operator_matrix(basis, op)
            @test dot(full_c, M * full_c) / dot(full_c, overlap * full_c) ≈ value atol = 1.0e-10
        end
    end
end

end
