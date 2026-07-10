using Test
using FewBodyECG

# Nuclear few-body benchmarks with Gaussian NN potentials (energies in MeV,
# lengths in fm; particle "mass" = mc²/(ħc)² so ħ²/2m → (ħc)²/2mc²).
@testset "Nuclear Gaussian potentials (deuteron, triton)" begin
    ħc = 197.3269804
    mpkg(mc²) = mc² / ħc^2
    mp, mn = mpkg(938.272), mpkg(939.565)

    # ħ²/2mₙ ≈ 20.7 MeV·fm² for a nucleon
    @test 1 / (2 * mpkg(938.918)) ≈ 20.736 atol = 1.0e-2

    # Deuteron — Minnesota triplet-even central: benchmark ≈ -2.202 MeV
    deut = Operators([mp, mn])
    deut += "Kinetic"
    deut += ("Gaussian", 1, 2, 200.0, 1.487)
    deut += ("Gaussian", 1, 2, -178.0, 0.639)
    sol_d = solve(deut, SVM(basis = 40, candidates = 25, scale = 3.0))
    @test sol_d.E₀ ≈ -2.202 atol = 5.0e-3
    @test sol_d.E₀ > -2.202 - 1.0e-3          # variational upper bound

    # Triton — Volkov V1 central on all pairs: benchmark ≈ -8.46 MeV
    γR, γA = 1 / 0.82^2, 1 / 1.6^2
    trit = Operators([mn, mn, mp])
    trit += "Kinetic"
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        trit += ("Gaussian", i, j, 144.86, γR)
        trit += ("Gaussian", i, j, -83.34, γA)
    end
    sol_t = solve(trit, SVM(basis = 100, candidates = 30, scale = 3.5))
    @test sol_t.E₀ ≈ -8.46 atol = 2.0e-2
    @test sol_t.E₀ > -8.48                     # variational upper bound

    # the three-body system is bound well below the two-body deuteron
    @test sol_t.E₀ < sol_d.E₀ - 5.0
end
