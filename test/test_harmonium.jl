using Test
using FewBodyECG

# Hooke's atom (harmonium): two electrons in a harmonic trap with Coulomb
# repulsion.  For ω = 1/2 the ground-state energy is exactly 2 Ha
# (Taut, Phys. Rev. A 48, 3561 (1993)).  Exercises the OscillatorOperator
# together with Coulomb repulsion on the plain stochastic solver.
@testset "Hooke's atom vs Taut (ω = 1/2)" begin
    ω = 0.5
    ops = Operators([1.0e15, 1.0, 1.0])
    ops += "Kinetic"
    ops += ("Oscillator", 1, 2, 0.5 * ω^2)
    ops += ("Oscillator", 1, 3, 0.5 * ω^2)
    ops += ("Coulomb", 2, 3, 1.0)

    sol = solve(ops, SVM(basis = 80, candidates = 40, scale = 2.0))

    @test sol.E₀ ≈ 2.0 atol = 1.0e-3      # Taut's exact energy
    @test sol.E₀ > 2.0 - 1.0e-9           # variational upper bound never violated

    # Relative wavefunction vs Taut's closed form χ(u) ∝ (1 + u/2) e^{-u²/8}.
    # Transform physical electron positions back to Jacobi coordinates via J.
    ψ = wavefunction(sol)
    J, _ = jacobi_transform([1.0e15, 1.0, 1.0])
    Ψ(z₁, z₂) = ψ(J * [0.0, z₁, z₂])
    u = range(0, 10, length = 200)
    χ_ecg = [Ψ(x / 2, -x / 2) for x in u]
    χ_exact = [(1 + x / 2) * exp(-x^2 / 8) for x in u]
    χ_ecg ./= maximum(abs, χ_ecg)
    χ_exact ./= maximum(χ_exact)
    χ_ecg .*= sign(sum(χ_ecg .* χ_exact))
    @test maximum(abs, χ_ecg .- χ_exact) < 0.02
end
