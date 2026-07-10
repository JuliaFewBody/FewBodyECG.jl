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
end
