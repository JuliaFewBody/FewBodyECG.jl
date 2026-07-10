```@meta
EditURL = "../../../examples/harmonium.jl"
```

# Hooke's atom (harmonium)

Two electrons in a common harmonic trap, repelling through the Coulomb
interaction:

```math
H = -\tfrac12\nabla_1^2 - \tfrac12\nabla_2^2
    + \tfrac12\omega^2 (r_1^2 + r_2^2) + \frac{1}{r_{12}}.
```

For the trap frequency ``\omega = \tfrac12`` the ground-state energy is known
in closed form (Taut, *Phys. Rev. A* **48**, 3561 (1993)): **exactly 2 Ha**.
The trap is supplied by an `OscillatorOperator` between a fixed heavy centre
and each electron (coefficient ``\tfrac12\omega^2``); the only Coulomb term is
the electron–electron repulsion.  The system is spherically symmetric and real,
so the plain stochastic `SVM` solver applies.

````@example harmonium
using FewBodyECG
using Plots

ω = 0.5
masses = [1.0e15, 1.0, 1.0]          # heavy trap centre + two electrons

ops = Operators(masses)
ops += "Kinetic"
ops += ("Oscillator", 1, 2, 0.5 * ω^2)   # ½ω² r₁²  (trap on electron 1)
ops += ("Oscillator", 1, 3, 0.5 * ω^2)   # ½ω² r₂²  (trap on electron 2)
ops += ("Coulomb", 2, 3, 1.0)            # +1/r₁₂ electron–electron repulsion

exact = 2.0                              # Taut 1993, ω = 1/2

sol = solve(ops, SVM(basis = 80, candidates = 40, scale = 2.0))
println("Hooke's atom E₀ = ", sol.E₀, " Ha   (Taut exact ", exact, ", Δ = ", sol.E₀ - exact, ")")
println("variational upper bound respected: ", sol.E₀ ≥ exact)
sol

# Convergence toward the exact energy
plot(sol, exact)

# Ground-state radial density of one electron
plot(wavefunction(sol); coord = 1, rmax = 8.0)
plot(convergence(sol))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

