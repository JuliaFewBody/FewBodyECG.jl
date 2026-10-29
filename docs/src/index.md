# FewBodyECG.jl

FewBodyECG.jl builds variational explicitly correlated Gaussian bases for
few-body quantum systems.  You write the Hamiltonian as a sum of
particle-indexed terms, choose a solver method, and get back a `Solution` with energies,
coefficients, convergence information, and plotting recipes.

## Installation

```julia
import Pkg
Pkg.add("FewBodyECG")
```

## Quick Start

```@example quickstart
using FewBodyECG
import Antique
using Plots

H = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])

sol = solve(H, DynamicGVM(basis = 10, candidates = 20))
exact = Antique.energy(Antique.HydrogenAtom(Z = 1), n = 1)
println("E0 = ", sol.E₀, " Ha  (Antique ", exact, ", Δ = ", sol.E₀ - exact, ")")
sol
```

[Building systems](systems.md) lists every term you can add, the unit
conventions, and how to choose `scale`; [Choosing a solver](solvers.md)
compares the solver methods.

## Custom numerical potentials

Define a radial pair potential as a scalar callable. The matrix elements are
reduced analytically to a one-dimensional radial integral, which `Custom`
evaluates with `QuadGK`:

```@example numerical-potential
using FewBodyECG

f(r) = -exp(-r^2) / (1 + r^2)
ops = Kinetic([1.0e15, 1.0]) + Custom(f, 1, 2)

sol = solve(ops, SVM(basis = 8, candidates = 2))
println(sol.E₀)
```

The callable receives the nonnegative pair distance `r`; users do not need to
call `QuadGK` or construct matrix elements manually. For a precomputed Jacobi
weight vector `w`, the equivalent low-level form is `NumericalPotential(f, w)`.
