# FewBodyECG.jl

FewBodyECG.jl builds variational explicitly correlated Gaussian bases for
few-body quantum systems.  You define particles and pair interactions with
`Operators`, choose a solver method, and get back a `Solution` with energies,
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

H = Operators([1.0e15, 1.0], [+1.0, -1.0])
H += "Kinetic"
H += "Coulomb"

sol = solve(H, GrowVariational(basis = 10, candidates = 20, scale = 1.0))
exact = Antique.E(Antique.HydrogenAtom(Z = 1), n = 1)
println("E0 = ", sol.E₀, " Ha  (Antique ", exact, ", Δ = ", sol.E₀ - exact, ")")
sol
```

## Custom numerical potentials

Define a radial pair potential as a scalar callable. The matrix elements are
reduced analytically to a one-dimensional radial integral, and `QuadGK` is
used internally by `NumericalPotential`:

```@example numerical-potential
using FewBodyECG

ops = Operators([1.0e15, 1.0])
ops += "Kinetic"
f(r) = -exp(-r^2) / (1 + r^2)
ops += (f, numerical, 1, 2)

sol = solve(ops, SVM(basis = 8, candidates = 2, scale = 1.0))
println(sol.E₀)
```

The callable receives the nonnegative pair distance `r`; users do not need to
call `QuadGK` or construct matrix elements manually. For a precomputed Jacobi
weight vector `w`, the equivalent low-level form is `NumericalPotential(f, w)`.
