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
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 12, candidates = 10, scale = 1.0))
sol
```

The exact hydrogen ground-state energy is `-0.5` Ha.  The reported
convergence is a statement about the sampled basis, while the energy remains a
variational upper bound.

```@example quickstart
plot(sol, -0.5)
```

Read [Building systems](@ref) for `Operators`, [Choosing a solver](@ref) for
method selection, [Convergence](@ref) for how to interpret a `Solution`, and
the example gallery for complete scripts.
