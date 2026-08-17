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