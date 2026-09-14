# Code examples

## Scattering

The standalone [scattering example](https://github.com/JuliaFewBody/FewBodyECG.jl/blob/main/examples/scattering.jl)
computes a converged trapped spectrum for a finite-range Gaussian interaction,
then extracts the s-wave scattering length, effective range, and shape
coefficient. See the [scattering guide](scattering.md) for the physical
conventions, diagnostics, and supported domain.

## Positronium

```@example positronium
using FewBodyECG
using Plots
import Antique

H = Operators([1.0, 1.0], [+1.0, -1.0])
H += "Kinetic"
H += "Coulomb"

ps = Antique.CoulombTwoBody(
    z₁ = 1, z₂ = -1, m₁ = 1.0, m₂ = 1.0, mₑ = 1.0, a₀ = 1.0, Eₕ = 1.0, ħ = 1.0
)

exact = Antique.E(ps, n = 1)
sol = solve(H, SVM(basis = 25, candidates = 20, scale = 1.4))

plot(sol, exact)
```
