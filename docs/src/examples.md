# Code examples

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