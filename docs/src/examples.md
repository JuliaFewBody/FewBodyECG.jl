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

exact = Antique.energy(ps, n = 1)
sol = solve(H, SVM(basis = 25, candidates = 20, scale = 1.4))

plot(sol, exact)
```
## Reference energies from FewBodyDB

The examples below compare ECG energies with published benchmarks collected in
[FewBodyDB.jl](https://github.com/JuliaFewBody/FewBodyDB.jl). Each entry is
looked up by reference, system, observable and state; `FewBodyDB.bib` returns
its BibTeX source.

### Helium atom

The helium atom with an infinitely heavy nucleus, compared with the Suzuki–Varga
benchmark [suzuki2002stochastic](@cite).

```@example helium
using FewBodyECG
using FewBodyDB: db
using Plots

helium = Operators([1.0e15, 1.0, 1.0], [+2.0, -1.0, -1.0])
helium += "Kinetic"
helium += "Coulomb"

he_ref = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value
sol = solve(helium, SVM(basis = 100, candidates = 25, scale = 1.0))
(E₀ = sol.E₀, reference = he_ref, Δ = sol.E₀ - he_ref)
```

```@example helium
plot(sol, he_ref)
```

### Positronium negative ion

Ps⁻ = e⁺e⁻e⁻ has three particles of equal mass, so no coordinate can be
treated as fixed. It is bound below the Ps + e⁻ threshold at −0.25 Ha
[suzuki2002stochastic](@cite).

```@example psminus
using FewBodyECG
using FewBodyDB: db
using Plots

ops = Operators([1.0, 1.0, 1.0], [+1.0, -1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

ref = db(:Suzuki2003Jul, "Ps⁻", :energy, "¹Sᵉ").value
sol = solve(ops, SVM(basis = 100, candidates = 25, scale = 4.0))
(E₀ = sol.E₀, reference = ref, Δ = sol.E₀ - ref)
```

```@example psminus
plot(sol, ref)
```

### HD⁺ without the Born–Oppenheimer approximation

The proton, deuteron and electron are all treated as dynamical particles. The
reference is the non-Born–Oppenheimer ECG ground state of
[Bubin2005Jan](@cite).

```@example hdplus
using FewBodyECG
using FewBodyDB: db
using Plots

mₚ, m_d = 1836.15267343, 3670.48296788
ops = Operators([mₚ, m_d, 1.0], [+1.0, +1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

ref = db(:Bubin2005Jan, "HD⁺", :energy, (J = 0, v = 0)).value
sol = solve(ops, SVM(basis = 150, candidates = 25, scale = 1.0))
(E₀ = sol.E₀, reference = ref, Δ = sol.E₀ - ref)
```

```@example hdplus
plot(sol, ref)
```
