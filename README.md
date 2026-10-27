# FewBodyECG.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaFewBody.github.io/FewBodyECG.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaFewBody.github.io/FewBodyECG.jl/dev/)
[![Build Status](https://github.com/JuliaFewBody/FewBodyECG.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/FewBodyECG.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaFewBody/FewBodyECG.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaFewBody/FewBodyECG.jl)

FewBodyECG.jl solves quantum few-body bound-state problems with explicitly
correlated Gaussian variational bases.  Build Hamiltonians with
`Operators`, choose a solver method with `solve`, and inspect a `Solution`
with convergence reports, stage histories, wavefunctions, and plotting
recipes.

## Install

```julia
import Pkg
Pkg.add("FewBodyECG")
```

## Quickstart

The hydrogen atom: a fixed proton (mass `1e15`, charge `+1`) and an electron
(mass `1`, charge `-1`), in atomic units.

```julia
using FewBodyECG
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0))
sol             # energy, convergence report, overlap conditioning

plot(sol, -0.5) # energy after each solver step, with the exact
                # ground-state energy -0.5 Ha drawn as a dashed reference line
```

`sol.E₀` is the ground-state energy, a variational upper bound on the exact
value. The second argument of `plot` is optional; pass any reference energy
you want to compare with.

## Which solver?

| method | use it for |
|---|---|
| `SVM(basis = n, scale = s)` | the starting point: grows a basis of `n` Gaussians by stochastic selection |
| `Refine(sweeps)` | improving an existing basis by replacing its functions one at a time |
| `GVM()` | the most accuracy per basis function, by gradient optimization of every Gaussian; slower |
| `DynamicGVM(basis = n)` | small, accurate bases: adds one function at a time and optimizes after each |

Methods chain with `→` (type `\to<tab>`), each stage starting from the
previous stage's basis. A good default for a new system is

```julia
sol = solve(ops, SVM(40; scale = 1.0) → Refine(2; scale = 1.0) → GVM())
```

`scale` is a length: roughly the typical distance between the particles, in
the system's length unit. The documentation explains how to choose it, and
how to work in units other than atomic units (`Operators(...; units = :nuclear)`).

Code written for FewBodyECG 1.x needs changes; `CHANGELOG.md` maps each 1.x
function to its replacement.

## Features

- Unified `solve` API with stochastic `SVM`/`Refine`, gradient-based
  `GVM`/`DynamicGVM`, and `→` pipelines.
- Honest `ConvergenceReport` values on every `Solution`.
- Incremental whitened eigensolver for stochastic basis growth.
- Rank-0 stochastic/gradient solvers plus Rank-1/Rank-2 manual matrix-layer
  support.
- RecipesBase plotting without depending on Plots at package load time.
