# FewBodyECG.jl

[![Julia 1.11+](https://badgen.net/static/Julia/1.11%2B/007ec6?icon=https%3A%2F%2Fraw.githubusercontent.com%2FJuliaLang%2Fjulia-logo-graphics%2Fmaster%2Fimages%2Fjulia-dots.svg)](https://julialang.org/downloads/)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaFewBody.github.io/FewBodyECG.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaFewBody.github.io/FewBodyECG.jl/dev/)
[![Citation](https://img.shields.io/badge/citation-CFF-7A8288)](CITATION.cff)
[![license](https://img.shields.io/github/license/JuliaFewBody/FewBodyECG.jl?label=license)](LICENSE)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![CI](https://github.com/JuliaFewBody/FewBodyECG.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/FewBodyECG.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Aqua](https://img.shields.io/github/actions/workflow/status/JuliaFewBody/FewBodyECG.jl/Aqua.yml?branch=main&event=push&label=Aqua&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA3OS4zNzUgNTguNjc0Ij48ZyBzdHJva2U9IiNmNWY1ZjUiIHN0cm9rZS13aWR0aD0iNC44ODciPjxjaXJjbGUgZmlsbD0iIzM4OTgyNiIgY3g9IjM5LjY4OCIgY3k9IjIzLjk5MiIgcj0iMjEuNTQ5Ii8%2BPGNpcmNsZSBmaWxsPSIjY2IzYzMzIiBjeD0iMTcuNDQzIiBjeT0iNDEuMjMxIiByPSIxNSIvPjxjaXJjbGUgZmlsbD0iIzk1NThiMiIgY3g9IjYxLjkzMiIgY3k9IjQxLjIzMSIgcj0iMTUiLz48L2c%2BPC9zdmc%2B)](https://github.com/JuliaFewBody/FewBodyECG.jl/actions/workflows/Aqua.yml?query=branch%3Amain)
[![coverage](https://codecov.io/gh/JuliaFewBody/FewBodyECG.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaFewBody/FewBodyECG.jl)

FewBodyECG.jl solves quantum few-body bound-state problems with explicitly
correlated Gaussian variational bases.  Write a Hamiltonian as a sum of
particle-indexed terms, choose a solver method with `solve`, and inspect a `Solution`
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

ops = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])

sol = solve(ops, SVM(basis = 25, candidates = 20))
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
| `SVM(basis = n)` | the starting point: grows a basis of `n` Gaussians by stochastic selection |
| `Refine(sweeps)` | improving an existing basis by replacing its functions one at a time |
| `GVM()` | the most accuracy per basis function, by gradient optimization of every Gaussian; slower |
| `DynamicGVM(basis = n)` | small, accurate bases: adds one function at a time and optimizes after each |

Methods chain with `→` (type `\to<tab>`), each stage starting from the
previous stage's basis. A good default for a new system is

```julia
sol = solve(ops, SVM(40) → Refine(2) → GVM())
```

The stochastic methods sample Gaussians at a length `scale`, roughly the
typical distance between the particles. By default (`scale = :adaptive`) it is
measured with a short pilot run; pass a number to set it by hand. The
documentation explains both, and how to work in units other than atomic units
(`Kinetic(masses; units = :nuclear)`).

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
