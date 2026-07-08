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

```julia
using FewBodyECG
using Plots

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(basis = 25, candidates = 20, scale = 1.0))
sol

plot(sol, -0.5)
```

## v2.0

v2.0 is a breaking API release.  The old `solve_ECG*` entry points and
`SolverResults` utilities are replaced by `solve(ops, Method())`,
`Solution`, `energies`, `wavefunction`, and plotting recipes.  See
`CHANGELOG.md` for the migration table.

## Features

- Unified `solve` API with `SVM`, `Refine`, `Variational`, `GrowVariational`,
  and `→` pipelines.
- Honest `ConvergenceReport` values on every `Solution`.
- Incremental whitened eigensolver for stochastic basis growth.
- Rank-0 stochastic/gradient solvers plus Rank-1/Rank-2 manual matrix-layer
  support.
- RecipesBase plotting without depending on Plots at package load time.
