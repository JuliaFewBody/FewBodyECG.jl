# Choosing a solver

## Which solver should I use?

| method | use it for |
|---|---|
| [`SVM`](@ref) | the starting point: grows a basis by stochastic selection; fast and robust |
| [`Refine`](@ref) | improving an existing basis by replacing its functions one at a time |
| [`GVM`](@ref) | the most accuracy per basis function, by gradient optimization of every Gaussian; slower |
| [`DynamicGVM`](@ref) | small, accurate bases: adds one function at a time and optimizes after each |

For a new system, start with `SVM` alone and a few values of `scale` (see
[Choosing `scale`](@ref)). Once the energy behaves, chain the methods with `→`
(`\to<tab>`); each stage starts from the previous stage's basis:

```@example solvers
using FewBodyECG

ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"

sol = solve(ops, SVM(20; scale = 1.0) → Refine(2; scale = 1.0) → GVM())
```

Single-scale stochastic sampling can level off at too high an energy for
systems with very different length scales. The gradient methods can move
Gaussians to widths that sampling at a fixed `scale` rarely draws, which is
why the pipeline ends with `GVM`.

## Common interface

All algorithms use the same entry point:

```julia
sol = solve(ops, SVM(basis = 50))
```

Problem-level options such as `state`, `tol`, `window`, `init`, and `verbose`
belong to `solve`.  Algorithm options belong to the method structs.  `tol` is
an absolute energy in the system's units (see [Units](@ref)).

Solver methods have two families: `SVM` and `Refine` are
`StochasticMethod`s; `GVM` and `DynamicGVM` are `GradientMethod`s.  Both
families share the `SolverMethod` interface and compose in pipelines.

## SVM

`SVM` is stochastic basis growth.  It draws candidates, scores each with the
incremental whitened eigensolver, and commits the best admissible function.

| option | meaning |
|---|---|
| `basis` | number of growth steps |
| `candidates` | candidates scored per step |
| `scale` | sampling length: `:adaptive` (default), `:auto`, or a number; see [Choosing `scale`](@ref) |
| `widths` | `:linear` (default) or `:wide`; see [Wide width sampling](@ref) |
| `sampler` | quasi-random sampler |
| `indep_tol` | Cholesky-residual independence cutoff |

`candidates = 1` is accept-first stochastic growth.  Larger values are more
expensive but usually produce better bases.

## Refine

`Refine` revisits existing basis slots and tries replacements.  It requires an
initial basis, either through `init = sol` or a pipeline.

| option | meaning |
|---|---|
| `sweeps` | cyclic refinement passes |
| `candidates` | replacements tried per slot |
| `scale` | replacement length scale; `:adaptive` (default) measures the existing basis |
| `widths` | `:linear` (default) or `:wide` |
| `indep_tol` | independence cutoff |

## GVM

`GVM` (gradient variational method) jointly optimizes all Gaussian parameters
with ForwardDiff/Hellmann-Feynman gradients. The optimizer is an OptimKit
algorithm object, so its iteration limit, gradient tolerance, line search, and
verbosity stay together:

```julia
using OptimKit: GradientDescent

gradient_method = GradientDescent(
    maxiter = 100,
    gradtol = 1e-6,
    verbosity = 1,
)
sol = solve(ops, GVM(30; scale = 1.0, optimizer = gradient_method))
```

`GradientDescent`, `ConjugateGradient`, and `LBFGS` are supported. `LBFGS` is
the default.

A warm start already has a basis, so the size is inferred and `scale` must be
omitted:

```julia
sol = solve(ops, GVM(); init = seed)
```

| option | meaning |
|---|---|
| `basis` | number of functions for a cold start; optional with `init` |
| `scale` | cold-start length scale, `:adaptive` when omitted; omit with `init` |
| `widths` | cold-start width sampling, `:linear` or `:wide`; omit with `init` |
| `optimizer` | OptimKit gradient algorithm and its settings |

## DynamicGVM

`DynamicGVM` adds one function at a time, then jointly optimizes the current
basis after each addition.

| option | meaning |
|---|---|
| `basis` | final number of functions, including a warm-start basis |
| `candidates` | candidates per growth step |
| `scale` | candidate length scale, `:adaptive` by default |
| `widths` | `:linear` (default) or `:wide` |
| `optimizer` | OptimKit gradient algorithm, reused at every growth step |

Pass `verbose = true` to `solve` for one FewBodyECG progress message per
accepted optimizer iteration (and per stochastic iteration or refinement
sweep). OptimKit's own output remains controlled by the optimizer's
`verbosity` setting.

## Pipelines

Use `→` to warm-start methods left to right:

```julia
sol = solve(ops, SVM(40; scale = 1.0) → Refine(2; scale = 1.0) → GVM())
```

The printed `Solution` lists every stage with its final energy, and
`energy_history(sol, i)` returns the history of stage `i`.

## Cost

| method | rough cost |
|---|---|
| `SVM` | `O(k^2)` per candidate with the incremental eigensolver |
| `Refine` | up to `O(k^4)` per sweep in the current implementation |
| `GVM` | `O(iter * n_param * k^3)` engine cost |
| `DynamicGVM` | repeated gradient solves over growing `k` |
