# Choosing a solver

All algorithms use the same entry point:

```julia
sol = solve(ops, SVM(basis = 50))
```

Problem-level options such as `state`, `tol`, `window`, `init`, and `verbose`
belong to `solve`.  Algorithm options belong to the method structs.

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
| `scale` | Gaussian length scale or `:auto` |
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
| `scale` | replacement length scale |
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
| `scale` | cold-start length scale; omit with `init` |
| `optimizer` | OptimKit gradient algorithm and its settings |

## DynamicGVM

`DynamicGVM` adds one function at a time, then jointly optimizes the current
basis after each addition.

| option | meaning |
|---|---|
| `basis` | final number of functions, including a warm-start basis |
| `candidates` | candidates per growth step |
| `scale` | candidate length scale |
| `optimizer` | OptimKit gradient algorithm, reused at every growth step |

Pass `verbose = true` to `solve` for one FewBodyECG progress message per
accepted optimizer iteration (and per stochastic iteration or refinement
sweep). OptimKit's own output remains controlled by the optimizer's
`verbosity` setting.

## Pipelines

Use `→` to warm-start methods left to right:

```julia
sol = solve(ops, SVM(40) → Refine(2) → GVM())
```

Single-scale stochastic sampling can saturate at the wrong energy on
multiscale systems.  Gradient methods move Gaussians where fixed-scale
sampling cannot reach.  The recommended default workflow is
`SVM → Refine → GVM`, with explicit stochastic `scale` values for hard systems.

## Cost

| method | rough cost |
|---|---|
| `SVM` | `O(k^2)` per candidate with the incremental eigensolver |
| `Refine` | up to `O(k^4)` per sweep in the current implementation |
| `GVM` | `O(iter * n_param * k^3)` engine cost |
| `DynamicGVM` | repeated gradient solves over growing `k` |
