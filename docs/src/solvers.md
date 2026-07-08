# Choosing a solver

All algorithms use the same entry point:

```julia
sol = solve(ops, SVM(basis = 50))
```

Problem-level options such as `state`, `tol`, `window`, `init`, and `verbose`
belong to `solve`.  Algorithm options belong to the method structs.

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

## Variational

`Variational` jointly optimizes all Gaussian parameters with LBFGS and
ForwardDiff/Hellmann-Feynman gradients.

| option | meaning |
|---|---|
| `basis` | number of functions |
| `scale` | cold-start length scale |
| `maxiter` | LBFGS iteration cap |
| `gtol` | gradient tolerance |
| `gradient` | gradient backend, currently `AutoDiff()` |

## GrowVariational

`GrowVariational` adds one function at a time, then jointly optimizes the
current basis after each addition.

| option | meaning |
|---|---|
| `basis` | final number of functions |
| `candidates` | candidates per growth step |
| `scale` | candidate length scale |
| `maxiter_step` | LBFGS iterations per step |
| `gtol` | gradient tolerance |

## Pipelines

Use `→` to warm-start methods left to right:

```julia
sol = solve(ops, SVM(40) → Refine(2) → Variational(40))
```

Single-scale stochastic sampling can saturate at the wrong energy on
multiscale systems.  Gradient methods move Gaussians where fixed-scale
sampling cannot reach.  The recommended default workflow is
`SVM → Refine → Variational`, with explicit `scale` values for hard systems.

## Cost

| method | rough cost |
|---|---|
| `SVM` | `O(k^2)` per candidate with the incremental eigensolver |
| `Refine` | up to `O(k^4)` per sweep in the current implementation |
| `Variational` | `O(iter * n_param * k^3)` engine cost |
| `GrowVariational` | repeated variational solves over growing `k` |
