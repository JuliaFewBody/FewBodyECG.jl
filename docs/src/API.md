# API

## Basis functions

- `Rank0Gaussian`, `Rank1Gaussian`, `Rank2Gaussian`
- `SpinState`, `SpinGaussian`, `BasisSet`

`Rank0Gaussian(A, s)` takes a positive-definite correlation matrix and an `N x 3` shift supervector. `Rank1Gaussian` and `Rank2Gaussian` implement the zero-shift tensor pre-factors from Fedorov et al. (2024); shifted rank-1/rank-2 matrix elements are rejected rather than extrapolated beyond those derivations.

## Operators and matrix assembly

- `KineticOperator`, `CoulombOperator`, `GaussianPotential`, `OscillatorPotential`
- `ManyBodyGaussianPotential`, `GaussianTensorPotential`, `GaussianSpinOrbitPotential`
- `build_overlap_matrix`, `build_hamiltonian_matrix`, `solve_generalized_eigenproblem`

`GaussianTensorPotential` and `GaussianSpinOrbitPotential` use spin-site indices separately from their coordinate selector `w`, so they remain meaningful after a Jacobi coordinate transformation.

## Solver utilities

- `solve_ECG`, `SolverResults`, `convergence`
- `ψ₀`, `correlation_function`, `plot_correlation`

`plot_correlation` receives the plotting function as its first argument. This keeps the package independent of a plotting backend:

```julia
using Plots
plot_correlation(plot, result; xlabel = "r", ylabel = "r²|ψ(r)|²")
```
