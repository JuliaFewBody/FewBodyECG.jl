# API Reference

## System building

```@docs
Operators
Operator
KineticOperator
CoulombOperator
GaussianOperator
OscillatorOperator
ManyBodyGaussianOperator
NumericalPotential
numerical
GaussianTensorOperator
GaussianSpinOrbitOperator
SpinProjection
SpinState
SpinGaussian
GaussianBase
Rank0Gaussian
Rank1Gaussian
Rank2Gaussian
BasisSet
```

## Solving

```@docs
solve
SolverMethod
StochasticMethod
GradientMethod
SVM
Refine
GVM
DynamicGVM
Pipeline
→
```

## Results

```@docs
Solution
ConvergenceReport
StageResult
converged
energy
energy_history
convergence
wavefunction
Wavefunction
radial_profile
```

## Matrix-level layer (public, not exported)

These names are part of the supported API but are not brought into scope by
`using FewBodyECG`. Call them as `FewBodyECG.name`, or import them explicitly:

```julia
using FewBodyECG: build_hamiltonian_matrix, build_overlap_matrix,
    solve_generalized_eigenproblem, up, down
```

```@docs
FewBodyECG.build_hamiltonian_matrix
FewBodyECG.build_overlap_matrix
FewBodyECG.solve_generalized_eigenproblem
FewBodyECG.Λ
FewBodyECG.jacobi_transform
FewBodyECG.default_scale
FewBodyECG.coulomb_weights
FewBodyECG.up
FewBodyECG.down
```
