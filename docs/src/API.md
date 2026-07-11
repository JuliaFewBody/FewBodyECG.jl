# API Reference

## System building

```@docs
Operators
coulomb_weights
Operator
KineticOperator
CoulombOperator
GaussianOperator
OscillatorOperator
ManyBodyGaussianOperator
GaussianTensorOperator
GaussianSpinOrbitOperator
SpinProjection
up
down
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
SVM
Refine
Variational
GrowVariational
Pipeline
→
AutoDiff
```

## Results

```@docs
Solution
ConvergenceReport
StageResult
converged
energies
convergence
wavefunction
Wavefunction
radial_profile
```

## Power-user layer

```@docs
build_hamiltonian_matrix
build_overlap_matrix
solve_generalized_eigenproblem
Λ
jacobi_transform
default_scale
```
