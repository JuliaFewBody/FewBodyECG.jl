# Theory

FewBodyECG uses explicitly correlated Gaussian basis functions to solve
few-body Schrödinger problems variationally.  A rank-0 basis function has the
form

```math
g(x) = \exp[-(x-s)^T A (x-s)],
```

where `x` are mass-weighted Jacobi coordinates, `A` is positive definite, and
`s` is a shift.  Rank-1 and Rank-2 Gaussians multiply this exponential by
linear or quadratic prefactors for non-s-wave channels.

## Matrix elements

Overlap, kinetic, Coulomb, and Gaussian-potential matrix elements are analytic
and dispatched on the Gaussian rank and operator type.  Once a basis is chosen,
the variational problem is the generalized eigenproblem

```math
H c = E S c.
```

The Rayleigh-Ritz principle makes every computed eigenvalue an upper bound to
the corresponding exact eigenvalue of the Hamiltonian.

## Jacobi coordinates

The center of mass is removed by a Jacobi transform.  Coordinates are
mass-weighted, so a physical pair distance is scaled by a reduced-mass factor.
This improves the kinetic-energy form but means multiscale systems can need
careful `scale` choices.

## Incremental whitened eigensolver

Stochastic growth repeatedly asks what would happen if one more Gaussian were
added.  A dense generalized eigensolve for every candidate would be wasteful.
FewBodyECG maintains the Cholesky-whitened overlap factor `S = R'R` and the
orthonormal eigendecomposition of the whitened Hamiltonian.

Adding a candidate produces an arrowhead eigenproblem.  The candidate can be
scored in `O(k^2)` by solving the secular equation for the target root, and
committed by updating the full arrowhead eigensystem.  Whitening keeps the
maintained eigenvectors orthogonal in ordinary Euclidean arithmetic, avoiding
drift in repeated non-orthogonal Gram-Schmidt updates.

## References

The stochastic and refinement algorithms follow Suzuki and Varga,
*Stochastic Variational Approach to Quantum-Mechanical Few-Body Problems*
(Lecture Notes in Physics m54, 1998).  The shifted ECG parameterization and
Hellmann-Feynman gradient route follow Fedorov,
*Explicitly Correlated Gaussians with Tensor Pre-factors* and related Few-Body
Systems work on analytic matrix elements and gradients.
