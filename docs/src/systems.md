# Building systems

The main system builder is `Operators`.  Give it particle masses and, when
convenient, charges:

```julia
ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"
```

Masses and charges are in atomic units.  A very large mass such as `1.0e15`
is the usual fixed-nucleus approximation.

## Operator shorthands

`ops += "Kinetic"` adds the kinetic term in Jacobi coordinates.  With charges
present, `ops += "Coulomb"` adds every pairwise Coulomb term using coefficients
`q_i q_j`.

Explicit pairs are also available:

```julia
ops = Operators([m1, m2, m3])
ops += "Kinetic"
ops += ("Coulomb", 1, 2, -1.0)
ops += ("Gaussian", 1, 3, -5.0, 1.0)
```

The Gaussian tuple represents `coeff * exp(-gamma * r_ij^2)`.

## Jacobi coordinates

FewBodyECG removes center-of-mass motion with mass-weighted Jacobi
coordinates.  The transform

```julia
J, U = jacobi_transform(masses)
```

maps particle coordinates to relative coordinates with a reduced-mass factor
`sqrt(mu)`.  The back-transform `U` is used to build pair weight vectors such
as `U' * [1, -1, 0]`.

`Λ(masses)` returns the kinetic matrix in those coordinates and is what
`KineticOperator(masses)` uses internally.  `coulomb_weights(ops)` returns the
Jacobi-frame pair weights already present in an `Operators` builder.

## Scale

Stochastic and variational methods accept `scale`.  `scale = :auto` uses
`default_scale(masses)`, based on the lightest finite particle mass.  Pass an
explicit value when the system has multiple length scales, such as muonic
molecules or molecular ions.

## Manual bases

The high-level solvers currently sample `Rank0Gaussian` bases.  Rank-1 and
Rank-2 prefactors are available through the matrix layer:

```julia
basis = BasisSet([Rank1Gaussian([alpha;;], [1.0], [0.0]) for alpha in alphas])
H = build_hamiltonian_matrix(basis, ops)
S = build_overlap_matrix(basis)
E, C = solve_generalized_eigenproblem(H, S)
```

Use this path for p-wave and d-wave hydrogen examples or custom angular
prefactors.
