# Spin-Aware Analytic Matrix Elements Design

## Goal

Extend FewBodyECG with the concise, analytic matrix elements needed for shifted correlated Gaussians in the presence of Gaussian central, tensor, and spin-orbit interactions. Keep the implementation consistent with Fedorov's shifted-Gaussian notation and the spin-product basis, while separating solver utilities from core physics.

No commits are part of this work.

## Reference Formulation

The primary source is Fedorov, *Analytic matrix elements with shifted correlated Gaussians*, Eqs. (8), (14), (15), (18), (38)-(40), (61), and (65), arXiv:1702.06784. The rank-0, rank-1, and rank-2 pre-factor conventions follow Fedorov et al., *Explicitly Correlated Gaussians with Tensor Pre-factors* (2024), Eqs. (1)-(41).

Coordinates are a column of three-vectors, so a shift is an `N x 3` supervector. For a shift matrix `s`, the source's compact product `sᵀ R s` means `tr(s' * R * s)`; the relative shift in a central potential is the three-vector `q = 1/2 * w' * R * (s_bra + s_ket)`.

Spin states are direct products of spin-1/2 projection states. The spin part factors from the orbital Gaussian, and `S_x`, `S_y`, and `S_z` use the Pauli-spin matrix elements documented in the correlated-Gaussian spin reference.

## Scope

### Included operators

- Gaussian central potential: `c * exp(-gamma * |wᵀr|^2)` (primary source Eq. 15).
- Quadratic/oscillator potential: `c * |wᵀr|^2` (Eq. 18).
- Gaussian many-body potential: `c * exp(-rᵀ W r)` (Eq. 65).
- Gaussian tensor interaction between spin sites `i` and `j`, with optional subtraction of the `r^2 * S_i dot S_j / 3` central component (Eqs. 33-40).
- Gaussian spin-orbit interaction between spin sites `i` and `j` (Eqs. 55-62).

The two-body operators carry both `w`, which selects the relative coordinate in the current coordinate system, and spin-site indices `i` and `j`. This deliberately does not assume that coordinate indices and particle-spin indices are identical after a Jacobi transformation.

### Deliberately deferred

- Yukawa, screened-Yukawa, exponential, and derivative-generated radial families. Their closed forms and the tensor/spin-orbit `q` derivatives are long and require a distinct numerical-stability design. Gaussian expansions provide the paper-recommended, analytic route for this release.
- Matrix elements between shifted rank-1 or rank-2 pre-factor Gaussians. The supplied 2024 reference derives rank-1 and rank-2 elements as zero-shift Taylor coefficients; the package will enforce that domain instead of retaining unverified scalar-shift formulas.
- Permutation symmetry, coupled-spin bases, and arbitrary particle spin. The direct-product spin-1/2 basis is the smallest representation that makes the requested tensor and spin-orbit operators physical.

## Types and Public Surface

`Rank0Gaussian` will store `s` as an `N x 3` real matrix. A vector constructor remains as a compatibility constructor for a fixed-axis shift, but internal sampling and all new tests use the explicit supervector form.

`Rank1Gaussian` and `Rank2Gaussian` retain their real `N`-component polarization vectors and use zero shifts. Their existing overlap, kinetic, and Coulomb expressions are retained only after being checked against the 2024 reference. Their constructors and matrix-element methods reject nonzero shifts with clear `ArgumentError`s.

`SpinState` records one `up` or `down` projection per spin site. `SpinGaussian{G}` wraps an orbital `G <: GaussianBase` and a `SpinState`. Orbital-only operators act diagonally in spin; spin-dependent operators act on `SpinGaussian`s and return complex values where required by `S_y` and angular momentum.

New `FewBodyHamiltonians.PotentialTerm` subtypes will describe the five included operator families. Each contains only its physical coefficient and source-formula parameters; no operator registry or factory is introduced.

## Matrix-Element Structure

`matrix_elements.jl` will be organized around a small set of private helpers:

- combined Gaussian data: `B = A_bra + A_ket`, `R = inv(B)`, `v = s_bra + s_ket`, and the overlap;
- rank-one update data for Gaussian pair potentials;
- spin-site matrix elements for `S_a`, `S_i dot S_j`, and the product factors needed by the tensor and spin-orbit contractions.

Methods stay as direct multiple dispatch on `(bra, ket, operator)`. The Gaussian central, tensor, and spin-orbit implementations use the paper's closed forms rather than finite differences or numerical quadrature. The traceless tensor form is obtained by subtracting precisely one third of the central spin-spin contribution.

## Solver and Utility Boundaries

The matrix builders will infer their element type from the first analytic matrix element, retain complex entries, and fill the Hermitian counterpart by conjugation. The generalized eigensolver will use Hermitian matrices and retain complex eigenvectors; it will no longer force `real.(eigenvectors)`.

`SolverResults` becomes parametric over its basis, operator, and eigenvector element types. The solver keeps rank-0 orbital generation as its initial optimization path; spin-aware bases can be assembled explicitly and passed to the matrix builders.

The single `src/utils.jl` file moves to focused files under `src/utils/`:

- `solver_results.jl` for `SolverResults`, energies, and wavefunction evaluation;
- `observables.jl` for correlation-profile data;
- `plotting.jl` for a plotting adapter that accepts a caller-supplied plotting function.

No `Plots` dependency is added. Examples and documentation continue to own their plotting backend; the package exports data and an optional adapter rather than referencing an undeclared global `plot`.

## Tests and Verification

Tests are written before each implementation change and cover:

- paper-equation checks for shifted overlap, kinetic, Coulomb, Gaussian central, oscillator, and many-body Gaussian values;
- rank-one-update agreement with the direct updated overlap;
- rank-1/rank-2 zero-shift values from the 2024 source and rejection of shifted inputs;
- spin selection rules and Pauli matrix elements, including the imaginary `S_y` transition;
- tensor central-subtraction, spin-orbit Hermiticity, and complex Hamiltonian/overlap assembly;
- public utility paths after the move to `src/utils/`.

The verification gate runs targeted test files, the complete `Pkg.test()` suite, and `runic --check src test` after applying Runic in place to changed Julia files.

## Compatibility and Errors

This is a physical-model correction: callers that supplied rank-1 or rank-2 shifts must use rank-0 shifted Gaussians or wait for a separately derived shifted pre-factor extension. Shape mismatches, non-three-component shift supervectors, invalid spin indices, and negative Gaussian ranges raise `ArgumentError`s before any matrix calculation. Empty bases are rejected with a clear error rather than relying on an uninitialized matrix element type.
