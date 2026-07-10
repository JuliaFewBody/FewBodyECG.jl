# Harmonic-oscillator benchmark

## Goal

Add a deterministic benchmark and runnable example for the three-dimensional,
two-particle relative harmonic oscillator. It validates the analytic
`OscillatorPotential` matrix element together with overlap construction,
kinetic energy, and the generalized eigensolver.

## Model

Use one relative coordinate `r` and atomic units:

```math
H = -\frac{1}{2}\nabla^2 + \frac{1}{2}r^2.
```

The rank-0 ECG basis represents the `ℓ = 0` sector. Its first three radial
levels therefore have exact energies

```math
E_{n_r} = 2n_r + \frac{3}{2}, \qquad n_r = 0, 1, 2,
```

namely `3/2`, `7/2`, and `11/2`.

## Implementation

- Add a deterministic test using a fixed, even-tempered `Rank0Gaussian`
  basis with zero shifts. Construct the Hamiltonian from
  `KineticOperator([1 / 2;;])` and `OscillatorPotential(1 / 2, [1])`, then test
  the three lowest generalized eigenvalues against the exact radial levels.
- Add `Examples/HarmonicOscillator.jl` using the same fixed basis. It prints a
  small exact-versus-computed energy table and plots the normalized radial
  probabilities `r^2 |ψ_{n_r}(r)|^2` for states one through three.
- Keep plotting in the example through `Plots`; the package core and its
  utilities remain free of a plotting dependency.

## Boundaries

This is a single relative coordinate in physical three-dimensional space, not
a spatially one-dimensional extension. It exercises the existing ECG physics
without introducing a spatial-dimension parameter or new solver machinery.

## Verification

- The regression test resolves all three energies within a documented,
  deterministic tolerance.
- The example runs headlessly and produces the radial-state figure without
  errors.
- Runic formatting and the complete test suite pass.
