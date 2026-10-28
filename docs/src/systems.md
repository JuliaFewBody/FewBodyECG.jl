# Building systems

A Hamiltonian is built with [`Operators`](@ref): give it the particle masses
(and, optionally, charges), then add terms one at a time.

```@example systems
using FewBodyECG

ops = Operators([1.0e15, 1.0, 1.0], [+2.0, -1.0, -1.0])   # helium: nucleus + 2 electrons
ops += "Kinetic"
ops += "Coulomb"
```

Particles are referred to by their position in the mass vector. The
center-of-mass motion is removed automatically: internally every term is
written in Jacobi coordinates, so you never handle those coordinates yourself
unless you build operators by hand (see [Raw operator types](@ref)).

A very large mass such as `1.0e15` is the usual way to model an infinitely
heavy particle (a fixed nucleus).

## Constructors

| call | what it enables |
|---|---|
| `Operators(masses)` | every term form below except the all-pairs `"Coulomb"` |
| `Operators(masses, charges)` | additionally `ops += "Coulomb"` for all pairs |
| `Operators()` | only pre-built operator objects |

All three take a `units` keyword; see [Units](@ref).

## Adding terms

`ops += term` builds a new `Operators` containing the extra term and rebinds
`ops` to it; `push!(ops, term)` adds the term to the existing object in place.
Any earlier reference to `ops` keeps its old term list after `+=`, but sees the
new term after `push!`.

Every way of adding a term, with ``r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|``:

| term | adds | requires |
|---|---|---|
| `"Kinetic"` | ``\sum_i -\nabla_i^2 / 2m_i``, center of mass removed | masses |
| `"Coulomb"` | ``q_i q_j / r_{ij}`` for every pair ``i < j`` | masses, charges, atomic units |
| `("Coulomb", i, j, c)` | ``c / r_{ij}`` | masses |
| `("Gaussian", i, j, V₀, γ)` | ``V_0\, e^{-\gamma r_{ij}^2}`` | masses |
| `("Oscillator", i, j, k)` | ``k\, r_{ij}^2`` | masses |
| `(f, numerical, i, j)` | ``f(r_{ij})`` for any callable `f(r)` | masses |
| an operator object | itself | nothing |

The order of the numbers in a tuple matters: `("Gaussian", i, j, V₀, γ)` takes
the strength first and the range parameter second. A negative strength is
attractive.

```@example systems
nucleon = Operators([1.0, 1.0])
nucleon += "Kinetic"
nucleon += ("Gaussian", 1, 2, 200.0, 1.487)    # repulsive core
nucleon += ("Gaussian", 1, 2, -178.0, 0.639)   # attraction
nucleon += ("Oscillator", 1, 2, 0.1)
nucleon += (r -> -exp(-r) / r, numerical, 1, 2)  # Yukawa, by quadrature
```

`numerical` terms reduce the matrix element analytically to a one-dimensional
radial integral, evaluated with adaptive quadrature (QuadGK). The callable
receives the nonnegative pair distance. They are slower than the analytic
terms; prefer a sum of `"Gaussian"` terms when the potential can be fit that
way.

## Raw operator types

The shorthand forms above construct these operator objects, which can also be
built directly and added with `ops += op`:

| type | potential |
|---|---|
| [`KineticOperator`](@ref)`(masses)` | kinetic energy |
| [`CoulombOperator`](@ref)`(c, w)` | ``c / \lvert w^T \mathbf{x} \rvert`` |
| [`GaussianOperator`](@ref)`(c, γ, w)` | ``c\, e^{-\gamma (w^T \mathbf{x})^2}`` |
| [`OscillatorOperator`](@ref)`(c, w)` | ``c\, (w^T \mathbf{x})^2`` |
| [`NumericalPotential`](@ref)`(f, w; rtol, atol, maxevals)` | ``f(\lvert w^T \mathbf{x} \rvert)`` |
| [`ManyBodyGaussianOperator`](@ref)`(c, W)` | ``c\, e^{-\mathbf{x}^T W \mathbf{x}}`` (all coordinates at once) |
| [`GaussianTensorOperator`](@ref)`(c, γ, w, i, j)` | Gaussian-range tensor force between spins `i`, `j` |
| [`GaussianSpinOrbitOperator`](@ref)`(c, γ, w, i, j)` | Gaussian-range spin-orbit force |

Here ``\mathbf{x}`` are the Jacobi coordinates, and the weight vector `w`
picks out one pair distance: ``w^T \mathbf{x} = \mathbf{r}_i - \mathbf{r}_j``.
[`FewBodyECG.coulomb_weights`](@ref) returns the weight vectors of the Coulomb
terms already added; for any pair, build `w` from
[`FewBodyECG.jacobi_transform`](@ref):

```@example systems
masses = [1.0e15, 1.0, 1.0]
_, U = FewBodyECG.jacobi_transform(masses)
w₂₃ = U' * [0.0, 1.0, -1.0]                  # +1 at i = 2, -1 at j = 3

raw = Operators(masses)
raw += "Kinetic"
raw += GaussianOperator(-4.0, 0.5, w₂₃)      # same as ("Gaussian", 2, 3, -4.0, 0.5)
```

[`ManyBodyGaussianOperator`](@ref) has no shorthand because its exponent
matrix `W` acts on all Jacobi coordinates; the three-body example in
`examples/three_body_gaussian.jl` uses one as a repulsive regulator.

## Spin

Spin-dependent forces act on basis functions that carry a spin state: a
[`SpinGaussian`](@ref) pairs a [`Rank0Gaussian`](@ref) with a
[`SpinState`](@ref), one [`SpinProjection`](@ref) (`FewBodyECG.up` or
`FewBodyECG.down`) per particle. The tensor and spin-orbit operators then give
complex Hermitian Hamiltonian matrices.

`solve` builds its bases from spinless `Rank0Gaussian`s, so spin-dependent
problems use the matrix-level functions `FewBodyECG.build_hamiltonian_matrix`,
`FewBodyECG.build_overlap_matrix` and `FewBodyECG.solve_generalized_eigenproblem`
on a basis you construct. The [spin-orbit example](generated/spin_orbit.md)
walks through this.

## Units

FewBodyECG works with ``\hbar = 1`` and the kinetic operator
``-\nabla^2 / 2m``. It converts nothing: the masses and potential parameters
you pass fix the units of the energies and lengths that come out. The `units`
keyword of `Operators` records which convention you used, so that printed
energies and plot axes carry the right label.

| `units` | energy | length | masses | Coulomb constant ``e^2/4\pi\varepsilon_0`` |
|---|---|---|---|---|
| `:atomic` (default) | Ha | bohr | multiples of ``m_e`` | 1 |
| `:nuclear` | MeV | fm | ``mc^2 / (\hbar c)^2`` | ≈ 1.44 MeV fm |
| `UnitSystem(energy, length)` | your label | your label | your convention | your convention |

In nuclear units, give each particle the mass ``mc^2/(\hbar c)^2`` with
``\hbar c = 197.327`` MeV fm, so that ``\hbar^2/2m`` becomes
``(\hbar c)^2/2mc^2`` (≈ 20.7 MeV fm² for a nucleon). Gaussian strengths are
then in MeV and range parameters in fm⁻²:

```@example systems
ħc = 197.3269804
mass(mc²) = mc² / ħc^2                       # MeV/c² → package mass

deuteron = Operators([mass(938.272), mass(939.565)]; units = :nuclear)
deuteron += "Kinetic"
deuteron += ("Gaussian", 1, 2, 200.0, 1.487)
deuteron += ("Gaussian", 1, 2, -178.0, 0.639)
sol = solve(deuteron, SVM(basis = 20))
```

The all-pairs `"Coulomb"` shorthand uses the couplings ``q_i q_j``, which is
only correct in atomic units, so it raises an error for any other unit system.
Add charged pairs explicitly instead, for example
`ops += ("Coulomb", 1, 2, 1.44 * q₁ * q₂)` in nuclear units.

For other conventions pass a [`UnitSystem`](@ref); for example, with
``\hbar = c = 1`` and masses in GeV, `units = UnitSystem("GeV", "GeV⁻¹")`.

The convergence tolerance `tol` of [`solve`](@ref) is an absolute energy in
the system's energy unit (default `1e-4`).

## Choosing `scale`

The stochastic methods ([`SVM`](@ref), [`Refine`](@ref), [`DynamicGVM`](@ref),
and cold-start [`GVM`](@ref)) draw trial Gaussians
``\exp(-\sum_{i<j} r_{ij}^2 / b_{ij}^2)`` with each pair width ``b_{ij}``
sampled between `0.02 * scale` and `5 * scale`. So `scale` is a length, in the
system's length unit, and should be about the typical distance between the
particles.

By default (`scale = :adaptive`) the package chooses it. A short pilot run
(30 functions, `widths = :wide`) estimates the wavefunction, and `scale` is set
to 0.7 times the largest root-mean-square pair distance
``\langle r_{ij}^2 \rangle^{1/2}``, computed exactly from the pilot's basis. The
pilot is re-centered on its own estimate for up to three rounds, and the size
is taken from the pilot with the lowest energy. On a warm start (`init`, or a
later stage of a pipeline) the size is measured on the existing basis instead,
with no pilot. The convergence report records the value chosen, for example

```
scale = :adaptive → 1.21 (0.7 × rms pair distance 1.73 bohr, from the lowest-energy of 2 pilot runs …)
```

At equal basis size, `:adaptive` comes close to a hand-tuned `scale` (energies
above the reference, in mHa, or keV for the nuclear systems):

| system | basis | hand-tuned `scale` | `:adaptive` | `:auto` |
|---|---|---|---|---|
| H | 20 | 0.0046 | 0.0075 | 0.0046 |
| He | 50 | 3.1 | 2.5 | 3.1 |
| H⁻ | 50 | 1.7 | 1.9 | 14 |
| Ps⁻ | 60 | 0.26 | 0.23 | 59 |
| tdμ | 150 | 4.6 | 6.8 | 1190 |
| Hooke's atom | 40 | 0.64 | 0.60 | 1.0 |
| deuteron | 30 | 0.23 | 0.30 | 0.19 |
| triton (Volkov) | 60 | −6.6 | 1.6 | −2.2 |

The triton reference, −8.46 MeV, is rounded, so the negative values mean those
runs went below the rounded number.

The factor 0.7 is a compromise: Coulomb systems do slightly better near 0.5,
smooth nuclear potentials near 1. For production runs, a hand-tuned `scale`
can still gain a little. Values that work well:

| system | typical size | hand-tuned `scale` |
|---|---|---|
| hydrogen, helium | ≈ 1 bohr | 1.0 |
| positronium | ≈ 2 bohr | 1.4 |
| H⁻, Ps⁻ (weakly bound, diffuse) | several bohr | 4.0 |
| muonic molecules (tdμ, ttμ) | ≈ 0.01 bohr | 0.01–0.03 |
| deuteron, triton | 2–4 fm | 3.0–3.5 |

If `scale` is too small, no trial function reaches the outer part of the
wavefunction and the energy levels off too high. If it is too large, most
trial functions are too wide to help and convergence is slow. When in doubt,
run two or three values and compare the energies.

### Wide width sampling

For a new system where the right `scale` is unclear, pass `widths = :wide` to
the method. The pair widths are then sampled log-uniformly between
`0.01 * scale` and `100 * scale`, so every decade of that range is tried
equally often:

```julia
sol = solve(ops, SVM(basis = 60, scale = 2.0, widths = :wide))
```

This makes the result much less sensitive to `scale`. In a comparison at equal
cost, positronium negative ion (Ps⁻, 60 functions) stayed within 0.4–1.1 mHa
of the reference for `scale` from 1 to 8 with `widths = :wide`, but ranged from
0.2 to 59 mHa with the default. The price is that draws are spread over four
decades, so with a well-chosen `scale` the default converges faster (for tdμ
with 150 functions: 4.6 mHa above the reference with the default at
`scale = 0.01`, 15 mHa with `widths = :wide`). `scale = :adaptive` uses
`widths = :wide` only for its pilot runs and then samples the default widths
at the measured size, which combines both.

`scale = :auto` uses [`FewBodyECG.default_scale`](@ref): ``1/\sqrt{m}`` for the
lightest particle with mass below `1e10`. It ignores the interactions, so it
can be far off (for Ps⁻ and tdμ in the table above); `:adaptive` starts its
pilot from it.
