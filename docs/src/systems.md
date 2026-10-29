# Building systems

A Hamiltonian is written as a sum of terms that name the particles they act on.
The [`Kinetic`](@ref) term takes the particle masses and so defines the system;
every other term refers to particles by their position in that mass vector.

```@example systems
using FewBodyECG

H = Kinetic([1.0e15, 1.0, 1.0]) + Coulomb([+2.0, -1.0, -1.0])   # helium: nucleus + 2 electrons
```

The center-of-mass motion is removed automatically: `solve` rewrites every
term in Jacobi coordinates, so you never handle those coordinates yourself
unless you build operators by hand (see [Raw operator types](@ref)).

A very large mass such as `1.0e15` is the usual way to model an infinitely
heavy particle (a fixed nucleus).

## Terms

With ``r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|``:

| term | potential |
|---|---|
| [`Kinetic`](@ref)`(masses; units = :atomic)` | ``\sum_i -\nabla_i^2 / 2m_i``, center of mass removed |
| [`Coulomb`](@ref)`(i, j; coefficient = 1)` | `coefficient` ``/ r_{ij}`` |
| [`Coulomb`](@ref)`(charges)` | ``q_i q_j / r_{ij}`` for every pair ``i < j`` (atomic units only) |
| [`Gaussian`](@ref)`(i, j; coefficient = 1, exponent)` | `coefficient` ``\cdot e^{-\text{exponent}\, r_{ij}^2}`` |
| [`Oscillator`](@ref)`(i, j; coefficient = 1)` | `coefficient` ``\cdot r_{ij}^2`` |
| [`Custom`](@ref)`(f, i, j)` | ``f(r_{ij})`` for any callable `f(r)` |
| [`ManyBodyGaussian`](@ref)`(G; coefficient = 1)` | `coefficient` ``\cdot e^{-\mathbf{r}^T G\, \mathbf{r}}`` in all particle positions |
| [`Tensor`](@ref)`(i, j; coefficient = 1, exponent)` | Gaussian-range tensor force between the spins of `i` and `j` |
| [`SpinOrbit`](@ref)`(i, j; coefficient = 1, exponent)` | Gaussian-range spin-orbit force |

Coupling constants and range parameters are always keywords. `coefficient`
defaults to 1, so a coupling can also be written as a multiplier; `exponent`
has no default. A negative coefficient is attractive.

```@example systems
nucleon = Kinetic([1.0, 1.0]) +
    200.0 * Gaussian(1, 2; exponent = 1.487) -     # repulsive core
    178.0 * Gaussian(1, 2; exponent = 0.639) +     # attraction
    0.1 * Oscillator(1, 2) +
    Custom(r -> -exp(-r) / r, 1, 2)                # Yukawa, by quadrature
```

`Custom` terms reduce the matrix element analytically to a one-dimensional
radial integral, evaluated with adaptive quadrature (QuadGK); `rtol`, `atol`
and `maxevals` keywords set its tolerances. The callable receives the
nonnegative pair distance. They are slower than the analytic terms; prefer a
sum of Gaussians when the potential can be fit that way.

## Combining terms

Terms combine with `+`, `-`, multiplication and division by a real number, and
`sum` over a generator. The result is a [`Hamiltonian`](@ref), which keeps the
terms in the order they were written:

```@example systems
q = [+2.0, -1.0, -1.0]
H = Kinetic([1.0e15, 1.0, 1.0]) +
    sum(Coulomb(i, j; coefficient = q[i] * q[j]) for i in 1:3 for j in (i + 1):3)
```

A sum without a `Kinetic` term is an ordinary value, so a potential can be
built once and reused; a Hamiltonian passed to [`solve`](@ref) needs exactly
one `Kinetic` term. Since a Hamiltonian is just a value, one written as a
function of the masses serves several isotopes:

```@example systems
muonic(m₁, m₂) = Kinetic([m₁, m₂, 206.768]) + Coulomb([+1.0, +1.0, -1.0])
H_ddμ, H_dtμ = muonic(3670.481, 3670.481), muonic(3670.481, 5496.918)
nothing # hide
```

Scaling a sum scales every term in it. `Kinetic` takes no scalar factor;
change the masses instead.

### Products of Gaussians

The product of Gaussian terms is a Gaussian in several pair distances,
``e^{-\sum_k \gamma_k r_k^2}``, which is how many-body forces are written. A
symmetric three-body force:

```@example systems
γ₃ = 0.5
three_body = 2.0 * Gaussian(1, 2; exponent = γ₃) * Gaussian(1, 3; exponent = γ₃) * Gaussian(2, 3; exponent = γ₃)
nothing # hide
```

In general such a term is a [`ManyBodyGaussian`](@ref)
``e^{-\mathbf{r}^T G\, \mathbf{r}}``, with ``\mathbf{r}`` the particle
positions and ``G`` a symmetric matrix whose rows sum to zero (so that only
relative positions enter); it can also be given directly. No other product of
terms has an analytic matrix element: multiplying, say, a `Coulomb` and a
`Gaussian` raises an error, and such a pair potential is written as one
`Custom` term instead.

## Raw operator types

`solve` rewrites each term as one of these Jacobi-frame operators. They can
also be built directly, and a vector of them passed to [`solve`](@ref) or
`FewBodyECG.build_hamiltonian_matrix`:

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
terms of a Hamiltonian; for any pair, build `w` from
[`FewBodyECG.jacobi_transform`](@ref):

```@example systems
masses = [1.0e15, 1.0, 1.0]
_, U = FewBodyECG.jacobi_transform(masses)
w₂₃ = U' * [0.0, 1.0, -1.0]                  # +1 at i = 2, -1 at j = 3

raw = Operator[KineticOperator(masses), GaussianOperator(-4.0, 0.5, w₂₃)]   # -4 e^{-0.5 r₂₃²}
nothing # hide
```

Raw operators and physical terms do not mix in one sum, because a raw operator
is tied to the Jacobi coordinates of particular masses.

## Spin

Spin-dependent forces act on basis functions that carry a spin state: a
[`SpinGaussian`](@ref) pairs a [`Rank0Gaussian`](@ref) with a
[`SpinState`](@ref), one [`SpinProjection`](@ref) (`FewBodyECG.up` or
`FewBodyECG.down`) per particle. The [`Tensor`](@ref) and [`SpinOrbit`](@ref)
terms then give complex Hermitian Hamiltonian matrices.

`solve` builds its bases from spinless `Rank0Gaussian`s and rejects
spin-dependent terms, so these problems use the matrix-level functions
`FewBodyECG.build_hamiltonian_matrix(basis, H)`,
`FewBodyECG.build_overlap_matrix` and `FewBodyECG.solve_generalized_eigenproblem`
on a basis you construct. The [spin-orbit example](generated/spin_orbit.md)
walks through this.

## Units

FewBodyECG works with ``\hbar = 1`` and the kinetic operator
``-\nabla^2 / 2m``. It converts nothing: the masses and potential parameters
you pass fix the units of the energies and lengths that come out. The `units`
keyword of [`Kinetic`](@ref) records which convention you used, so that
printed energies and plot axes carry the right label.

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

deuteron = Kinetic([mass(938.272), mass(939.565)]; units = :nuclear) +
    200.0 * Gaussian(1, 2; exponent = 1.487) -
    178.0 * Gaussian(1, 2; exponent = 0.639)
sol = solve(deuteron, SVM(basis = 20))
```

`Coulomb(charges)` uses the couplings ``q_i q_j``, which is only correct in
atomic units, so solving it with any other unit system raises an error. Write
charged pairs explicitly instead, for example
`Coulomb(1, 2; coefficient = 1.44 * q₁ * q₂)` in nuclear units.

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
