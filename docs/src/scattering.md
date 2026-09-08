# Scattering from trapped spectra

An artificial harmonic trap turns the near-threshold continuum into discrete
levels that an ECG basis can represent. Following their dependence on trap
size gives access to low-energy scattering parameters
[fedorov2025scattering](@cite). The workflow first produces a
[`TrapSpectrum`](@ref), then fits it with [`fit_scattering_parameters`](@ref).

## Supported physics

This workflow describes three-dimensional, neutral, central, short-range,
elastic scattering in a single channel, with orbital angular momentum
``\ell=0,1,2`` (s, p, or d). Each fragment must have a scalar internal state,
with zero internal orbital angular momentum. Interfragment Coulomb,
coupled channels, noncentral or parity-changing interactions, and internal
fragment angular momentum are excluded. Coulomb interactions wholly inside
one fragment may be present.

Use a finite-range or sufficiently rapidly decaying interaction. For a user
supplied radial potential, `interaction_range` is an assertion about its
characteristic range; the package cannot certify its tail or channel physics.
Supply the elastic threshold explicitly, and an `inelastic_threshold` when
known. Fitted absolute energies must be strictly below that inelastic threshold.
For composite fragments, the elastic threshold is the energy of the separated
internal states, not an energy inferred from the combined Hamiltonian.

The channel basis has zero shifts and definite angular momentum. The p-wave
prefactor selects a component of the fragment separation, and the d-wave
prefactor selects a traceless component such as ``R_xR_y``. A generic quadratic
Gaussian prefactor is not necessarily a pure d wave. Rotational invariance
makes one magnetic component sufficient. The prefactor oscillator matrix
elements follow Teilmann's Eqs. 3.46 and 3.48, with one-particle limits in
Eqs. B.11 and B.12 [teilmann2023prefactor](@cite).

## Coordinate and trap convention

For fragments ``A`` and ``B``, define their masses and centers of mass by

```math
M_A=\sum_{i\in A}m_i,\qquad
\mathbf R_A=\frac{1}{M_A}\sum_{i\in A}m_i\mathbf r_i,
\qquad
\mathbf R=\mathbf R_A-\mathbf R_B,\qquad
\mu=\frac{M_AM_B}{M_A+M_B}.
```

The package expresses ``\mathbf R`` in Jacobi coordinates and adds a trap only
to this interfragment separation. Its length convention follows
Fedorov and Pedersen [fedorov2025scattering](@cite):

```math
b=\sqrt{\frac{\hbar}{2\mu\omega}},\qquad
V_{\mathrm{trap}}(R)=\frac12\mu\omega^2R^2
=\frac{\hbar^2}{8\mu b^4}R^2.
```

The common oscillator length ``\sqrt{\hbar/(\mu\omega)}`` is ``\sqrt{2}b``.
All numerical inputs use consistent units with ``\hbar=1``, so the added
potential is ``R^2/(8\mu b^4)`` and `frequencies` stores
``\omega=1/(2\mu b^2)``. `energies` contains absolute eigenenergies;
`relative_energies` subtracts the supplied elastic threshold.

## From levels to the effective-range expansion

Let ``E`` be threshold-relative energy and ``k^2=2\mu E`` in the package's
units. The neutral Busch–Englert–Rzażewski–Wilkens (BERW) relation used here is
the general partial-wave formula [zhang2024harmonic](@cite):

```math
K_\ell(E;b)=k^{2\ell+1}\cot\delta_\ell
=(-1)^{\ell+1}\frac{2^{\ell+1/2}}{b^{2\ell+1}}
\frac{\Gamma\!\left(\frac34+\frac\ell2-\frac{E}{2\omega}\right)}
{\Gamma\!\left(\frac14-\frac\ell2-\frac{E}{2\omega}\right)}.
```

For ``E<0``, the same real expression provides the standard continuation in
``k^2`` below threshold; no complex momentum is constructed. This does not
extend the package to absorptive complex potentials. Applying the trap
relation to a finite-range interaction requires checking the large-trap and
low-energy limits.

The effective-range expansion (ERE) convention is

```math
K_\ell(k)=-\frac{1}{a_\ell}+\frac{r_\ell}{2}k^2+v_\ell k^4+\cdots.
```

`ere_order=0`, `1`, or `2` fits through the constant, ``k^2``, or ``k^4``
term. `coefficients` holds the retained ``c_0,c_2,c_4`` in that order:

| Field | Definition | Dimension |
|:--|:--|:--|
| `a` | ``a_\ell=-1/c_0`` | ``L^{2\ell+1}`` |
| `r` | ``r_\ell=2c_2`` | ``L^{1-2\ell}`` |
| `shape` | ``v_\ell=c_4`` | ``L^{3-2\ell}`` |

Omitted `r` or `shape` terms are `nothing`. Thus `a` is a length for s waves,
a scattering volume for p waves, and has dimension ``L^5`` for d waves.
Fedorov–Pedersen use ``+1/a_{\mathrm{FP}}`` in their s-wave ERE, so
``a_0=-a_{\mathrm{FP}}``. Their dimensionless shape parameter ``P`` converts
as ``v_0=P r_0^3`` [fedorov2025scattering](@cite).

## Choosing traps and reading diagnostics

Choose several distinct trap lengths large compared with `interaction_range`.
`range_ratios` records ``b/\texttt{interaction\_range}``; there is no universal
cutoff and no automatic grid. Larger traps probe lower energies but demand a
basis spanning both the interaction scale and the extended trapped state.
Check several basis settings and overlapping trap windows before interpreting
an ERE coefficient as a free-space parameter.

[`trapped_spectrum`](@ref) uses one common basis per trap and minimizes the
equal-weight sum of the selected levels `1:n`. Use `SVM`, optionally followed
by `Refine` in a pipeline. Gradient solvers are unsupported in this workflow.

- `converged` marks a whole trap row usable only when every requested level
  exists and its recorded energy decrease is nonnegative and below `tol`.
  For SVM, `energy_changes` compares with `window` accepted additions earlier;
  for refinement it records the final sweep. Unconverged energies remain
  inspectable but the fit excludes them through its `used` mask. Missing
  levels or unavailable changes are `NaN`.
- `condition_numbers` measures the raw overlap matrix condition for each trap;
  `basis_sizes` reports retained basis sizes. Saturated energy changes can
  coexist with a basis error floor from candidate rejection. Allowing nearly
  dependent functions can instead destabilize eigenvalues. Neither a small
  change nor a larger basis alone certifies accuracy.
- `pole_distance` measures dimensionless distance to a numerator gamma pole.
  A used point exactly at such a pole is rejected; a denominator gamma pole
  gives ``K_\ell=0``. `K_energy_sensitivity` estimates
  ``|\partial K_\ell/\partial E\;\Delta E|`` from the recorded energy change,
  showing where energy errors may be amplified. Missing changes give `NaN`.
- The fit scales ``k^2`` and uses QR, without normal equations.
  `design_rank` and `design_condition` describe that scaled design;
  `residuals` shows the ERE mismatch at used points. `covariance`, when there
  are residual degrees of freedom, describes the polynomial coefficients in
  physical units. An exactly determined fit has `covariance=nothing`;
  undersampled or rank-deficient fits fail.
- `window_starts` and `window_parameters` repeat the fit after progressively
  removing the smallest trap lengths. Their columns are `a`, `r`, and `shape`;
  insufficient windows and omitted parameters are `NaN`. Look for stable
  large-trap values, and compare ERE orders. Higher coefficients can remain
  sensitive after `a` has stabilized.

The structured `warnings` flag pole proximity, energy sensitivity, marginal
convergence, ill conditioning, or unstable windows. These are numerical and
model diagnostics, not rigorous error bars or statistical uncertainties.
Small residuals, small covariance, or an empty warning list do not establish
physical validity.

## Two-body Gaussian example

Consider two unit masses with ``V(R)=-\exp(-R^2)`` and zero elastic threshold.
The Gaussian has a characteristic range of one length unit, although it has
no sharp cutoff. The following small, deterministic Halton calculation first
inspects the spectrum. The basis and independence cutoff are suitable for
this example; a different interaction needs its own convergence study.

```@example scattering
using FewBodyECG

ops = Operators([1.0, 1.0])
ops += "Kinetic"
ops += ("Gaussian", 1, 2, -1.0, 1.0)
channel = ScatteringChannel(
    [1], [2]; ℓ = 0, threshold = 0.0, interaction_range = 1.0,
)
trap_lengths = collect(4.0:0.5:6.0)
algorithm = SVM(16; candidates = 12, scale = 4.0, indep_tol = 1e-8)
spectrum = trapped_spectrum(
    ops, channel, trap_lengths, algorithm; levels = 1:2, tol = 1e-7, window = 2,
)
@assert all(spectrum.converged)
(all_converged = all(spectrum.converged), max_change = maximum(spectrum.energy_changes),
    max_condition = maximum(spectrum.condition_numbers))
```

The threshold-relative energies have one row per trap and one column per level:

```@example scattering
spectrum.relative_energies
```

Fit the inspected spectrum and examine the parameter and sensitivity outputs:

```@example scattering
params = fit_scattering_parameters(spectrum; ere_order = 2)
@assert all(params.used)
(a = params.a, r = params.r, shape = params.shape)
```

```@example scattering
(design_condition = params.design_condition, warnings = params.warnings)
```

```@example scattering
params.K_energy_sensitivity
```

The window fits show `a`, `r`, and `shape` as the minimum trap length increases:

```@example scattering
params.window_parameters
```

These finite-trap estimates illustrate the inference workflow. In particular,
the shape coefficient requires further trap-window and basis checks before
being quoted as a free-space value. The displayed energy-change sensitivity
should be read alongside the residuals, even when every convergence flag is true.

Once the solver settings and trap window have been examined, the convenience
function composes the same operations and retains the spectrum in `spectrum`:

```@example scattering
combined = scattering_parameters(
    ops, channel, trap_lengths, algorithm;
    levels = 1:2, ere_order = 2, tol = 1e-7, window = 2,
)
@assert all(combined.spectrum.converged)
@assert combined.coefficients ≈ params.coefficients
(a = combined.a, r = combined.r, shape = combined.shape)
```

## References

```@bibliography
Pages = ["scattering.md"]
Canonical = false
```
