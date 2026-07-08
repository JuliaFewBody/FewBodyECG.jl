# Convergence

Every `Solution` carries a `ConvergenceReport`.  The report is honest about
what the method can certify.

## Saturation

Stochastic methods report `criterion = :saturation` when the energy change
over the last `window` committed additions is below `tol`.  This means the
current sampler and scale stopped finding improvements.  It is not a proof of
the exact eigenvalue.

That distinction matters.  A single-scale H2+ run can plateau above the
physical energy because proton-proton and electron coordinates live on very
different length scales.  Increase `basis`, change `scale`, add `Refine`, or
follow with `Variational` before treating the plateau as physical.

## Stationarity

Gradient methods report `criterion = :stationarity` when the optimizer meets
its gradient tolerance.  This is a local stationarity statement in the
Gaussian-parameter landscape.  The variational upper-bound property still
holds: computed energies do not go below the exact eigenvalue for the chosen
Hamiltonian.

## Early stops

`criterion = :early_stop` means no admissible candidate was found, often
because the basis became nearly singular.  Try a smaller `scale`, fewer basis
functions, or a different warm start.

## Conditioning

ECG bases are non-orthogonal.  Large overlap condition numbers are normal.
The stochastic solver uses a whitened incremental eigensolver, and the dense
power-user solver regularizes ill-conditioned overlap matrices when needed.
The report includes `cond_S` so you can distinguish physical convergence from
linear-algebra stress.

## Plots

`plot(sol)` shows the stage energy history.  In a pipeline, each stage gets its
own curve.  `plot(sol, reference)` adds a horizontal reference line.  A flat
curve is a useful diagnostic, but always read it together with the method and
the report criterion.
