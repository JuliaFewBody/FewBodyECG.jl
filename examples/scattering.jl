using FewBodyECG

# Two unit-mass particles with V(R) = -exp(-R²), in units where ℏ = 1.
ops = Operators([1.0, 1.0])
ops += "Kinetic"
ops += ("Gaussian", 1, 2, -1.0, 1.0)

channel = ScatteringChannel(
    [1], [2];
    ℓ = 0,
    threshold = 0.0,
    interaction_range = 1.0,
)

trap_lengths = collect(4.0:0.5:6.0)
algorithm = SVM(16; candidates = 12, scale = 4.0, indep_tol = 1.0e-8)

spectrum = trapped_spectrum(
    ops,
    channel,
    trap_lengths,
    algorithm;
    levels = 1:2,
    tol = 1.0e-7,
    window = 2,
)
all(spectrum.converged) || error("trapped spectrum did not converge; increase the basis")

params = fit_scattering_parameters(spectrum; ere_order = 2)

println("Threshold-relative trap energies (rows: b, columns: levels):")
show(stdout, MIME("text/plain"), spectrum.relative_energies)
println()
println("Maximum recorded energy change: ", maximum(spectrum.energy_changes))
println("Maximum overlap condition number: ", maximum(spectrum.condition_numbers))
println("s-wave scattering length a₀: ", params.a)
println("effective range r₀: ", params.r)
println("shape coefficient v₀: ", params.shape)
println("Numerical/model warnings: ", isempty(params.warnings) ? "none" : params.warnings)

# Compare basis sizes, trap windows, and ERE orders before quoting the
# higher-order coefficients as free-space observables.
