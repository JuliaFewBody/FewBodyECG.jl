using LinearAlgebra: tr, dot, eigen, Symmetric, cond, qr, Diagonal, diag
using SpecialFunctions: besselj, bessely

# Independent reference: integrate the physical radial equation, match to free
# spherical waves, and regress with QR. No trap quantization or fit helpers.
function radial_scattering_K(V, μ, ℓ, k; step = 0.002, radius = 10.0)
    r0 = 1.0e-5
    n = ceil(Int, (radius - r0) / step)
    h = (radius - r0) / n
    u, v = r0^(ℓ + 1), (ℓ + 1) * r0^ℓ
    acceleration(r, u) = (ℓ * (ℓ + 1) / r^2 + 2μ * V(r) - k^2) * u
    for i in 0:(n - 1)
        r = r0 + i * h
        a1, b1 = v, acceleration(r, u)
        a2, b2 = v + h * b1 / 2, acceleration(r + h / 2, u + h * a1 / 2)
        a3, b3 = v + h * b2 / 2, acceleration(r + h / 2, u + h * a2 / 2)
        a4, b4 = v + h * b3, acceleration(r + h, u + h * a3)
        u += h * (a1 + 2a2 + 2a3 + a4) / 6
        v += h * (b1 + 2b2 + 2b3 + b4) / 6
    end
    z = k * radius
    j = sqrt(π / (2z)) * besselj(ℓ + 0.5, z)
    y = sqrt(π / (2z)) * bessely(ℓ + 0.5, z)
    jnext = sqrt(π / (2z)) * besselj(ℓ + 1.5, z)
    ynext = sqrt(π / (2z)) * bessely(ℓ + 1.5, z)
    f, g = radius * j, radius * y
    fp, gp = (ℓ + 1) * j - z * jnext, (ℓ + 1) * y - z * ynext
    L = v / u
    tanδ = (fp - L * f) / (gp - L * g)
    return k^(2ℓ + 1) / tanδ
end

function radial_scattering_ere(V, μ, ℓ, momenta; order = 2, kwargs...)
    x = momenta .^ 2
    scale = maximum(x)
    design = [(k2 / scale)^power for k2 in x, power in 0:order]
    values = [radial_scattering_K(V, μ, ℓ, k; kwargs...) for k in momenta]
    c = (qr(design) \ values) ./ scale .^ (0:order)
    return (;
        a = -1 / c[1], r = order ≥ 1 ? 2c[2] : nothing,
        shape = order ≥ 2 ? c[3] : nothing, coefficients = c, values,
    )
end

# Pure angular components are constructed explicitly: 1, x, and xy.
# Normalize the public H/S matrices before the dense generalized eigensolve.
function direct_scattering_spectrum(μ, terms, ℓ, lengths; n = 32, range = 1.0, levels = 1:2)
    energies = zeros(length(lengths), length(levels))
    conditions = zeros(length(lengths))
    for (i, b) in pairs(lengths)
        exponents = exp.(LinRange(log(0.02 / b^2), log(20 / range^2), n))
        gaussians = if ℓ == 0
            [Rank0Gaussian([α;;], [0.0]) for α in exponents]
        elseif ℓ == 1
            [Rank1Gaussian([α;;], [1.0], [0.0]) for α in exponents]
        else
            [Rank2Gaussian([α;;], [1.0 0.0 0.0], [0.0 1.0 0.0], [0.0]) for α in exponents]
        end
        basis = BasisSet(gaussians)
        S = build_overlap_matrix(basis)
        H = build_hamiltonian_matrix(basis, [terms; OscillatorOperator(1 / (8μ * b^4), [1.0])])
        D = Diagonal(1 ./ sqrt.(diag(S)))
        S = Symmetric(D * S * D)
        H = Symmetric(D * H * D)
        conditions[i] = cond(S)
        energies[i, :] = eigen(H, S).values[levels]
    end
    channel = ScatteringChannel([1], [2]; ℓ, threshold = 0, interaction_range = range)
    return TrapSpectrum(
        channel, lengths, energies; reduced_mass = μ,
        condition_numbers = conditions, basis_sizes = fill(n, length(lengths))
    )
end

@testset "Independent finite-range scattering physics" begin
    # The finite square well has u=sin(sqrt(k²+2μV₀)r) inside R=1.
    # This catches sign or logarithmic-derivative errors in the radial oracle.
    for k in (0.08, 0.2)
        q = sqrt(k^2 + 0.5)
        δ = atan(k * tan(q) / q) - k
        @test radial_scattering_K(_ -> -0.5, 0.5, 0, k; radius = 1.0) ≈ k / tan(δ) rtol = 1.0e-9
    end

    # ℏ=1, μ=1/2, length unit L, energy unit 1/(2μL²).
    # V=-exp(-r²) is weak, attractive, and has no threshold pole. The
    # radial oracle's kL=0.08:0.02:0.20 probes the low-energy ERE limit.
    # Its independently fitted (a,r,v) are approximately:
    # s: (-0.692192725, 2.89790436, 0.17750117)
    # p: (-0.078563875, 10.8828176, 1.5837795)
    # d: (-0.007519752, 130.506226, 12.86513).
    # Units are L^(2ℓ+1), L^(1-2ℓ), L^(3-2ℓ), respectively.
    V(r) = -exp(-r^2)
    μ = 0.5
    momenta = collect(0.08:0.02:0.2)
    lengths = collect(4.0:0.5:6.0)
    terms = Operator[KineticOperator([1.0;;]), GaussianOperator(-1.0, 1.0, [1.0])]
    for ℓ in 0:2
        @testset "Partial wave ℓ=$ℓ" begin
            reference = radial_scattering_ere(V, μ, ℓ, momenta; step = 0.001, radius = 10.0)
            coarse_radial = radial_scattering_ere(V, μ, ℓ, momenta; step = 0.002, radius = 8.0)
            far_radial = radial_scattering_ere(V, μ, ℓ, momenta; step = 0.001, radius = 12.0)
            wider_momenta = radial_scattering_ere(V, μ, ℓ, collect(0.1:0.025:0.25); step = 0.001)
            for check in (coarse_radial, far_radial)
                @test check.a ≈ reference.a rtol = 1.0e-7
                @test check.r ≈ reference.r rtol = 1.0e-6
                @test check.shape ≈ reference.shape rtol = 1.0e-4
            end
            @test wider_momenta.a ≈ reference.a rtol = 1.0e-6
            @test wider_momenta.r ≈ reference.r rtol = 1.0e-4
            @test wider_momenta.shape ≈ reference.shape rtol = 0.01

            # Geometric α=exp(range(log(.02/b²),log(20/L²),length=n)).
            # Radial levels n_r=0,1; increasing basis 32/36/40 → 36/40/44
            # changes E by <5e-10 and r by at most 0.018 (d wave).
            n = 36 + 4ℓ
            coarse = direct_scattering_spectrum(μ, terms, ℓ, lengths; n = n - 4)
            spectrum = direct_scattering_spectrum(μ, terms, ℓ, lengths; n)
            # The inner d-wave window needs only 40 members; 44 makes its
            # overlap much closer to singular without improving the comparison.
            inner = direct_scattering_spectrum(μ, terms, ℓ, collect(3.0:0.5:5.0); n = min(n, 40))
            fit = fit_scattering_parameters(spectrum)
            coarse_fit = fit_scattering_parameters(coarse)
            inner_fit = fit_scattering_parameters(inner)
            if ℓ == 2
                # Keep the inner-window comparison below the regime where
                # double-precision overlap solves lose most significant digits.
                @test maximum(inner.condition_numbers) < 1.0e14
                @info "Inner d-wave benchmark" basis = only(unique(inner.basis_sizes)) condition = maximum(inner.condition_numbers) parameters = (inner_fit.a, inner_fit.r, inner_fit.shape)
            end
            @test all(>(0), spectrum.energies)
            @test maximum(abs.(spectrum.energies - coarse.energies)) < 1.0e-9
            @test coarse_fit.a ≈ fit.a rtol = 1.0e-5
            @test coarse_fit.r ≈ fit.r rtol = 2.0e-4
            @test fit.a < 0
            # The basis error is far below the finite-trap/ERE error. Moving
            # b=3:0.5:5 to 4:0.5:6 reduces |Δa| for every channel; p/d Δr
            # falls from 1.6%/2.3% to 0.6%/0.95%. These are model tolerances.
            @test abs(fit.a - reference.a) < abs(inner_fit.a - reference.a)
            @test fit.a ≈ reference.a rtol = 0.001
            @test fit.r ≈ reference.r rtol = (ℓ == 0 ? 0.002 : 0.015)
            if ℓ > 0
                @test abs(fit.r - reference.r) < abs(inner_fit.r - reference.r)
            end
            if ℓ == 1
                # p-wave v is stable under basis refinement (0.000315),
                # with a 0.0603 trap-window change and 1.65% radial error.
                @test coarse_fit.shape ≈ fit.shape rtol = 0.001
                @test fit.shape ≈ reference.shape rtol = 0.05
                @test abs(fit.shape - reference.shape) < abs(inner_fit.shape - reference.shape)
            end
            # s/d shape values have larger trap-window dependence and are
            # deliberately not certified as accurate free-space coefficients.

            # A physical dilation r→2r, V→V/4, b→2b independently checks the
            # partial-wave dimensions, including inverse-length d-wave r.
            dilated_terms = Operator[
                KineticOperator([1.0;;]), GaussianOperator(-0.25, 0.25, [1.0]),
            ]
            dilated = direct_scattering_spectrum(μ, dilated_terms, ℓ, 2 .* lengths; n, range = 2.0)
            dilated_fit = fit_scattering_parameters(dilated)
            @test dilated.energies ≈ spectrum.energies ./ 4 rtol = 1.0e-9
            @test dilated_fit.a ≈ reference.a * 2^(2ℓ + 1) rtol = 0.001
            @test dilated_fit.r ≈ reference.r * 2.0^(1 - 2ℓ) rtol = (ℓ == 0 ? 0.002 : 0.015)
            @info "Independent Gaussian benchmark" ℓ radial = (reference.a, reference.r, reference.shape) trapped = (fit.a, fit.r, fit.shape) basis_energy_change = maximum(abs.(spectrum.energies - coarse.energies))
        end
    end

    @testset "Default-Halton s-wave orchestration" begin
        # Operators uses mass-normalized Jacobi coordinates (R=√2 x).
        # The direct fixture above uses physical R; agreement tests this seam.
        ops = Operators([1.0, 1.0])
        ops += "Kinetic"
        ops += ("Gaussian", 1, 2, -1.0, 1.0)
        channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
        direct = direct_scattering_spectrum(μ, terms, 0, lengths; n = 36)
        small = trapped_spectrum(ops, channel, lengths, SVM(16; candidates = 12, scale = 4.0); tol = 1.0e-7, window = 2)
        smoke = trapped_spectrum(ops, channel, lengths, SVM(24; candidates = 12, scale = 4.0); tol = 1.0e-7, window = 2)
        tighter = trapped_spectrum(ops, channel, lengths, SVM(16; candidates = 12, scale = 4.0, indep_tol = 1.0e-8); tol = 1.0e-7, window = 2)
        # Empirical default-cutoff envelope: 3.15e-6 at both basis limits;
        # allowing better-resolved near-dependent candidates gives <1e-8.
        # Final accepted-step ΔE alone is not an absolute-error estimate.
        for s in (small, smoke, tighter)
            @test all(s.converged)
            @test all(isfinite, s.condition_numbers)
            @test maximum(abs.(s.energy_changes)) ≤ 1.0e-7
            @test maximum(abs.(s.relative_energies - direct.energies)) < 5.0e-6
        end
        @test maximum(abs.(tighter.relative_energies - direct.energies)) < 1.0e-8
        @info "Halton benchmark" small_error = maximum(abs.(small.energies - direct.energies)) smoke_error = maximum(abs.(smoke.energies - direct.energies)) tighter_error = maximum(abs.(tighter.energies - direct.energies)) energy_changes = smoke.energy_changes
    end

    @testset "Fedorov–Pedersen Volkov sign and units" begin
        # Low-Energy Scattering Parameters.pdf, p.3 Eqs.(8)–(10):
        # VR=144.86 MeV, bR=.82 fm, VA=-83.34 MeV, bA=1.60 fm;
        # μphysical=(ℏc)²/(2*41.47 MeV fm²)=469.471 MeV/c².
        # In the ℏ=1 numerical Hamiltonian with E in MeV and r in fm,
        # μ=1/(2*41.47), so T=-41.47∇² and Vtrap=41.47r²/(4b⁴).
        volkov_mass = 1 / (2 * 41.47)
        volkov_terms = Operator[
            KineticOperator([41.47;;]),
            GaussianOperator(144.86, 1 / 0.82^2, [1.0]),
            GaussianOperator(-83.34, 1 / 1.6^2, [1.0]),
        ]
        volkov(r) = 144.86 * exp(-(r / 0.82)^2) - 83.34 * exp(-(r / 1.6)^2)
        # Fig.1 spans 4–10 fm; the text fits the last two fm. The paper
        # specifies the interval, not a numerical grid; use 8:0.5:10 fm.
        paper_lengths = collect(8.0:0.5:10.0)
        coarse = direct_scattering_spectrum(volkov_mass, volkov_terms, 0, paper_lengths; n = 32, range = 1.6)
        spectrum = direct_scattering_spectrum(volkov_mass, volkov_terms, 0, paper_lengths; n = 36, range = 1.6)
        smaller = direct_scattering_spectrum(volkov_mass, volkov_terms, 0, collect(6.0:0.5:8.0); n = 36, range = 1.6)
        larger = direct_scattering_spectrum(volkov_mass, volkov_terms, 0, collect(10.0:0.5:12.0); n = 36, range = 1.6)
        fit = fit_scattering_parameters(spectrum)
        coarse_fit = fit_scattering_parameters(coarse)
        smaller_fit = fit_scattering_parameters(smaller)
        larger_fit = fit_scattering_parameters(larger)
        reference = radial_scattering_ere(volkov, volkov_mass, 0, collect(0.02:0.01:0.08); step = 0.002, radius = 15.0)
        refined = radial_scattering_ere(volkov, volkov_mass, 0, collect(0.02:0.01:0.08); step = 0.001, radius = 18.0)
        @test reference.a ≈ refined.a rtol = 1.0e-8
        @test reference.r ≈ refined.r rtol = 1.0e-6
        @test maximum(abs.(spectrum.energies - coarse.energies)) < 2.0e-9
        @test coarse_fit.a ≈ fit.a atol = 1.0e-6
        @test all(-0.54592 .< spectrum.energies[:, 1] .< 0)
        @test all(>(0), spectrum.energies[:, 2])
        @test all(<(0), diff(spectrum.energies[:, 1]))
        @test abs(larger_fit.a - reference.a) < abs(fit.a - reference.a) < abs(smaller_fit.a - reference.a)
        # p.6 Eq.(18): k cotδ=+1/a_paper+r_e*k²/2+P*r_e³*k⁴.
        # Main text and plot legend give a_paper=-10.08 fm; Fig.1 caption
        # drops its minus sign. Standard a=-a_paper, r=r_e, shape=P*r_e³.
        paper_a = -10.08
        @test fit.a > 0 > paper_a
        @test fit.a ≈ -paper_a atol = 0.005
        @test fit.a ≈ reference.a atol = 0.005
        @test fit.r ≈ reference.r atol = 0.003
        @test fit.r ≈ 2.38 atol = 0.01
        @test fit.shape / fit.r^3 ≈ 0.028 atol = 0.001
        @info "Volkov benchmark" radial = (reference.a, reference.r, reference.shape) trapped = (fit.a, fit.r, fit.shape) paper_P = fit.shape / fit.r^3 energies = spectrum.energies
    end
end

@testset "Pure channel Gaussians" begin
    A = [1.2 0.1; 0.1 0.8]
    w = [1.0, -0.4]
    g0, g1, g2 = [FewBodyECG._channel_gaussian(A, ℓ, w) for ℓ in 0:2]
    @test g0 isa Rank0Gaussian && all(iszero, g0.s)
    @test g1 isa Rank1Gaussian && g1.a == w && all(iszero, g1.s)
    @test g2 isa Rank2Gaussian && all(iszero, g2.s)
    # The symmetric Cartesian quadratic tensor has zero trace: its
    # Laplacian vanishes, independently of the radial Gaussian exponent.
    a, b = vec(g2.a), vec(g2.b)
    tensor = (a * b' + b * a') / 2
    @test tr(tensor) == 0
    x = [0.3 -0.7 0.1; 0.8 0.2 -0.5]
    @test dot(vec(x), tensor * vec(x)) ≈ dot(w, x[:, 1]) * dot(w, x[:, 2])
    @test_throws ArgumentError FewBodyECG._channel_gaussian(A, 3, w)
end

@testset "Trapped spectrum workflow" begin
    ops = Operators([1.0, 1.0])
    ops += "Kinetic"
    channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 0.5)

    @testset "Composite candidate geometry spans Jacobi space" begin
        composite = Operators([1.0, 1.0, 1.0])
        composite += "Kinetic"
        composite += ManyBodyGaussianOperator(-5.0, [1.0 0.0; 0.0 1.0])
        for ℓ in 0:2
            ch = ScatteringChannel([1, 2], [3]; ℓ, threshold = -1.0, interaction_range = 1.0)
            result = trapped_spectrum(composite, ch, [1.0], SVM(2; candidates = 1))
            @test all(isfinite, result.energies)
            @test result.basis_sizes == [2]
            @test !any(result.converged)
        end
    end

    @testset "Equal-weight selection against dense analytic matrices" begin
        # Independent 3D s-wave Gaussian integrals, μ=1/2, trap b=1:
        # Sij=(π/(ai+aj))^(3/2), Tij/Sij=6ai*aj/(ai+aj),
        # Vij/Sij=3/(8(ai+aj)). No incremental eigen or scoring helpers.
        function dense_levels(a)
            S = [(π / (ai + aj))^1.5 for ai in a, aj in a]
            H = S .* [(6ai * aj + 3 / 8) / (ai + aj) for ai in a, aj in a]
            return eigen(Symmetric(H), Symmetric(S)).values[1:min(2, length(a))], cond(Symmetric(S))
        end
        a = Float64[]
        draw = 0
        for _ in 1:4
            proposals = Vector{Float64}[]
            for _ in 1:6
                draw += 1
                width = only(FewBodyECG.generate_bij(:quasirandom, draw, 1, 1.0))
                push!(proposals, [a; 1 / width^2])
            end
            a = proposals[argmin([sum(first(dense_levels(proposal))) for proposal in proposals])]
        end
        expected, condition = dense_levels(a)
        growth = SVM(basis = 4, candidates = 6, scale = 1.0, indep_tol = eps(Float64))
        result = trapped_spectrum(ops, channel, [1.0], growth; window = 1)
        @test vec(result.energies) ≈ expected rtol = 1.0e-10
        @test only(result.condition_numbers) ≈ condition rtol = 1.0e-8
        @test result.basis_sizes == [4]
        # Stable slots define an independent cyclic-sweep oracle: each slot is
        # replaced in place, so every original member is reconsidered once.
        for sweep in 1:2
            before = first(dense_levels(a))
            for i in eachindex(a)
                best = a[i]
                bestE = sum(first(dense_levels(a)))
                for _ in 1:3
                    draw += 1
                    width = only(FewBodyECG.generate_bij(:quasirandom, draw, 1, 1.0))
                    proposal = copy(a)
                    proposal[i] = 1 / width^2
                    E = sum(first(dense_levels(proposal)))
                    if E < bestE - 1.0e-12
                        best, bestE = proposal[i], E
                    end
                end
                a[i] = best
            end
            expected = first(dense_levels(a))
            refined = trapped_spectrum(ops, channel, [1.0], growth → Refine(sweep; candidates = 3, scale = 1.0, indep_tol = eps(Float64)))
            @test vec(refined.energies) ≈ expected rtol = 1.0e-10
            @test vec(refined.energy_changes) ≈ before - expected atol = 1.0e-10
        end
    end

    @testset "Cyclic refinement preserves retained members and consumes every draw" begin
        w = FewBodyECG._channel_kinematics(ops, channel).weight
        terms = [ops.terms; OscillatorOperator(0.25, w)]
        ctx = FewBodyECG._ctx(terms, ops.masses; state = 2, tol = 1.0e-4, window = 1, verbose = false)
        for ℓ in 0:2
            make_gaussian = A -> FewBodyECG._channel_gaussian(A, ℓ, w)
            basis = [make_gaussian([a;;]) for a in (0.1, 0.3, 1.0, 3.0)]
            seed = FewBodyECG.BasisState(basis, terms)
            # All finite-width candidates overlap this basis: a residual cut
            # immediately below one rejects replacements without skipping draws.
            refined, _ = FewBodyECG._trapped_stage(
                seed, Refine(2; candidates = 2, indep_tol = prevfloat(1.0)),
                ctx, 1:2, make_gaussian
            )
            @test all(refined.basis[i] === basis[i] for i in eachindex(basis))
            @test refined.draw == seed.draw + 2 * 4 * 2
            @test refined.eig.ε ≈ seed.eig.ε rtol = 1.0e-10
        end
    end

    @testset "Exact harmonic levels" begin
        lengths = [1.5, 0.8, 1.0]
        for ℓ in 0:2
            ch = ScatteringChannel([1], [2]; ℓ, threshold = 0, interaction_range = 0.5)
            spectrum = trapped_spectrum(
                ops, ch, lengths, SVM(basis = 25, candidates = 20);
                tol = 1.0e-4, window = 2
            )
            # μ=1/2 gives ω=1/b²; oscillator radial quantum numbers n=0,1.
            expected = [(2n + ℓ + 1.5) / b^2 for b in lengths, n in 0:1]
            @test spectrum.relative_energies ≈ expected rtol = 1.0e-6
            @test all(spectrum.converged)
            @test all(isfinite, spectrum.energies)
            @test spectrum.trap_lengths == lengths
            @test spectrum.reduced_mass == 0.5
            @test spectrum.range_ratios == 2 .* lengths
            @test spectrum.frequencies ≈ 1 ./ lengths .^ 2
            @test all(isfinite, spectrum.condition_numbers)
            @test all(>(1), spectrum.basis_sizes)
        end
        @test length(ops.terms) == 1
    end

    @testset "Level histories and incomplete traps" begin
        alg(n) = SVM(basis = n, candidates = 1, indep_tol = eps(Float64))
        lengths = [1.5, 0.8]
        prefix = trapped_spectrum(ops, channel, lengths, alg(2); window = 2)
        result = trapped_spectrum(ops, channel, lengths, alg(4); tol = 1.0e6, window = 2)
        @test result.energy_changes ≈ prefix.energies - result.energies
        @test all(result.converged)
        @test all(isnan, prefix.energy_changes)
        @test !any(prefix.converged)
        repeated = trapped_spectrum(ops, channel, lengths, alg(4); tol = 1.0e6, window = 2)
        @test repeated.energies == result.energies
        @test repeated.energy_changes == result.energy_changes
        @test repeated.condition_numbers == result.condition_numbers
        @test result.basis_sizes == [4, 4]
        @test result.tolerance == 1.0e6 && result.window == 2
        @test result.algorithm == alg(4)

        incomplete = trapped_spectrum(ops, channel, lengths, alg(1); window = 0)
        @test all(isfinite, incomplete.energies[:, 1])
        @test all(isnan, incomplete.energies[:, 2])
        @test !any(incomplete.converged)
        @test incomplete.basis_sizes == [1, 1]
        @test_throws ArgumentError fit_scattering_parameters(incomplete; ere_order = 0)
        partial = trapped_spectrum(ops, channel, lengths, alg(3); tol = 1.0e6, window = 2)
        @test all(isfinite, partial.energy_changes[:, 1])
        @test all(isnan, partial.energy_changes[:, 2])
        @test !any(partial.converged)

        # A shared tolerance between the two actual changes admits only
        # one trap. The other row remains available, but cannot enter a fit.
        changes = vec(maximum(result.energy_changes; dims = 2))
        mixed = trapped_spectrum(ops, channel, lengths, alg(4); tol = sum(changes) / 2, window = 2)
        @test count(identity, mixed.converged) == 2
        @test all(isfinite, mixed.energies)
        fit = fit_scattering_parameters(mixed; ere_order = 0)
        @test fit.used == mixed.converged
        @test all(isnan, fit.residuals[.!fit.used])

        shifted_channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = -0.3, interaction_range = 0.5)
        shifted = trapped_spectrum(ops, shifted_channel, lengths, alg(4); tol = 1.0e6, window = 2)
        @test shifted.energies == result.energies
        @test shifted.relative_energies ≈ result.energies .+ 0.3
        single = trapped_spectrum(ops, channel, [lengths[2]], alg(4); tol = 1.0e6, window = 2)
        @test single.energies[1, :] == result.energies[2, :]
    end

    @testset "Shared-basis refinement and pipelines" begin
        growth = SVM(basis = 5, candidates = 3, scale = 3.0)
        refinement(n) = Refine(sweeps = n, candidates = 4, scale = 1.0)
        for ℓ in 0:2
            ch = ScatteringChannel([1], [2]; ℓ, threshold = 0, interaction_range = 0.5)
            seed = trapped_spectrum(ops, ch, [1.0], growth; window = 1)
            one = trapped_spectrum(ops, ch, [1.0], growth → refinement(1); window = 1)
            two = trapped_spectrum(ops, ch, [1.0], growth → refinement(2); window = 1)
            split = trapped_spectrum(ops, ch, [1.0], growth → refinement(1) → refinement(1); window = 1)
            @test one.basis_sizes == seed.basis_sizes == two.basis_sizes
            @test sum(one.energies) ≤ sum(seed.energies) + 1.0e-10
            @test sum(two.energies) ≤ sum(one.energies) + 1.0e-10
            @test one.energy_changes ≈ seed.energies - one.energies
            @test two.energy_changes ≈ one.energies - two.energies
            @test two.energies == split.energies
            @test two.energy_changes == split.energy_changes
            @test all(two.converged) == all(x -> 0 ≤ x < 1.0e-4, two.energy_changes)
        end
        for bad in (
                Refine(), Pipeline(()), GVM(3), DynamicGVM(3),
                Refine() → SVM(), SVM() → GVM(3), SVM() → DynamicGVM(3),
            )
            @test_throws ArgumentError trapped_spectrum(ops, channel, [1.0], bad)
        end
        for levels in ([1, 2], 1:2:3, 2:3, 1:0)
            @test_throws ArgumentError trapped_spectrum(ops, channel, [1.0]; levels)
        end
        for lengths in ([0.0], [-1.0], [Inf], [NaN])
            @test_throws ArgumentError trapped_spectrum(ops, channel, lengths)
        end
        for tol in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError trapped_spectrum(ops, channel, [1.0]; tol)
        end
        for window in (-1, 1.5)
            @test_throws ArgumentError trapped_spectrum(ops, channel, [1.0]; window)
        end
        @test_logs trapped_spectrum(ops, channel, [1.0], SVM(1); verbose = false)
        @test_logs (:info, r"SVM: iteration 1/1") trapped_spectrum(ops, channel, [1.0], SVM(1); verbose = true)
        @test_logs (:info, r"SVM: iteration 1/1") (:info, r"Refine: iteration 1/1") trapped_spectrum(
            ops, channel, [1.0], SVM(1) → Refine(1); verbose = true
        )
    end

    @testset "Trapped stage settings are validated before any solve" begin
        invalid_svm = [
            [SVM(n) for n in (-1, 0)];
            [SVM(3; candidates = n) for n in (-2, 0)];
            [SVM(3; scale) for scale in (-1.0, 0.0, NaN, Inf, -Inf, :unsupported)];
            [SVM(3; indep_tol) for indep_tol in (-Inf, -0.1, 0.0, 1.0, 1.1, Inf, NaN)];
        ]
        invalid_refine = [
            [Refine(n) for n in (-1, 0)];
            [Refine(1; candidates = n) for n in (-2, 0)];
            [Refine(1; scale) for scale in (-1.0, 0.0, NaN, Inf, -Inf, :unsupported)];
            [Refine(1; indep_tol) for indep_tol in (-Inf, -0.1, 0.0, 1.0, 1.1, Inf, NaN)];
        ]
        # An empty grid prevents a late sampling error from accidentally
        # satisfying these assertions: validation must happen before trap loops.
        for bad in invalid_svm
            @test_throws ArgumentError trapped_spectrum(ops, channel, Float64[], bad)
        end
        for bad in [invalid_svm; invalid_refine]
            @test_throws ArgumentError trapped_spectrum(
                ops, channel, Float64[], SVM(1) → Refine(1) → bad
            )
        end
        # With actual work queued, an invalid later stage still fails without
        # emitting the first SVM iteration or consuming its candidate stream.
        @test_logs begin
            @test_throws ArgumentError trapped_spectrum(
                ops, channel, [1.0], SVM(2) → Refine(1; candidates = -1); verbose = true
            )
        end
        for indep_tol in (nextfloat(0.0), prevfloat(1.0))
            # Positive one-member bases remain legal with two requested levels.
            result = trapped_spectrum(
                ops, channel, [1.0], SVM(1; candidates = 1, indep_tol)
            )
            @test result.basis_sizes == [1]
            @test !any(result.converged)
        end
    end

    @testset "Convenience fit composition" begin
        finite_range = Operators([1.0, 1.0])
        finite_range += "Kinetic"
        finite_range += ("Gaussian", 1, 2, -0.5, 1.0)
        lengths = [1.0, 1.5, 2.0]
        alg = SVM(basis = 5, candidates = 3)
        spectrum = trapped_spectrum(finite_range, channel, lengths, alg; levels = 1:1, tol = 1.0e6, window = 1)
        expected = fit_scattering_parameters(spectrum; ere_order = 1)
        result = scattering_parameters(finite_range, channel, lengths, alg; levels = 1:1, ere_order = 1, tol = 1.0e6, window = 1)
        @test result.coefficients == expected.coefficients
        @test result.spectrum.energies == spectrum.energies
        @test result.spectrum.levels == 1:1
        @test result.shape === nothing
    end
end

@testset "ScatteringChannel" begin
    channel = ScatteringChannel(
        [1, 2], [3];
        ℓ = 2,
        threshold = -1.5,
        interaction_range = 0.75,
        inelastic_threshold = 0.5,
    )
    @test channel.fragments == ([1, 2], [3])
    @test channel.ℓ == 2
    @test channel.threshold == -1.5
    @test channel.interaction_range == 0.75
    @test channel.inelastic_threshold == 0.5

    for ℓ in 0:2
        @test ScatteringChannel([1], [2]; ℓ, threshold = 0, interaction_range = 1).ℓ == ℓ
    end

    @test_throws ArgumentError ScatteringChannel(
        [1], [2]; ℓ = 3, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2]; ℓ = -1, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        Int[], [1]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], Int[]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1, 1], [2]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2, 2]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [0], [2]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [-2]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [1, 2]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2]; ℓ = 0, threshold = Inf, interaction_range = 1
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2]; ℓ = 0, threshold = 0, interaction_range = 0
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2]; ℓ = 0, threshold = 0, interaction_range = Inf
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2];
        ℓ = 0,
        threshold = 0,
        interaction_range = 1,
        inelastic_threshold = Inf,
    )
    @test_throws ArgumentError ScatteringChannel(
        [1], [2];
        ℓ = 0,
        threshold = 0,
        interaction_range = 1,
        inelastic_threshold = 0,
    )
end

struct UnsupportedScatteringOperator <: Operator end

@testset "Scattering Hamiltonian domain" begin
    channel = ScatteringChannel(
        [1, 2], [3]; ℓ = 0, threshold = 0, interaction_range = 1
    )

    allowed = Operators([2.0, 3.0, 5.0])
    allowed += "Kinetic"
    allowed += ("Coulomb", 1, 2, 1.0)
    allowed += ("Coulomb", 2, 1, -0.5)
    internal_weight = allowed._U' * [1.0, -1.0, 0.0]
    allowed += CoulombOperator(0.1, (1 + 1.0e-10) .* internal_weight)
    allowed += ("Oscillator", 1, 2, 0.25)
    allowed += ("Gaussian", 1, 3, -2.0, 0.5)
    allowed += (r -> -exp(-r^2), numerical, 2, 3)
    allowed += ManyBodyGaussianOperator(0.1, [1.0 0.0; 0.0 1.0])
    @test isnothing(FewBodyECG._validate_scattering_hamiltonian(allowed, channel))

    cross_coulomb = Operators([2.0, 3.0, 5.0])
    cross_coulomb += "Kinetic"
    cross_coulomb += ("Coulomb", 1, 3, -1.0)
    @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
        cross_coulomb, channel
    )

    cross_oscillator = Operators([2.0, 3.0, 5.0])
    cross_oscillator += "Kinetic"
    cross_oscillator += ("Oscillator", 2, 3, 1.0)
    @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
        cross_oscillator, channel
    )

    for forbidden in (
            GaussianTensorOperator(1.0, 0.5, [1.0, 0.0], 1, 3),
            GaussianSpinOrbitOperator(1.0, 0.5, [1.0, 0.0], 1, 3),
            UnsupportedScatteringOperator(),
        )
        ops = Operators([2.0, 3.0, 5.0])
        ops += "Kinetic"
        ops += forbidden
        @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
            ops, channel
        )
    end

    missing_kinetic = Operators([2.0, 3.0, 5.0])
    missing_kinetic += ("Gaussian", 1, 3, -1.0, 0.5)
    @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
        missing_kinetic, channel
    )

    duplicate_kinetic = Operators([2.0, 3.0, 5.0])
    duplicate_kinetic += "Kinetic"
    duplicate_kinetic += "Kinetic"
    @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
        duplicate_kinetic, channel
    )

    for unidentifiable in (
            CoulombOperator(1.0, [0.2, 0.7]),
            OscillatorOperator(1.0, [0.2, 0.7]),
        )
        ops = Operators([2.0, 3.0, 5.0])
        ops += "Kinetic"
        ops += unidentifiable
        @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
            ops, channel
        )
    end

    zero_coulomb = Operators([2.0, 3.0, 5.0])
    zero_coulomb += "Kinetic"
    zero_coulomb += CoulombOperator(0.0, [0.2, 0.7])
    @test isnothing(FewBodyECG._validate_scattering_hamiltonian(zero_coulomb, channel))
end

@testset "Scattering operator data validation" begin
    @testset "Kinetic energy must use the declared particle masses" begin
        for masses in ([1.0, 1.0], [2.0, 3.0], [2.0, 3.0, 5.0], [1.0, 1.0e16, 1.0e16])
            channel = ScatteringChannel(
                1:(length(masses) - 1), [length(masses)];
                ℓ = 0, threshold = 0, interaction_range = 1
            )
            for kinetic in ("Kinetic", KineticOperator(masses), KineticOperator(Matrix(Λ(masses))))
                ops = Operators(masses)
                ops += kinetic
                @test isnothing(FewBodyECG._validate_scattering_hamiltonian(ops, channel))
            end
        end

        channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
        for K in ([1.0;;], [0.5 + 1.0e-9;;], [0.0;;], [-0.5;;], [NaN;;], [Inf;;], [0.5 0.0], [0.5 0.0; 0.0 0.5])
            ops = Operators([1.0, 1.0])
            ops += KineticOperator(K)
            # An empty grid proves rejection precedes matrix assembly.
            err = try
                trapped_spectrum(ops, channel, Float64[])
            catch error
                error
            end
            @test err isa ArgumentError && occursin("kinetic matrix", sprint(showerror, err)) &&
                occursin("ops.masses", sprint(showerror, err))
        end

        composite = Operators([2.0, 3.0, 5.0])
        composite += KineticOperator([0.5 1.0e-16; 0.0 0.5])
        composite_channel = ScatteringChannel([1, 2], [3]; ℓ = 0, threshold = 0, interaction_range = 1)
        @test_throws ArgumentError trapped_spectrum(composite, composite_channel, Float64[])

        # A hand-built 0.5 kinetic coefficient is correct in the normalized
        # Jacobi coordinate, including unequal masses. Here μ=6/5, b=1.
        unequal = Operators([2.0, 3.0])
        unequal += KineticOperator([0.5;;])
        spectrum = trapped_spectrum(unequal, channel, [1.0], SVM(25; candidates = 20); window = 2)
        @test spectrum.relative_energies ≈ [0.625 35 / 24] rtol = 1.0e-6
        @test all(spectrum.converged)
    end

    @testset "Mass-imbalanced particle pairs are identified by proximity" begin
        channel = ScatteringChannel([1, 2], [3]; ℓ = 0, threshold = 0, interaction_range = 1)
        for name in ("Coulomb", "Oscillator"), (i, j) in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))
            ops = Operators([1.0, 1.0e16, 1.0e16])
            ops += "Kinetic"
            ops += (name, i, j, 1.0)
            @test FewBodyECG._particle_pair(ops, last(ops.terms).w) == minmax(i, j)
            if i != 3 && j != 3
                @test isnothing(FewBodyECG._validate_scattering_hamiltonian(ops, channel))
            else
                err = try
                    trapped_spectrum(ops, channel, Float64[])
                catch error
                    error
                end
                @test err isa ArgumentError && occursin("cross-fragment", sprint(showerror, err))
            end
        end

        ops = Operators([1.0, 1.0e16, 1.0e16])
        internal = ops._U' * [1.0, -1.0, 0.0]
        external = ops._U' * [1.0, 0.0, -1.0]
        # Both pairs lie inside the admissibility tolerance, but these
        # perturbed weights still identify a unique closest physical pair.
        for sign in (-1, 1), (fraction, expected) in ((0.25, (1, 2)), (0.75, (1, 3)))
            w = sign .* ((1 - fraction) .* internal + fraction .* external)
            @test FewBodyECG._particle_pair(ops, w) == expected
        end
        # The midpoint is equally close to physically different pairs; its
        # sign cannot supply the missing information. A far weight has no match.
        for w in ((internal + external) / 2, -(internal + external) / 2, [0.2, 0.7])
            for potential in (CoulombOperator(1.0, w), OscillatorOperator(1.0, w))
                ambiguous = Operators(copy(ops.masses))
                ambiguous += "Kinetic"
                ambiguous += potential
                @test isnothing(FewBodyECG._particle_pair(ambiguous, w))
                err = try
                    trapped_spectrum(ambiguous, channel, Float64[])
                catch error
                    error
                end
                @test err isa ArgumentError && occursin("cannot identify the particle pair", sprint(showerror, err))
            end
        end
    end

    @testset "Pair ambiguity accounts for roundoff at the physical weight scale" begin
        # Pair (1,3) is internal here: choosing it for the rounded midpoint
        # would admit a term whose cross-fragment identity is unresolved.
        channel = ScatteringChannel([1, 3], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
        for mass_scale in (1.0e-100, 1.0, 1.0e100)
            masses = mass_scale .* [1.0, 1.0e20, 1.0e20]
            geometry = Operators(masses)
            a = geometry._U' * [1.0, -1.0, 0.0]
            b = geometry._U' * [1.0, 0.0, -1.0]
            midpoint = (a + b) / 2
            # A quarter ulp of the leading weight remains below its floating-
            # point uncertainty, even when the small component can resolve it.
            nearby = midpoint + [0.0, eps(abs(midpoint[1])) / 4]
            for sign in (-1, 1), point in (midpoint, nearby)
                w = sign .* point
                @test isnothing(FewBodyECG._particle_pair(geometry, w))
                for potential in (CoulombOperator(1.0, w), OscillatorOperator(1.0, w))
                    ops = Operators(masses)
                    ops += "Kinetic"
                    ops += potential
                    err = try
                        # No assembly can accidentally satisfy the rejection.
                        trapped_spectrum(ops, channel, Float64[])
                    catch error
                        error
                    end
                    @test err isa ArgumentError && occursin("cannot identify the particle pair", sprint(showerror, err))
                end
            end

            # Exact pair identities retain priority at large and small weight
            # scales, for both operator types and both pair orientations.
            for name in ("Coulomb", "Oscillator"), (i, j) in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))
                ops = Operators(masses)
                ops += "Kinetic"
                ops += (name, i, j, 1.0)
                @test FewBodyECG._particle_pair(ops, last(ops.terms).w) == minmax(i, j)
                if minmax(i, j) == (1, 3)
                    @test isempty(trapped_spectrum(ops, channel, Float64[]).energies)
                else
                    @test_throws r"cross-fragment" trapped_spectrum(ops, channel, Float64[])
                end
            end
        end

        # At a still larger mass ratio, distinct exact weights are only a few
        # ulps apart. Their unique zero residual must beat an approximate tie.
        extreme = Operators([1.0, 1.0e30, 1.0e30])
        a = extreme._U' * [1.0, -1.0, 0.0]
        b = extreme._U' * [1.0, 0.0, -1.0]
        for sign in (-1, 1)
            @test FewBodyECG._particle_pair(extreme, sign .* a) == (1, 2)
            @test FewBodyECG._particle_pair(extreme, sign .* b) == (1, 3)
            @test isnothing(FewBodyECG._particle_pair(extreme, sign .* ((a + b) / 2)))
        end
    end

    @testset "Declared-zero terms do not enter trapped sampling or assembly" begin
        channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
        clean = Operators([1.0, 1.0])
        clean += "Kinetic"
        algorithm = SVM(4; candidates = 6)
        baseline = trapped_spectrum(clean, channel, [1.0], algorithm; window = 1)
        for w in (Float64[], [1.0, 0.0], [Inf], [NaN], [0.0]), coefficient in (0.0, -0.0)
            for potential in (CoulombOperator(coefficient, w), GaussianOperator(coefficient, NaN, w))
                ops = Operators([1.0, 1.0])
                ops += "Kinetic"
                ops += potential
                original = copy(ops.terms)
                result = try
                    trapped_spectrum(ops, channel, [1.0], algorithm; window = 1)
                catch error
                    error
                end
                @test result isa TrapSpectrum
                if result isa TrapSpectrum
                    @test result.energies == baseline.energies
                    @test isequal(result.energy_changes, baseline.energy_changes)
                    @test result.condition_numbers == baseline.condition_numbers
                    @test result.basis_sizes == baseline.basis_sizes
                end
                @test all(ops.terms[i] === original[i] for i in eachindex(original))
                @test length(ops.terms) == length(original)
            end
            for potential in (CoulombOperator(1.0, w), GaussianOperator(1.0, NaN, w))
                nonzero = Operators([1.0, 1.0])
                nonzero += "Kinetic"
                nonzero += potential
                @test_throws ArgumentError trapped_spectrum(nonzero, channel, Float64[])
            end
        end
        for potential in (CoulombOperator(eps(Float64), [NaN]), GaussianOperator(eps(Float64), NaN, [NaN]))
            tiny = Operators([1.0, 1.0])
            tiny += "Kinetic"
            tiny += potential
            @test_throws ArgumentError trapped_spectrum(tiny, channel, Float64[])
        end
    end

    @testset "Nonfinite pair weights are rejected" begin
        channel = ScatteringChannel(
            [1, 2], [3]; ℓ = 0, threshold = 0, interaction_range = 1
        )
        for operator in (
                CoulombOperator(1.0, [Inf, 0.0]),
                OscillatorOperator(1.0, [Inf, 0.0]),
            )
            ops = Operators([2.0, 3.0, 5.0])
            ops += "Kinetic"
            ops += operator
            @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
                ops, channel
            )
        end
    end

    @testset "Nondecaying Gaussian terms are rejected" begin
        channel = ScatteringChannel(
            [1], [2]; ℓ = 0, threshold = 0, interaction_range = 1
        )
        for γ in (-1.0, 0.0, Inf, NaN)
            ops = Operators([1.0, 1.0])
            ops += "Kinetic"
            ops += GaussianOperator(1.0, γ, [1.0])
            @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
                ops, channel
            )
        end

        for operator in (
                GaussianOperator(Inf, 1.0, [1.0]),
                GaussianOperator(1.0, 1.0, [Inf]),
                GaussianOperator(1.0, 1.0, [0.0]),
                GaussianOperator(1.0, 1.0, [1.0, 0.0]),
            )
            ops = Operators([1.0, 1.0])
            ops += "Kinetic"
            ops += operator
            @test_throws ArgumentError FewBodyECG._validate_scattering_hamiltonian(
                ops, channel
            )
        end

        zero_gaussian = Operators([1.0, 1.0])
        zero_gaussian += "Kinetic"
        zero_gaussian += GaussianOperator(0.0, NaN, [Inf])
        @test isnothing(
            FewBodyECG._validate_scattering_hamiltonian(zero_gaussian, channel)
        )
    end
end

@testset "TrapSpectrum low-level constructor" begin
    channel = ScatteringChannel(
        [1], [2];
        ℓ = 1,
        threshold = 1.0,
        interaction_range = 2.0,
        inelastic_threshold = 4.0,
    )
    trap_lengths = [2.0, 4.0]
    absolute_energies = [1.5 2.0; 2.5 3.0]
    convergence_flags = Bool[1 0; 1 1]
    changes = [1.0e-5 NaN; 2.0e-5 3.0e-5]
    algorithm = SVM(8; candidates = 2)
    spectrum = TrapSpectrum(
        channel,
        trap_lengths,
        absolute_energies;
        reduced_mass = 2.5,
        channel_weight = [0.25],
        levels = 1:2,
        converged = convergence_flags,
        energy_changes = changes,
        condition_numbers = [10.0, 20.0],
        basis_sizes = [7, 8],
        tolerance = 1.0e-4,
        window = 3,
        algorithm,
    )

    @test spectrum.channel === channel
    @test spectrum.reduced_mass == 2.5
    @test spectrum.channel_weight == [0.25]
    @test spectrum.trap_lengths == trap_lengths
    @test spectrum.range_ratios == [1.0, 2.0]
    @test spectrum.frequencies ≈ [0.05, 0.0125]
    @test spectrum.levels == 1:2
    @test spectrum.energies == absolute_energies
    @test spectrum.relative_energies == [0.5 1.0; 1.5 2.0]
    @test spectrum.converged == convergence_flags
    @test spectrum.converged isa BitMatrix
    @test isequal(spectrum.energy_changes, changes)
    @test spectrum.condition_numbers == [10.0, 20.0]
    @test spectrum.basis_sizes == [7, 8]
    @test spectrum.tolerance == 1.0e-4
    @test spectrum.window == 3
    @test spectrum.algorithm === algorithm

    @testset "Iterable diagnostics normalize tuples" begin
        for kwargs in (
                (; channel_weight = (0.25,)),
                (; condition_numbers = (10, 20)),
                (; basis_sizes = (7, 8)),
            )
            tuple_spectrum = try
                TrapSpectrum(channel, (2, 4), absolute_energies; reduced_mass = 2.5, kwargs...)
            catch error
                error
            end
            @test tuple_spectrum isa TrapSpectrum
            if tuple_spectrum isa TrapSpectrum
                @test tuple_spectrum.relative_energies == [0.5 1.0; 1.5 2.0]
                for (field, values) in pairs(kwargs)
                    @test getproperty(tuple_spectrum, field) == collect(values)
                end
            end
        end
        @test_throws DimensionMismatch TrapSpectrum(
            channel, (2, 4), absolute_energies; reduced_mass = 2.5, condition_numbers = (10,)
        )
        @test_throws DimensionMismatch TrapSpectrum(
            channel, (2, 4), absolute_energies; reduced_mass = 2.5, basis_sizes = (7,)
        )
        @test_throws ArgumentError TrapSpectrum(
            channel, (2, 0), absolute_energies; reduced_mass = 2.5
        )
    end

    defaults = TrapSpectrum(
        channel, [2.0], [1.5;;]; reduced_mass = 2.5
    )
    @test defaults.levels == 1:1
    @test defaults.converged == trues(1, 1)
    @test isnan(defaults.energy_changes[1, 1])
    @test isnan(defaults.condition_numbers[1])
    @test defaults.basis_sizes == [0]
    @test isnan(defaults.tolerance)
    @test defaults.window == 0
    @test isnothing(defaults.algorithm)

    @test_throws ArgumentError TrapSpectrum(
        channel, trap_lengths, absolute_energies; reduced_mass = 0
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, trap_lengths, absolute_energies; reduced_mass = Inf
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, [2.0, 0.0], absolute_energies; reduced_mass = 2.5
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, [2.0, Inf], absolute_energies; reduced_mass = 2.5
    )
    @test_throws DimensionMismatch TrapSpectrum(
        channel, [2.0], absolute_energies; reduced_mass = 2.5
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, trap_lengths, absolute_energies; reduced_mass = 2.5, levels = 2:3
    )
    @test_throws DimensionMismatch TrapSpectrum(
        channel,
        trap_lengths,
        absolute_energies;
        reduced_mass = 2.5,
        converged = trues(1, 2),
    )
    @test_throws DimensionMismatch TrapSpectrum(
        channel,
        trap_lengths,
        absolute_energies;
        reduced_mass = 2.5,
        energy_changes = zeros(2, 1),
    )
    @test_throws DimensionMismatch TrapSpectrum(
        channel,
        trap_lengths,
        absolute_energies;
        reduced_mass = 2.5,
        condition_numbers = [1.0],
    )
    @test_throws DimensionMismatch TrapSpectrum(
        channel,
        trap_lengths,
        absolute_energies;
        reduced_mass = 2.5,
        basis_sizes = [1],
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, trap_lengths, absolute_energies; reduced_mass = 2.5, tolerance = 0
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, trap_lengths, absolute_energies; reduced_mass = 2.5, tolerance = Inf
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, trap_lengths, absolute_energies; reduced_mass = 2.5, window = -1
    )
    @test_throws ArgumentError TrapSpectrum(
        channel, [2.0], [4.0;;]; reduced_mass = 2.5
    )
    unconverged = TrapSpectrum(
        channel, [2.0], [4.0;;]; reduced_mass = 2.5, converged = falses(1, 1)
    )
    @test !unconverged.converged[1, 1]
end

@testset "Scattering channel kinematics" begin
    ops = Operators([2.0, 3.0, 5.0])
    channel = ScatteringChannel(
        [1, 2], [3]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    kin = FewBodyECG._channel_kinematics(ops, channel)
    MA, MB = 5.0, 5.0
    @test kin.reduced_mass ≈ MA * MB / (MA + MB)
    d = [2 / 5, 3 / 5, -1.0]
    @test kin.weight ≈ ops._U' * d

    incomplete = ScatteringChannel(
        [1], [2]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    out_of_range = ScatteringChannel(
        [1, 2], [4]; ℓ = 0, threshold = 0, interaction_range = 1
    )
    @test_throws ArgumentError FewBodyECG._channel_kinematics(ops, incomplete)
    @test_throws ArgumentError FewBodyECG._channel_kinematics(ops, out_of_range)
    @test_throws ArgumentError FewBodyECG._channel_kinematics(Operators(), channel)

    nonpositive_mass_ops = Operators([2.0, 3.0, 5.0])
    nonpositive_mass_ops.masses[1] = 0.0
    @test_throws ArgumentError FewBodyECG._channel_kinematics(
        nonpositive_mass_ops, channel
    )
end
using SpecialFunctions: gamma

@testset "Real signed trap quantization" begin
    @test isdefined(FewBodyECG, :_trap_K)
    if isdefined(FewBodyECG, :_trap_K)
        for ℓ in 0:2
            μ, b, E = 1.25, 2.0, -1.0
            ω = 1 / (2μ * b^2)
            expected = (-1)^(ℓ + 1) * 2^(ℓ + 0.5) / b^(2ℓ + 1) *
                gamma(0.75 + ℓ / 2 - E / (2ω)) /
                gamma(0.25 - ℓ / 2 - E / (2ω))
            @test FewBodyECG._trap_K(ℓ, E, μ, b) ≈ expected rtol = 1.0e-13
            # At μ=1/2, b=1, ω=1; the first numerator pole is E=ℓ+3/2.
            pole = ℓ + 1.5
            left = FewBodyECG._trap_K(ℓ, pole - 1.0e-8, 0.5, 1.0)
            right = FewBodyECG._trap_K(ℓ, pole + 1.0e-8, 0.5, 1.0)
            @test left > 1.0e7
            @test right < -1.0e7
            @test_throws ArgumentError FewBodyECG._trap_K(ℓ, pole, 0.5, 1.0)
            @test FewBodyECG._trap_K(ℓ, 0.5 - ℓ, 0.5, 1.0) === 0.0
            @test FewBodyECG._quantization_pole_distance(ℓ, pole - 0.2, 0.5, 1.0) ≈ 0.1
            @test FewBodyECG._quantization_pole_distance(ℓ, pole + 2.2, 0.5, 1.0) ≈ 0.1
            @test isfinite(FewBodyECG._trap_K(ℓ, -1000.0, 0.5, 1.0))
        end
        # Negative gamma arguments also carry their signs through the ratio.
        @test FewBodyECG._trap_K(0, 1.0, 0.5, 1.0) ≈
            -sqrt(2) * gamma(0.25) / gamma(-0.25)
    end
end
function scattering_bisect(f, lo, hi; iterations = 100)
    flo = f(lo)
    for _ in 1:iterations
        mid = (lo + hi) / 2
        fm = f(mid)
        if signbit(fm) == signbit(flo)
            lo, flo = mid, fm
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

# Independent direct-gamma oracle; it does not call the production quantization.
function scattering_direct_K(ℓ, E, μ, b)
    t = μ * b^2 * E
    denominator = 0.25 - ℓ / 2 - t
    denominator ≤ 0 && isinteger(denominator) && return 0.0
    return (-1)^(ℓ + 1) * 2^(ℓ + 0.5) / b^(2ℓ + 1) *
        gamma(0.75 + ℓ / 2 - t) / gamma(denominator)
end

function synthetic_scattering_spectrum(ℓ, coefficients; lengths = [1.0, 1.3, 1.8, 2.4])
    μ = 0.75
    energies = zeros(length(lengths), 2)
    for (i, b) in pairs(lengths), level in 1:2
        f(E) = scattering_direct_K(ℓ, E, μ, b) - evalpoly(2μ * E, coefficients)
        # Each interval lies strictly between consecutive numerator poles.
        low = (0.75 + ℓ / 2 + level - 1 + 1.0e-7) / (μ * b^2)
        high = (0.75 + ℓ / 2 + level - 1.0e-7) / (μ * b^2)
        grid = range(low, high; length = 1001)
        found = false
        for j in 1:(length(grid) - 1)
            lo, hi = grid[j], grid[j + 1]
            flo, fhi = f(lo), f(hi)
            if isfinite(flo) && isfinite(fhi) && signbit(flo) != signbit(fhi)
                energies[i, level] = scattering_bisect(f, lo, hi)
                found = true
                break
            end
        end
        found || error("synthetic root not bracketed")
    end
    channel = ScatteringChannel([1], [2]; ℓ, threshold = -0.25, interaction_range = 0.1)
    return TrapSpectrum(channel, lengths, energies .- 0.25; reduced_mass = μ)
end

@testset "Effective-range scaled QR fits" begin
    @test isdefined(FewBodyECG, :fit_scattering_parameters)
    if isdefined(FewBodyECG, :fit_scattering_parameters)
        for ℓ in 0:2, order in 0:2
            coefficients = [-0.7, 0.12, 0.03][1:(order + 1)]
            spectrum = synthetic_scattering_spectrum(ℓ, coefficients)
            fit = fit_scattering_parameters(spectrum; ere_order = order)
            @test fit isa ScatteringParameters
            @test fit.ℓ == ℓ
            @test fit.spectrum === spectrum
            @test fit.coefficients ≈ coefficients rtol = 1.0e-8
            @test fit.a ≈ 10 / 7 rtol = 1.0e-8
            @test order ≥ 1 ? isapprox(fit.r, 0.24; rtol = 1.0e-8) : isnothing(fit.r)
            @test order == 2 ? isapprox(fit.shape, 0.03; rtol = 1.0e-8) : isnothing(fit.shape)
            @test fit.k2 ≈ 1.5 .* spectrum.relative_energies
            @test fit.K ≈ evalpoly.(fit.k2, Ref(coefficients)) rtol = 1.0e-8
            @test maximum(abs, fit.residuals) < 1.0e-10
            @test fit.design_rank == order + 1
            @test isfinite(fit.design_condition)
            @test size(fit.covariance) == (order + 1, order + 1)
        end
        channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
        insufficient = TrapSpectrum(channel, [1.0], [0.1 0.2]; reduced_mass = 0.5)
        @test_throws ArgumentError fit_scattering_parameters(insufficient)
        @test_throws ArgumentError fit_scattering_parameters(insufficient; ere_order = -1)
        @test_throws ArgumentError fit_scattering_parameters(insufficient; ere_order = 3)
        exact = fit_scattering_parameters(insufficient; ere_order = 1)
        @test isnothing(exact.covariance)
        @test maximum(abs, exact.residuals) < 1.0e-14
        for E in (0.0, -0.2)
            singular = TrapSpectrum(channel, [1.0, 2.0], fill(E, 2, 2); reduced_mass = 0.5)
            @test_throws ArgumentError fit_scattering_parameters(singular; ere_order = 1)
            @test all(isfinite, fit_scattering_parameters(singular; ere_order = 0).K)
        end
        negative = TrapSpectrum(channel, [1.0], [-0.2 -0.3 -0.4]; reduced_mass = 0.5)
        @test all(<(0), fit_scattering_parameters(negative).k2)
        @test all(isfinite, fit_scattering_parameters(negative).coefficients)
        none = TrapSpectrum(channel, [1.0], [0.1 0.2]; reduced_mass = 0.5, converged = falses(1, 2))
        @test_throws ArgumentError fit_scattering_parameters(none; ere_order = 0)
    end
end
using LinearAlgebra: svd, Diagonal

@testset "Scattering fit diagnostics and failures" begin
    channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
    raw = [0.1 0.2; 0.3 0.4; 0.05 0.08; NaN 0.12]
    flags = trues(4, 2)
    flags[2, 2] = flags[4, 1] = false
    spectrum = TrapSpectrum(
        channel, [1.0, 1.2, 2.0, 3.0], raw;
        reduced_mass = 0.5, converged = flags, energy_changes = fill(0.006, 4, 2), tolerance = 0.01
    )
    fit = fit_scattering_parameters(spectrum; ere_order = 1)
    @test fit.used == flags
    @test fit.K[2, 2] ≈ scattering_direct_K(0, 0.4, 0.5, 1.2)
    @test isnan(fit.K[4, 1])
    @test all(isnan, fit.residuals[.!flags])
    @test all(isnan, fit.pole_distance[.!flags])
    @test all(isnan, fit.K_energy_sensitivity[.!flags])
    @test all(isfinite, fit.pole_distance[flags])
    @test all(isfinite, fit.K_energy_sensitivity[flags])
    @test :marginal_convergence in first.(fit.warnings)
    @test fit.window_starts == [1.0, 1.2, 2.0, 3.0]
    @test size(fit.window_parameters) == (4, 3)
    @test all(isnan, fit.window_parameters[:, 3])
    @test all(isnan, fit.window_parameters[4, :])
    @test fit.window_parameters[1, 1:2] ≈ [fit.a, fit.r]
    # Independent SVD oracle for coefficients and residual covariance in physical units.
    x = raw[flags]
    y = [
        scattering_direct_K(0, raw[index], 0.5, spectrum.trap_lengths[index[1]])
            for index in CartesianIndices(raw) if flags[index]
    ]
    X = hcat(ones(length(x)), x)
    U, s, V = svd(X)
    expected = V * Diagonal(1 ./ s) * U' * y
    variance = sum(abs2, y - X * expected) / (length(y) - 2)
    expected_covariance = variance * V * Diagonal(1 ./ s .^ 2) * V'
    @test fit.coefficients ≈ expected rtol = 1.0e-12
    @test fit.covariance ≈ expected_covariance rtol = 1.0e-12
    for index in CartesianIndices(raw)
        flags[index] || continue
        E, b = raw[index], spectrum.trap_lengths[index[1]]
        derivative = (
            scattering_direct_K(0, E + 1.0e-6, 0.5, b) -
                scattering_direct_K(0, E - 1.0e-6, 0.5, b)
        ) / 2.0e-6
        @test fit.K_energy_sensitivity[index] ≈ abs(derivative * 0.006) rtol = 1.0e-7
    end

    near_pole = TrapSpectrum(
        channel, [1.0], [1.5 - 1.0e-5;;];
        reduced_mass = 0.5, energy_changes = [1.0e-8;;]
    )
    near_fit = fit_scattering_parameters(near_pole; ere_order = 0)
    @test :near_quantization_pole in first.(near_fit.warnings)
    @test :energy_sensitive in first.(near_fit.warnings)
    @test near_fit.pole_distance[1] ≈ 5.0e-6
    ill = TrapSpectrum(channel, [1.0], [0.1 0.1 + 1.0e-10 0.1 + 2.0e-10]; reduced_mass = 0.5)
    ill_fit = fit_scattering_parameters(ill; ere_order = 1)
    @test ill_fit.design_rank == 2
    @test ill_fit.design_condition > 1 / sqrt(eps(Float64))
    @test :ill_conditioned_fit in first.(ill_fit.warnings)

    # Trap-dependent constant ERE values produce known window means and a large drift.
    lengths = [2.4, 1.0, 1.8, 1.3]
    drifting = [
        synthetic_scattering_spectrum(0, [-c]; lengths = [b]).energies[1, :]
            for (c, b) in zip([4.0, 1.0, 3.0, 2.0], lengths)
    ]
    drift = TrapSpectrum(channel, lengths, reduce(vcat, permutedims.(drifting)) .+ 0.25; reduced_mass = 0.75)
    drift_fit = fit_scattering_parameters(drift; ere_order = 0)
    @test drift_fit.window_starts == sort(lengths)
    @test drift_fit.window_parameters[:, 1] ≈ [1 / 2.5, 1 / 3, 1 / 3.5, 1 / 4]
    @test :unstable_trap_window in first.(drift_fit.warnings)

    for ℓ in 0:2
        zero_channel = ScatteringChannel([1], [2]; ℓ, threshold = 0, interaction_range = 1)
        E = 0.5 - ℓ
        zero_spectrum = TrapSpectrum(zero_channel, [1.0], [E;;]; reduced_mass = 0.5, energy_changes = [1.0e-6;;])
        zero_fit = fit_scattering_parameters(zero_spectrum; ere_order = 0)
        derivative = (
            scattering_direct_K(ℓ, E + 1.0e-6, 0.5, 1.0) -
                scattering_direct_K(ℓ, E - 1.0e-6, 0.5, 1.0)
        ) / 2.0e-6
        @test zero_fit.K[1] === 0.0
        @test zero_fit.K_energy_sensitivity[1] ≈ abs(derivative * 1.0e-6) rtol = 1.0e-7
    end

    pole = TrapSpectrum(channel, [1.0], [1.5;;]; reduced_mass = 0.5)
    @test_throws ArgumentError fit_scattering_parameters(pole; ere_order = 0)
    nonfinite = TrapSpectrum(channel, [1.0], [NaN;;]; reduced_mass = 0.5)
    @test_throws ArgumentError fit_scattering_parameters(nonfinite; ere_order = 0)
    # The fit rechecks the threshold because the stored convergence mask is mutable.
    closed = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1, inelastic_threshold = 0.1)
    threshold = TrapSpectrum(closed, [1.0], [0.2;;]; reduced_mass = 0.5, converged = falses(1, 1))
    threshold.converged[1] = true
    @test_throws ArgumentError fit_scattering_parameters(threshold; ere_order = 0)
end
@testset "Excluded poles and viable trap windows" begin
    channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
    excluded_pole = TrapSpectrum(
        channel, [1.0], [0.1 1.5];
        reduced_mass = 0.5, converged = Bool[1 0], energy_changes = [0.0 1.0]
    )
    # A rejected level must not prevent fitting the remaining converged data.
    fit = try
        fit_scattering_parameters(excluded_pole; ere_order = 0)
    catch error
        error
    end
    @test fit isa ScatteringParameters
    if fit isa ScatteringParameters
        @test fit.k2 == [0.1 1.5]
        @test isnan(fit.K[1, 2])
        @test fit.coefficients[1] ≈ scattering_direct_K(0, 0.1, 0.5, 1.0)
        @test isempty(fit.warnings)
    end
    singular_window = TrapSpectrum(channel, [1.0, 2.0], [0.1 0.2; 0.3 0.3]; reduced_mass = 0.5)
    @test all(isnan, fit_scattering_parameters(singular_window; ere_order = 1).window_parameters[2, :])
    stable = synthetic_scattering_spectrum(2, [-0.7, 0.12, 0.03])
    stable_fit = fit_scattering_parameters(stable)
    @test isempty(stable_fit.warnings)
    @test stable_fit.window_parameters[1:3, :] ≈ repeat([10 / 7 0.24 0.03], 3) rtol = 1.0e-8
    @test all(isnan, stable_fit.window_parameters[4, :])
    for ℓ in 0:2, n in 1:3
        c = ScatteringChannel([1], [2]; ℓ, threshold = 0, interaction_range = 1)
        E = 0.5 - ℓ + 2n
        s = TrapSpectrum(c, [1.0], [E;;]; reduced_mass = 0.5, energy_changes = [1.0e-6;;])
        zero_fit = fit_scattering_parameters(s; ere_order = 0)
        derivative = (
            scattering_direct_K(ℓ, E + 1.0e-6, 0.5, 1.0) -
                scattering_direct_K(ℓ, E - 1.0e-6, 0.5, 1.0)
        ) / 2.0e-6
        @test zero_fit.K_energy_sensitivity[1] ≈ abs(derivative * 1.0e-6) rtol = 1.0e-7
    end
end
@testset "Resonant trap-window instability" begin
    channel = ScatteringChannel([1], [2]; ℓ = 0, threshold = 0, interaction_range = 1)
    # At E=0, K=-sqrt(2)*Γ(3/4)/(b*Γ(1/4)); at E=1/(2b²), K=0.
    # Thus the last window is exactly resonant while earlier windows have finite a.
    spectrum = TrapSpectrum(
        channel, [1.0, 2.0, 4.0], [0.0; 0.0; 0.03125;;];
        reduced_mass = 0.5, energy_changes = zeros(3, 1)
    )
    fit = fit_scattering_parameters(spectrum; ere_order = 0)
    q = sqrt(2) * gamma(0.75) / gamma(0.25)
    @test fit.window_parameters[1:2, 1] ≈ [2 / q, 4 / q]
    @test fit.window_parameters[3, 1] == -Inf
    @test :unstable_trap_window in first.(fit.warnings)

    # Three identical exact resonances do not establish window dependence.
    resonant = TrapSpectrum(
        channel, [1.0, 2.0, 4.0], [0.5; 0.125; 0.03125;;];
        reduced_mass = 0.5, energy_changes = zeros(3, 1)
    )
    resonant_fit = fit_scattering_parameters(resonant; ere_order = 0)
    @test all(==(-Inf), resonant_fit.window_parameters[:, 1])
    @test !(:unstable_trap_window in first.(resonant_fit.warnings))

    # Tiny nonzero d-wave K values can overflow a with physically opposite signs.
    # Γ(1/4-t) in the s-wave formula is replaced by Γ(-3/4-t) here:
    # at t=-1/2 the ratio has negative sign, and at t=1/2 positive sign.
    dchannel = ScatteringChannel([1], [2]; ℓ = 2, threshold = 0, interaction_range = 1)
    lengths = [1.0e62, 2.0e62, 4.0e62]
    for t in ([-0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        energies = reshape(2 .* t ./ lengths .^ 2, :, 1)
        extreme = TrapSpectrum(dchannel, lengths, energies; reduced_mass = 0.5, energy_changes = zeros(3, 1))
        extreme_fit = fit_scattering_parameters(extreme; ere_order = 0)
        @test all(isinf, extreme_fit.window_parameters[:, 1])
        @test extreme_fit.window_parameters[:, 1] == (t[1] < 0 ? [-Inf, Inf, Inf] : [Inf, Inf, Inf])
        @test (:unstable_trap_window in first.(extreme_fit.warnings)) == (t[1] < 0)
    end
end
