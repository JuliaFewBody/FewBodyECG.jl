# Spin-Aware Analytic Matrix Elements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paper-consistent supervector, Gaussian-potential, tensor, and spin-orbit matrix elements while making the solvers complex-safe and moving ancillary helpers into `src/utils/`.

**Architecture:** Store shifted ECGs as an `N x 3` shift supervector, derive every shifted rank-0 matrix element from shared Gaussian data, and factor spin-product matrix elements from their orbital counterparts. Keep rank-1/rank-2 pre-factors in their source-derived zero-shift domain. Multiple dispatch remains the sole operator-extension mechanism.

**Tech Stack:** Julia 1.7+, LinearAlgebra, SpecialFunctions, QuasiMonteCarlo, FewBodyHamiltonians, Test, Aqua, Runic.

## Global Constraints

- Follow Fedorov (2017) Eqs. (8), (14), (15), (18), (38)-(40), (61), and (65), and Fedorov et al. (2024) Eqs. (1)-(41).
- Shifts are real `N x 3` matrices; vector inputs map to the fixed `z` axis only for compatibility.
- Rank-1 and rank-2 Gaussian pre-factors require zero shifts.
- Spin is a direct product of spin-1/2 projection states; complex elements are preserved end-to-end.
- Do not add `Plots` or any other dependency.
- Do not create commits, stage files, or modify Git configuration.
- Apply `runic --inplace` only to changed Julia source and test files.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/types.jl` | Supervector Gaussian constructors, spin-product basis, and operator data types. |
| `src/matrix_elements.jl` | Shared shifted-Gaussian data and all analytic operator dispatch. |
| `src/hamiltonian.jl` | Typed Hermitian assembly and complex generalized eigensolve. |
| `src/sampling.jl` | `N x 3` rank-0 shift generation. |
| `src/utils/solver_results.jl` | Result container, wavefunction evaluation, and convergence data. |
| `src/utils/observables.jl` | Correlation-profile data. |
| `src/utils/plotting.jl` | Caller-supplied plotting adapter with no plotting dependency. |
| `src/FewBodyECG.jl` | Exports and includes for the new source layout. |
| `test/test_types.jl` | Shape, spin, and operator-construction tests. |
| `test/test_matrix_elements.jl` | Reference-formula and spin-dependent matrix-element tests. |
| `test/test_hamiltonian.jl` | Complex Hermitian assembly and generalized eigensolve tests. |
| `test/test_sampling.jl` | Supervector sampler tests. |
| `test/test_utils.jl` | Moved utility and plotting-adapter tests. |

## Task 1: Establish Paper-Consistent Basis and Operator Types

**Files:**
- Modify: `src/types.jl:1-60`
- Modify: `src/FewBodyECG.jl:6-25`
- Test: `test/test_types.jl`

**Interfaces:**
- Produces `Rank0Gaussian(A, s::AbstractMatrix)` with `size(s) == (size(A, 1), 3)`.
- Produces `SpinProjection`, `SpinState`, `SpinGaussian`, `GaussianPotential`, `OscillatorPotential`, `ManyBodyGaussianPotential`, `GaussianTensorPotential`, and `GaussianSpinOrbitPotential`.
- Produces `spin_overlap(bra::SpinState, ket::SpinState)` and `spin_element(bra, ket, site, component)` for `component in (:x, :y, :z)`.

- [ ] **Step 1: Write failing supervector and spin tests**

Append focused cases to `test/test_types.jl`:

```julia
@testset "Supervector and spin types" begin
    A = [1.0 0.1; 0.1 1.3]
    s = [0.0 0.1 0.2; -0.3 0.0 0.4]
    @test Rank0Gaussian(A, s).s == s
    @test Rank0Gaussian(A, [0.2, -0.1]).s == [0.0 0.0 0.2; 0.0 0.0 -0.1]
    @test_throws ArgumentError Rank0Gaussian(A, zeros(2, 2))

    updown = SpinState([up, down])
    downup = SpinState([down, up])
    @test spin_overlap(updown, updown) == 1
    @test spin_overlap(updown, downup) == 0
    @test spin_element(updown, downup, 1, :x) == 0.5
    @test spin_element(updown, downup, 1, :y) == -0.5im
    @test spin_element(updown, downup, 1, :z) == 0
end
```

- [ ] **Step 2: Run the focused type tests and verify the expected RED failure**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_types.jl")'`

Expected: failure because matrix shifts and `SpinState`/`spin_element` do not exist.

- [ ] **Step 3: Implement the minimal type layer**

Replace the three scalar-shift fields with a shared matrix-shift representation. The compatibility helper and the new spin primitives must have these semantics:

```julia
function _shift_matrix(s::AbstractVector{T}) where {T<:Real}
    return hcat(zeros(T, length(s), 2), s)
end

function _check_shift(A, s)
    size(s) == (size(A, 1), 3) ||
        throw(ArgumentError("s must have size ($(size(A, 1)), 3)"))
    return s
end

@enum SpinProjection::Int8 down = -1 up = 1

struct SpinState
    projections::Vector{SpinProjection}
end

SpinState(projections::AbstractVector{SpinProjection}) = SpinState(collect(projections))

struct SpinGaussian{G<:GaussianBase} <: GaussianBase
    orbital::G
    spin::SpinState
end

spin_overlap(bra::SpinState, ket::SpinState) = bra.projections == ket.projections ? 1.0 : 0.0
```

Implement `spin_element` as the **single-site** matrix element: use `+/-0.5` for `:z`, `0.5` for a flipped `:x`, and `-0.5im` (`up` bra) or `0.5im` (`down` bra) for a flipped `:y`. It deliberately ignores other sites so tensor contractions can multiply two local factors. Reject invalid site indices and component symbols with `ArgumentError`.

Define the five immutable `PotentialTerm` operator structs with fields `(coefficient, gamma, w)`, `(coefficient, w)`, `(coefficient, W)`, and `(coefficient, gamma, w, i, j, traceless)` / `(coefficient, gamma, w, i, j)` respectively. Constructors check positive `gamma`, square `W`, and `i != j`.

- [ ] **Step 4: Export the new public API**

Update `src/FewBodyECG.jl` so callers can construct the new types:

```julia
export GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, SpinState, SpinProjection, up, down
export SpinGaussian, BasisSet, ECG, KineticOperator, CoulombOperator
export GaussianPotential, OscillatorPotential, ManyBodyGaussianPotential
export GaussianTensorPotential, GaussianSpinOrbitPotential
```

Keep `spin_overlap` and `spin_element` internal and import them explicitly in tests.

- [ ] **Step 5: Run the focused type tests and verify GREEN**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_types.jl")'`

Expected: all type tests pass, including the imaginary `S_y` matrix element.

- [ ] **Step 6: Preserve the 2024 pre-factor domain**

Update `Rank1Gaussian` and `Rank2Gaussian` constructors to accept an `N x 3` shift (or vector compatibility input), then add a private `_require_zero_shift(g)` that throws `ArgumentError("Rank$(rank)Gaussian matrix elements require zero shifts")` unless `all(iszero, g.s)`. Call it at the start of every rank-1/rank-2 overlap, kinetic, and Coulomb method.

Add this regression case to `test/test_types.jl`:

```julia
@test_throws ArgumentError FewBodyECG._compute_matrix_element(
    Rank1Gaussian([1.0;;], [1.0], [0.0 0.0 0.1]),
    Rank1Gaussian([1.0;;], [1.0], zeros(1, 3)),
)
```

- [ ] **Step 7: Do not commit**

The user explicitly retains Git ownership. Leave changes unstaged.

## Task 2: Rebuild Shifted Rank-0 Matrix Elements and Add Central Gaussian Families

**Files:**
- Modify: `src/matrix_elements.jl:1-252`
- Test: `test/test_matrix_elements.jl`

**Interfaces:**
- Consumes matrix shifts and the new scalar potential types.
- Produces `_compute_matrix_element(::Rank0Gaussian, ::Rank0Gaussian[, ::Operator])` consistent with the cited equations.

- [ ] **Step 1: Write failing paper-equation tests**

Add these tests to `test/test_matrix_elements.jl` after importing the new operator types:

```julia
@testset "Shifted rank-0 Gaussian potentials" begin
    A = [1.2;;]
    B = [0.8;;]
    sA = reshape([0.2, -0.1, 0.3], 1, 3)
    sB = reshape([-0.1, 0.4, 0.2], 1, 3)
    bra, ket = Rank0Gaussian(B, sB), Rank0Gaussian(A, sA)
    w, gamma = [1.0], 0.7
    R = inv(A + B)
    v = sA + sB
    updated = A + B + gamma * w * w'
    expected_gaussian = exp(tr(v' * inv(updated) * v) / 4) * (pi / det(updated))^(3 / 2)
    expected_oscillator = (3 * (w' * R * w)[1] / 2 + norm(vec(w' * R * v))^2 / 4) *
        _compute_matrix_element(bra, ket)

    @test _compute_matrix_element(bra, ket, GaussianPotential(1.0, gamma, w)) ≈ expected_gaussian
    @test _compute_matrix_element(bra, ket, OscillatorPotential(1.0, w)) ≈ expected_oscillator
end
```

Add a many-body check with diagonal `W` and an invalid negative-range constructor case.

- [ ] **Step 2: Run the matrix-element tests and verify RED**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_matrix_elements.jl")'`

Expected: failure because the new operator methods are absent and the current shifted formulas accept vectors only.

- [ ] **Step 3: Implement shared paper notation once**

At the top of `src/matrix_elements.jl`, use source-consistent bra/ket names and helpers:

```julia
_superdot(x, M, y) = tr(x' * M * y)
_overlap_prefactor(B) = (pi^size(B, 1) / det(B))^(3 / 2)

function _rank0_data(bra::Rank0Gaussian, ket::Rank0Gaussian)
    B, A = bra.A, ket.A
    R = inv(A + B)
    v = ket.s + bra.s
    M = exp(_superdot(v, R, v) / 4) * _overlap_prefactor(A + B)
    return (; A, B, R, v, M)
end

function _updated_rank0_data(bra::Rank0Gaussian, ket::Rank0Gaussian, W)
    data = _rank0_data(bra, ket)
    Bprime = data.A + data.B + W
    Rprime = inv(Bprime)
    Mprime = exp(_superdot(data.v, Rprime, data.v) / 4) * _overlap_prefactor(Bprime)
    return (; data..., Bprime, Rprime, Mprime)
end
```

Rewrite rank-0 overlap, kinetic, and Coulomb in terms of these helpers. In kinetic terms, replace every source vector product with `_superdot`; for Coulomb use `q = vec(w' * R * v / 2)`, `qnorm = norm(q)`, and the finite `qnorm == 0` limit.

- [ ] **Step 4: Implement the three new central methods**

Use these direct formulas:

```julia
function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::GaussianPotential)
    data = _updated_rank0_data(bra, ket, op.gamma * op.w * op.w')
    return op.coefficient * data.Mprime
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::OscillatorPotential)
    data = _rank0_data(bra, ket)
    q = vec(op.w' * data.R * data.v / 2)
    radial_square = 3 * (op.w' * data.R * op.w)[1] / 2 + dot(q, q)
    return op.coefficient * radial_square * data.M
end

function _compute_matrix_element(bra::Rank0Gaussian, ket::Rank0Gaussian, op::ManyBodyGaussianPotential)
    data = _updated_rank0_data(bra, ket, op.W)
    return op.coefficient * data.Mprime
end
```

- [ ] **Step 5: Make orbital-only operators diagonal in spin**

Add a forwarding method for every orbital-only operator:

```julia
function _compute_matrix_element(bra::SpinGaussian, ket::SpinGaussian, op::Union{KineticOperator, CoulombOperator, GaussianPotential, OscillatorPotential, ManyBodyGaussianPotential})
    return spin_overlap(bra.spin, ket.spin) * _compute_matrix_element(bra.orbital, ket.orbital, op)
end
```

Do the same for overlap without an operator.

- [ ] **Step 6: Run the focused matrix tests and verify GREEN**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_matrix_elements.jl")'`

Expected: the new formula checks pass and prior rank-0/rank-1/rank-2 checks remain green after adapting all shifts to `zeros(N, 3)`.

- [ ] **Step 7: Do not commit**

Leave all modifications unstaged.

## Task 3: Add Physical Gaussian Tensor and Spin-Orbit Matrix Elements

**Files:**
- Modify: `src/matrix_elements.jl`
- Test: `test/test_matrix_elements.jl`

**Interfaces:**
- Consumes `SpinGaussian{Rank0Gaussian}`, `GaussianTensorPotential`, and `GaussianSpinOrbitPotential`.
- Produces complex scalar matrix elements for the physical spin-product states.

- [ ] **Step 1: Write failing spin-operator tests**

Add the following test set, using two spin sites and a non-collinear shifted one-coordinate orbital:

```julia
@testset "Gaussian tensor and spin-orbit potentials" begin
    orbital = Rank0Gaussian([1.0;;], reshape([0.2, -0.3, 0.1], 1, 3))
    updown = SpinGaussian(orbital, SpinState([up, down]))
    downup = SpinGaussian(orbital, SpinState([down, up]))
    tensor = GaussianTensorPotential(1.0, 0.4, [1.0], 1, 2; traceless = true)
    spin_orbit = GaussianSpinOrbitPotential(0.7, 0.4, [1.0], 1, 2)

    @test _compute_matrix_element(updown, updown, tensor) isa Complex
    @test _compute_matrix_element(updown, downup, tensor) ≈
        conj(_compute_matrix_element(downup, updown, tensor))
    @test _compute_matrix_element(updown, downup, spin_orbit) ≈
        conj(_compute_matrix_element(downup, updown, spin_orbit))
end
```

Add a separate zero-shift test asserting that the traceless tensor result equals the raw tensor minus exactly one third of the spin-spin radial term.

- [ ] **Step 2: Run the focused matrix tests and verify RED**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_matrix_elements.jl")'`

Expected: `MethodError` for both spin-dependent potential types.

- [ ] **Step 3: Implement spin contractions**

Implement these two helpers, with `axes = (:x, :y, :z)` and `a != b` site validation handled by constructors:

```julia
function _spin_pair_element(bra, ket, i, j, alpha, beta)
    total = one(ComplexF64)
    for site in eachindex(bra.projections)
        if site == i
            total *= spin_element(bra, ket, site, alpha)
        elseif site == j
            total *= spin_element(bra, ket, site, beta)
        elseif bra.projections[site] != ket.projections[site]
            return 0.0 + 0.0im
        end
    end
    return total
end

function _spin_single_element(bra, ket, i, alpha)
    if !all(
            bra.projections[site] == ket.projections[site]
            for site in eachindex(bra.projections) if site != i
        )
        return 0.0 + 0.0im
    end
    return spin_element(bra, ket, i, alpha)
end

function _spin_dot_element(bra, ket, i, j)
    return sum(_spin_pair_element(bra, ket, i, j, alpha, alpha) for alpha in (:x, :y, :z))
end
```

Use a direct product-state loop rather than allocating Pauli Kronecker products.

- [ ] **Step 4: Implement the tensor method from Eqs. (38)-(40)**

For updated Gaussian data, calculate `q = vec(w' * Rprime * v / 2)`, `variance = (w' * Rprime * w)[1] / 2`, and the Cartesian coordinate moment `T[alpha, beta] = Mprime * (variance * (alpha == beta) + q[alpha] * q[beta])`. Contract it with `_spin_pair_element`. If `traceless`, subtract `spin_dot * Mprime * (3 * variance + dot(q, q)) / 3`. Multiply by `coefficient`.

- [ ] **Step 5: Implement the spin-orbit method from Eqs. (58), (60), and (61)**

Calculate the source's unscaled cross-derivative vector:

```julia
u = data.Rprime * data.v
left = vec(op.w' * u)
right = vec(op.w' * ket.orbital.s - op.w' * data.A * u)
cross_derivative = cross(left, right) * data.Mprime / 2
```

Convert this to orbital angular momentum with `orbital_L = -im * cross_derivative / 2`, then contract `orbital_L[alpha]` with `_spin_single_element(..., i, alpha) + _spin_single_element(..., j, alpha)`. Multiply by `coefficient`.

- [ ] **Step 6: Run the focused matrix tests and verify GREEN**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_matrix_elements.jl")'`

Expected: tensor central subtraction and Hermiticity checks pass; at least one spin-orbit matrix element has nonzero imaginary part before Hermitian pairing.

- [ ] **Step 7: Do not commit**

Leave all modifications unstaged.

## Task 4: Make Assembly, Eigenvectors, and Sampling Complex-Safe

**Files:**
- Modify: `src/hamiltonian.jl:4-256`
- Modify: `src/sampling.jl:21-50`
- Test: `test/test_hamiltonian.jl`
- Test: `test/test_sampling.jl`

**Interfaces:**
- Produces Hermitian `Matrix{T}` from an arbitrary `BasisSet{<:GaussianBase}` where `T` is the matrix-element type.
- Accepts complex Hermitian `H` and `S` in `solve_generalized_eigenproblem` and returns real eigenvalues with complex eigenvectors.

- [ ] **Step 1: Write failing complex-assembly and sampler tests**

Add the following to `test/test_hamiltonian.jl`:

```julia
@testset "Complex Hermitian assembly" begin
    H = ComplexF64[2 1im; -1im 2]
    S = Matrix{ComplexF64}(I, 2, 2)
    values, vectors = solve_generalized_eigenproblem(H, S)
    @test values ≈ [1.0, 3.0]
    @test eltype(vectors) == ComplexF64
    @test vectors' * S * vectors ≈ I
end
```

Add to `test/test_sampling.jl`:

```julia
@test size(generate_shift(:quasirandom, 1, 2, 0.5)) == (2, 3)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_hamiltonian.jl"); include("test/test_sampling.jl")'`

Expected: the eigensolver method rejects complex input and the sampled shift is a vector.

- [ ] **Step 3: Allocate typed Hermitian matrices**

Replace `Matrix{Float64}` allocation with first-element type inference. For a nonempty basis, compute `(1, 1)` first, allocate `Matrix{typeof(first)}`, write it, and for `i > j` set `matrix[j, i] = conj(value)`. Reject an empty basis at the public matrix-builder boundary.

Build each operator matrix independently, then use `reduce(+, matrices)` so Julia promotes real and complex operator-matrix element types without forcing `ComplexF64` for a real problem.

- [ ] **Step 4: Generalize the eigensolver without dropping imaginary parts**

Change the signatures to `AbstractMatrix{<:Number}`. Wrap inputs in `Hermitian((H + H') / 2)` and `Hermitian((S + S') / 2)`, retain `L'` in the back transformation, return `real.(evals), vecs`, and delete `real.(vecs)`.

- [ ] **Step 5: Generate supervector shifts**

Change `generate_shift` to request `3 * dim` quasi-random values and reshape the result:

```julia
return reshape(scale .* (2 .* u .- 1), dim, 3)
```

Update `build_rank0` to accept an `AbstractMatrix` shift and update `solve_ECG`'s `basis_fns`/history storage so the returned result preserves the inferred eigenvector type.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_hamiltonian.jl"); include("test/test_sampling.jl")'`

Expected: the complex eigenvector test and supervector sampler test pass with all existing cases adapted to matrix shifts.

- [ ] **Step 7: Do not commit**

Leave all modifications unstaged.

## Task 5: Split Utilities Without Adding a Plotting Dependency

**Files:**
- Create: `src/utils/solver_results.jl`
- Create: `src/utils/observables.jl`
- Create: `src/utils/plotting.jl`
- Modify: `src/FewBodyECG.jl:16-25`
- Delete: `src/utils.jl`
- Modify: `test/test_utils.jl`

**Interfaces:**
- Produces `SolverResults`, `ψ₀`, `convergence`, `correlation_function`, and `plot_correlation(plotter, result; kwargs...)`.
- Does not reference an undeclared `plot` global or introduce a package dependency.

- [ ] **Step 1: Write failing utility-layout tests**

Add this test to `test/test_utils.jl`:

```julia
@testset "Plotting adapter" begin
    result = create_mock_solver_results(n_basis = 2, dim = 1)
    received = Ref{Any}(nothing)
    plotter(x, y; kwargs...) = (received[] = (; x, y, kwargs); :plot)
    @test plot_correlation(plotter, result; npoints = 8, label = "density") == :plot
    @test length(received[].x) == 8
    @test received[].kwargs.label == "density"
end
```

- [ ] **Step 2: Run the utility tests and verify RED**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_utils.jl")'`

Expected: `plot_correlation` is undefined.

- [ ] **Step 3: Move utilities into focused files**

Move `SolverResults`, `ψ₀`, and `convergence` into `src/utils/solver_results.jl`; move `correlation_function` into `src/utils/observables.jl`; create this dependency-free adapter in `src/utils/plotting.jl`:

```julia
function plot_correlation(
        plotter, result::SolverResults;
        rmin::Real = 0.01, rmax::Real = 10.0, npoints::Int = 400,
        coord_index::Int = 1, normalize::Bool = true, kwargs...
    )
    r, density = correlation_function(
        result; rmin, rmax, npoints, coord_index, normalize,
    )
    return plotter(r, density; kwargs...)
end
```

Replace the direct `ψ` implementation, which calls an undeclared `plot`, with `plot_correlation`. Update module includes in dependency order and export `plot_correlation`; retain `ψ₀` spelling for compatibility.

- [ ] **Step 4: Update correlation-profile inputs for supervectors**

Build radial probe coordinates as `zeros(Float64, ncoordinates, 3)` and vary a documented axis (the third column) at `coord_index`. Keep this utility explicitly as a one-axis profile, not a rotationally averaged density, and continue to return data vectors.

- [ ] **Step 5: Run the utility tests and verify GREEN**

Run: `julia --project=. -e 'using Test, FewBodyECG; include("test/test_utils.jl")'`

Expected: helper tests and the plotting-adapter test pass without importing `Plots`.

- [ ] **Step 6: Do not commit**

Leave all modifications unstaged.

## Task 6: Final Regression, Documentation, and Formatting Gate

**Files:**
- Modify: `README.md` only if exported names are listed.
- Modify: `docs/src/API.md` and `docs/src/theory.md` to document supervectors, spin-product bases, deferred Yukawa-family operators, and `plot_correlation`.
- Modify: all touched source and test files only through Runic.

**Interfaces:**
- Documents the exact physical scope and the no-`Plots` utility design.

- [ ] **Step 1: Add public API and physics-scope documentation**

Document that `Rank0Gaussian(A, s)` uses `s::N x 3`, rank-1/rank-2 pre-factors must have zero shifts, `SpinGaussian` combines an orbital and product spin state, and Gaussian expansions are the supported analytic short-range strategy. Include one minimal spin-aware construction:

```julia
orbital = Rank0Gaussian(A, zeros(size(A, 1), 3))
basis = BasisSet([SpinGaussian(orbital, SpinState([up, down]))])
tensor = GaussianTensorPotential(1.0, 0.5, w, 1, 2; traceless = true)
H = build_hamiltonian_matrix(basis, [tensor])
```

- [ ] **Step 2: Apply Runic to changed Julia files**

Run: `runic --inplace src test`

Expected: changed Julia files are formatted in place.

- [ ] **Step 3: Verify formatting**

Run: `runic --check src test`

Expected: exit code 0 and no diff output.

- [ ] **Step 4: Run the complete regression suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: exit code 0. Record any Julia/Pkg resolver warning separately from test failures.

- [ ] **Step 5: Inspect the final diff and report quality findings**

Run: `git diff --check && git diff -- src test docs README.md Project.toml`

Expected: no whitespace errors; report all changed files, test evidence, deferred Yukawa family, compatibility break for shifted rank-1/rank-2 inputs, and the structural/code-quality review requested by the user.

- [ ] **Step 6: Do not commit**

Confirm no commit was created and leave all work for the user to review and commit.
