# Harmonic-Oscillator Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the three-dimensional relative `OscillatorPotential` with the first three radial harmonic-oscillator energies and provide a plotted runnable example.

**Architecture:** The package physics needs no new operator or solver abstraction. A dedicated deterministic regression test builds a fixed rank-0 ECG basis, while one example independently evaluates the resulting eigenvectors along a radial line and plots their normalized radial probabilities.

**Tech Stack:** Julia, FewBodyECG, FewBodyHamiltonians, LinearAlgebra, Test, Plots (example-only).

## Global Constraints

- Model `H = -∇² / 2 + r² / 2` in one relative coordinate in physical 3D.
- Use `KineticOperator([1 / 2;;])`, `OscillatorPotential(1 / 2, [1])`, and 15 zero-shift rank-0 ECGs with exponents `10 .^ range(-1, 1, length = 15)`.
- Compare the `ℓ = 0` states to `[3 / 2, 7 / 2, 11 / 2]` with `atol = 1e-5`.
- Keep `Plots` confined to `Examples/`; do not add a core plotting dependency.
- Format source and tests with Runic.
- Do not commit, stage, push, merge, or discard changes; the user owns Git operations.

---

### Task 1: Add the deterministic oscillator-spectrum regression

**Files:**
- Create: `test/test_harmonic_oscillator.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `Rank0Gaussian(A, s)`, `BasisSet`, `KineticOperator`, `OscillatorPotential`, `build_overlap_matrix`, `build_hamiltonian_matrix`, and `solve_generalized_eigenproblem`.
- Produces: A regression that fails when the rank-0 oscillator spectrum ceases to reproduce the first three exact `ℓ = 0` energies.

- [x] **Step 1: Write the spectrum regression**

```julia
using Test
using FewBodyECG
using FewBodyHamiltonians

@testset "3D relative harmonic oscillator" begin
    exponents = 10 .^ range(-1, 1, length = 15)
    basis = BasisSet(Rank0Gaussian[
        Rank0Gaussian([α;;], zeros(1, 3)) for α in exponents
    ])
    operators = Operator[
        KineticOperator([1 / 2;;]),
        OscillatorPotential(1 / 2, [1]),
    ]

    H = build_hamiltonian_matrix(basis, operators)
    S = build_overlap_matrix(basis)
    energies, _ = solve_generalized_eigenproblem(H, S)

    @test energies[1:3] ≈ [3 / 2, 7 / 2, 11 / 2] atol = 1e-5
end
```

- [x] **Step 2: Register and run the package tests**

Add `include("test_harmonic_oscillator.jl")` after the matrix-element tests in
`test/runtests.jl`, then run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: the new `3D relative harmonic oscillator` test passes and the full
suite remains green.

### Task 2: Add the radial-state example and figure

**Files:**
- Create: `Examples/HarmonicOscillator.jl`

**Interfaces:**
- Consumes: the same fixed basis and operators as Task 1; `ψ₀(coordinates, coefficients, basis_functions)` evaluates a rank-0 radial wavefunction.
- Produces: a console energy comparison and a `Plots.jl` figure containing the normalized radial probabilities for the first three states.

- [x] **Step 1: Write the example**

```julia
using FewBodyECG
using FewBodyHamiltonians
using Plots

exponents = 10 .^ range(-1, 1, length = 15)
basis_functions = Rank0Gaussian[
    Rank0Gaussian([α;;], zeros(1, 3)) for α in exponents
]
basis = BasisSet(basis_functions)
operators = Operator[
    KineticOperator([1 / 2;;]),
    OscillatorPotential(1 / 2, [1]),
]

energies, eigenvectors = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis, operators),
    build_overlap_matrix(basis),
)
exact_energies = [3 / 2, 7 / 2, 11 / 2]

for state in 1:3
    println("n_r = $(state - 1): E = $(energies[state]), exact = $(exact_energies[state])")
end

function radial_probability(r, coefficients)
    coordinates = reshape([0.0, 0.0, r], 1, 3)
    return r^2 * abs2(ψ₀(coordinates, coefficients, basis_functions))
end

radius = range(0.0, 6.0, length = 600)
radial_states = plot(
    xlabel = "r",
    ylabel = "normalized r²|ψ(r)|²",
    title = "3D harmonic-oscillator radial states",
)
for state in 1:3
    density = radial_probability.(radius, Ref(eigenvectors[:, state]))
    density ./= sum((density[i] + density[i + 1]) * (radius[i + 1] - radius[i]) / 2 for i in 1:(length(radius) - 1))
    plot!(radial_states, radius, density, label = "n_r = $(state - 1)")
end
display(radial_states)
```

- [x] **Step 2: Run the example headlessly**

```bash
GKSwstype=100 julia --project=docs Examples/HarmonicOscillator.jl
```

Expected: it prints three energy comparisons and renders the figure without
creating a generated repository artifact. Inspect the plot: the curves have
zero, one, and two interior radial nodes respectively.

### Task 3: Format and verify the handoff

**Files:**
- Modify: files from Tasks 1–2 only if formatting or verification exposes a defect.

**Interfaces:**
- Consumes: the test and example added above.
- Produces: a formatted, verified, unstaged implementation.

- [x] **Step 1: Apply and check Runic formatting**

```bash
runic --inplace test/test_harmonic_oscillator.jl Examples/HarmonicOscillator.jl
runic --check src test Examples
```

Expected: the check exits successfully.

- [x] **Step 2: Run final verification**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
git diff --check
git status --short
```

Expected: all tests pass, the diff has no whitespace errors, and all work is
left uncommitted for the user.
