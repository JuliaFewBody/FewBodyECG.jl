using Test
using FewBodyECG
using OptimKit: LBFGS

_adaptive_lbfgs(maxiter) = LBFGS(; maxiter, gradtol = 1.0e-6, verbosity = 0, ls_verbosity = 0)

coulomb_system(masses, charges) = (o = Operators(masses, charges); o += "Kinetic"; o += "Coulomb"; o)

@testset "scale = :adaptive" begin
    @test SVM().scale === Refine().scale === DynamicGVM().scale === :adaptive

    H = coulomb_system([1.0e15, 1.0], [+1.0, -1.0])
    sol = solve(H, SVM(basis = 20, candidates = 20, scale = 1.0))
    # hydrogen 1s: ⟨r²⟩ = 3 bohr²
    @test FewBodyECG._rms_pair_max(sol, H.masses) ≈ sqrt(3) rtol = 1.0e-3

    @testset "pilot on a cold start" begin
        sol = solve(H, SVM(basis = 20, candidates = 20))
        note = only(filter(n -> startswith(n, "scale = :adaptive"), sol.convergence.notes))
        @test occursin("× rms pair distance 1.73 bohr, from the lowest-energy of 2 pilot runs", note)
        @test sol.E₀ ≈ -0.5 atol = 1.0e-4
        @test sol.draws == 20 * 20                     # pilot draws are not counted
    end

    @testset "warm starts measure the initial basis" begin
        ref = solve(H, Refine(1; candidates = 5); init = sol)
        @test any(n -> occursin("bohr of the initial basis", n), ref.convergence.notes)
        dyn = solve(H, DynamicGVM(basis = 22, candidates = 2, optimizer = _adaptive_lbfgs(10)); init = sol)
        @test any(n -> occursin("bohr of the initial basis", n), dyn.convergence.notes)
        gvm = solve(H, GVM(4; optimizer = _adaptive_lbfgs(20)))
        @test any(n -> occursin("pilot run", n), gvm.convergence.notes)
    end

    @testset "pilot size comes from the lowest-energy pilot" begin
        # tdμ is ≈ 0.017 bohr across.  Re-centered pilots are less converged and
        # overestimate its size (0.023, then 0.38 bohr); only the best one is used.
        tdμ = coulomb_system([3670.481, 5496.918, 206.7686], [+1.0, +1.0, -1.0])
        ctx = FewBodyECG._ctx(
            tdμ.terms, tdμ.masses;
            state = 1, tol = 1.0e-4, window = 20, verbose = false, units = tdμ.units, init = nothing
        )
        scale, note = FewBodyECG._adaptive_scale(ctx, nothing)
        @test 0.008 < scale < 0.016
        @test occursin("lowest-energy of", note)
    end

    @testset "beats the mass-based :auto where 1/√m is a poor size estimate" begin
        # Ps⁻ is about 10 bohr across; :auto gives scale = 1.
        psm = coulomb_system([1.0, 1.0, 1.0], [+1.0, -1.0, -1.0])
        E(scale) = solve(psm, SVM(; basis = 40, candidates = 20, scale)).E₀
        @test E(:adaptive) < E(:auto) - 1.0e-2
    end

    @test_throws "scale = :adaptive requires Operators(masses[, charges]); pass a numeric scale" solve(
        H.terms, SVM(basis = 5)
    )
    @test_throws "unknown scale :big; use :adaptive, :auto or a number" solve(H, SVM(basis = 5, scale = :big))
end
