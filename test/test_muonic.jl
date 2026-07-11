using Test
using FewBodyECG

# Muonic molecular ions: three-body Coulomb systems reproducing the Suzuki–Varga
# Table 8.1 K = 200 benchmark energies.  A moderate basis is used here to keep
# the suite fast (the example shows the few-mHa agreement at basis = 300).
@testset "Muonic molecular ions vs Suzuki–Varga" begin
    mμ, md, mt = 206.7686, 3670.481, 5496.918

    solve_ion(masses) = solve(
        (o = Operators(masses, [+1.0, +1.0, -1.0]); o += "Kinetic"; o += "Coulomb"; o),
        SVM(basis = 200, candidates = 40, scale = 0.02); tol = 1.0e-4, window = 15,
    ).E₀

    E_dt = solve_ion([md, mt, mμ])
    E_tt = solve_ion([mt, mt, mμ])

    # reproduce the benchmarks from above (variational upper bound) to <40 mHa
    @test E_dt ≈ -111.36444 atol = 0.04
    @test E_dt > -111.36444              # variational upper bound
    @test E_tt ≈ -112.973 atol = 0.04
    @test E_tt > -112.973                # variational upper bound

    # heavier nuclei bind deeper: ttμ below dtμ
    @test E_tt < E_dt
end
