using Test
using FewBodyECG
using FewBodyDB: db

# Muonic molecular ions: three-body Coulomb systems against the Suzuki–Varga
# benchmark energies in FewBodyDB.
@testset "Muonic molecular ions vs Suzuki–Varga" begin
    mμ, md, mt = 206.7686, 3670.481, 5496.918

    solve_ion(masses) = solve(
        (o = Operators(masses, [+1.0, +1.0, -1.0]); o += "Kinetic"; o += "Coulomb"; o),
        SVM(basis = 200, candidates = 40, scale = 0.02); tol = 1.0e-4, window = 15,
    ).E₀

    E_dt = solve_ion([md, mt, mμ])
    E_tt = solve_ion([mt, mt, mμ])
    ref_dt = db(:Suzuki2003Jul, "tdμ", :energy, "¹Sᵉ").value
    ref_tt = db(:Suzuki2003Jul, "ttμ", :energy, "¹Sᵉ").value

    @test E_dt ≈ ref_dt atol = 0.04
    @test E_dt > ref_dt                  # variational upper bound
    @test E_tt ≈ ref_tt atol = 0.04
    @test E_tt > ref_tt                  # variational upper bound

    # heavier nuclei bind deeper: ttμ below dtμ
    @test E_tt < E_dt
end
