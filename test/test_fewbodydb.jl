using Test
using FewBodyECG
using FewBodyDB: db

# Ground states of three-body Coulomb systems against the benchmark energies
# collected in FewBodyDB.  Each system checks agreement within `atol` and the
# variational upper bound E₀ > E_ref.
@testset "Three-body Coulomb systems vs FewBodyDB" begin
    function coulomb(masses, charges)
        ops = Operators(masses, charges)
        ops += "Kinetic"
        ops += "Coulomb"
        return ops
    end
    mₚ, m_d = 1836.15267343, 3670.48296788

    @testset "∞He ¹Sᵉ (Suzuki–Varga)" begin
        ref = db(:Suzuki2003Jul, "∞He", :energy, "¹Sᵉ").value
        ops = coulomb([1.0e15, 1.0, 1.0], [+2.0, -1.0, -1.0])
        E = solve(ops, SVM(basis = 100, candidates = 25, scale = 1.0)).E₀
        @test E ≈ ref atol = 1.0e-3
        @test E > ref
    end

    @testset "∞H⁻ ¹Sᵉ (Suzuki–Varga)" begin
        ref = db(:Suzuki2003Jul, "∞H⁻", :energy, "¹Sᵉ").value
        ops = coulomb([1.0e15, 1.0, 1.0], [+1.0, -1.0, -1.0])
        E = solve(ops, SVM(basis = 100, candidates = 25, scale = 4.0)).E₀
        @test E ≈ ref atol = 1.0e-3
        @test E > ref
    end

    @testset "Ps⁻ ¹Sᵉ (Suzuki–Varga)" begin
        ref = db(:Suzuki2003Jul, "Ps⁻", :energy, "¹Sᵉ").value
        ops = coulomb([1.0, 1.0, 1.0], [+1.0, -1.0, -1.0])
        E = solve(ops, SVM(basis = 100, candidates = 25, scale = 4.0)).E₀
        @test E ≈ ref atol = 1.0e-3
        @test E > ref
    end

    @testset "HD⁺ (J = 0, v = 0) (Bubin et al.)" begin
        ref = db(:Bubin2005Jan, "HD⁺", :energy, (J = 0, v = 0)).value
        ops = coulomb([mₚ, m_d, 1.0], [+1.0, +1.0, -1.0])
        E = solve(ops, SVM(basis = 150, candidates = 25, scale = 1.0)).E₀
        @test E ≈ ref atol = 5.0e-3
        @test E > ref
    end
end
