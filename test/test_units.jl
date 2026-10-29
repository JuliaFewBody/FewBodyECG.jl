using Test
using RecipesBase
using FewBodyECG

@testset "Units" begin
    @testset "Kinetic presets and custom units" begin
        @test Kinetic([1.0, 1.0]).units == UnitSystem("Ha", "bohr")
        @test Kinetic([1.0, 1.0]; units = :nuclear).units == UnitSystem("MeV", "fm")
        gev = UnitSystem("GeV", "GeV⁻¹")
        @test Kinetic([1.8, 1.8]; units = gev).units == gev
        @test_throws "unknown unit system :cgs; use one of :atomic, :nuclear" Kinetic([1.0, 1.0]; units = :cgs)

        H = Kinetic([1.0, 1.0]; units = :nuclear) - 10.0 * Gaussian(1, 2; exponent = 0.5)
        @test occursin("(2 particles; MeV, fm)", sprint(show, MIME"text/plain"(), H))
        @test sprint(show, gev) == "UnitSystem(\"GeV\", \"GeV⁻¹\")"

        nuc = Kinetic([1.0, 1.0]; units = :nuclear) + Coulomb([1.0, 1.0])
        @test_throws "Coulomb(charges) uses couplings qᵢqⱼ, which assumes atomic units" solve(
            nuc, SVM(basis = 3, scale = 1.0)
        )
        explicit = Kinetic([1.0, 1.0]; units = :nuclear) + Coulomb(1, 2; coefficient = 1.44)
        @test length(first(FewBodyECG._to_jacobi(explicit))) == 2
    end

    @testset "Solution carries and displays units" begin
        ops = Kinetic([1.0, 1.0]; units = :nuclear) - 10.0 * Gaussian(1, 2; exponent = 0.5)
        sol = solve(ops, SVM(basis = 6, candidates = 4, scale = 1.0); window = 2)
        @test sol.units == UnitSystem("MeV", "fm")
        out = sprint(show, MIME"text/plain"(), sol)
        @test occursin(" MeV    (variational upper bound)", out)
        @test !occursin(r"\bHa\b", out)

        attrs = Dict{Symbol, Any}()
        RecipesBase.apply_recipe(attrs, sol)
        @test attrs[:yguide] == "E (MeV)"

        # units survive every stage of a pipeline
        @test solve(ops, SVM(4; scale = 1.0) → Refine(1; scale = 1.0)).units == sol.units
        @test solve(ops, GVM(3; scale = 1.0)).units == sol.units

        @test_throws "units are set by Kinetic(masses; units = ...), not by solve" solve(
            ops, SVM(basis = 3, scale = 1.0); units = :atomic
        )
    end

    @testset "Raw operator vectors and warm starts" begin
        ops = Kinetic([1.0, 1.0]; units = :nuclear) - 10.0 * Gaussian(1, 2; exponent = 0.5)
        terms = first(FewBodyECG._to_jacobi(ops))
        raw = solve(terms, SVM(basis = 4, scale = 1.0))
        @test raw.units == UnitSystem("Ha", "bohr")
        @test solve(terms, SVM(basis = 4, scale = 1.0); units = :nuclear).units == UnitSystem("MeV", "fm")

        @test_throws "init was solved in UnitSystem(\"Ha\", \"bohr\"), but this problem uses UnitSystem(\"MeV\", \"fm\")" solve(
            ops, Refine(1; scale = 1.0); init = raw
        )
    end
end
