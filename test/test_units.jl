using Test
using RecipesBase
using FewBodyECG

@testset "Units" begin
    @testset "Operators presets and custom units" begin
        @test Operators().units == UnitSystem("Ha", "bohr")
        @test Operators([1.0, 1.0]).units == UnitSystem("Ha", "bohr")
        @test Operators([1.0, 1.0], [1.0, -1.0]; units = :nuclear).units == UnitSystem("MeV", "fm")
        gev = UnitSystem("GeV", "GeV⁻¹")
        @test Operators([1.8, 1.8]; units = gev).units == gev
        @test_throws "unknown unit system :cgs; use one of :atomic, :nuclear" Operators(; units = :cgs)

        ops = Operators([1.0, 1.0]; units = :nuclear)
        ops += "Kinetic"
        @test (ops + ("Gaussian", 1, 2, -10.0, 0.5)).units == ops.units
        @test occursin("(MeV, fm)", sprint(show, ops))
        @test sprint(show, gev) == "UnitSystem(\"GeV\", \"GeV⁻¹\")"

        nuc = Operators([1.0, 1.0], [1.0, 1.0]; units = :nuclear)
        @test_throws "\"Coulomb\" uses couplings q_i q_j, which assumes atomic units" nuc + "Coulomb"
        @test length(nuc + ("Coulomb", 1, 2, 1.44)) == 1
    end

    @testset "Solution carries and displays units" begin
        ops = Operators([1.0, 1.0]; units = :nuclear)
        ops += "Kinetic"
        ops += ("Gaussian", 1, 2, -10.0, 0.5)
        sol = solve(ops, SVM(basis = 6, candidates = 4, scale = 1.0); window = 2)
        @test sol.units == UnitSystem("MeV", "fm")
        out = sprint(show, MIME"text/plain"(), sol)
        @test occursin(" MeV    (variational upper bound)", out)
        @test !occursin("Ha", out)

        attrs = Dict{Symbol, Any}()
        RecipesBase.apply_recipe(attrs, sol)
        @test attrs[:yguide] == "E (MeV)"

        # units survive every stage of a pipeline
        @test solve(ops, SVM(4; scale = 1.0) → Refine(1; scale = 1.0)).units == sol.units
        @test solve(ops, GVM(3; scale = 1.0)).units == sol.units

        @test_throws "units are set by Operators(...; units = ...), not by solve" solve(
            ops, SVM(basis = 3, scale = 1.0); units = :atomic
        )
    end

    @testset "Raw operator vectors and warm starts" begin
        ops = Operators([1.0, 1.0]; units = :nuclear)
        ops += "Kinetic"
        ops += ("Gaussian", 1, 2, -10.0, 0.5)
        raw = solve(ops.terms, SVM(basis = 4, scale = 1.0))
        @test raw.units == UnitSystem("Ha", "bohr")
        @test solve(ops.terms, SVM(basis = 4, scale = 1.0); units = :nuclear).units == ops.units

        @test_throws "init was solved in UnitSystem(\"Ha\", \"bohr\"), but this problem uses UnitSystem(\"MeV\", \"fm\")" solve(
            ops, Refine(1; scale = 1.0); init = raw
        )
    end
end
