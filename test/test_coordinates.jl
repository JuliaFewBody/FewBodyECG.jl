using Test
using FewBodyECG
using LinearAlgebra
using OffsetArrays
import FewBodyECG: jacobi_transform, _transform_coordinates, _inverse_transform_coordinates
using FewBodyECG: Λ

@testset "Coordinates Module Tests" begin
    @testset "jacobi_transform" begin
        masses = [1.0, 1.0, 1.0]
        J, U = jacobi_transform(masses)

        @test size(J) == (2, 3)
        @test size(U) == (3, 2)
        @test J * U ≈ I(2) atol = 1.0e-10

        masses = [1.0, 2.0, 3.0]
        J, U = jacobi_transform(masses)

        @test size(J) == (2, 3)
        @test size(U) == (3, 2)
        @test J * U ≈ I(2) atol = 1.0e-10

        masses = [1.0, 2.0]
        J, U = jacobi_transform(masses)

        @test size(J) == (1, 2)
        @test size(U) == (2, 1)
        @test J * U ≈ [1.0] atol = 1.0e-10

        @test_throws ArgumentError jacobi_transform([1.0])
        @test_throws "at least two masses are required" jacobi_transform([1.0])
    end


    @testset "_transform_coordinates / _inverse_transform_coordinates" begin
        masses = [1.0, 2.0, 3.0]
        J, U = jacobi_transform(masses)
        r = [1.0, 2.0, 3.0]

        x = _transform_coordinates(J, r)
        @test size(x) == (2,)

        r_back = _inverse_transform_coordinates(U, x)
        @test size(r_back) == (3,)

        x_back = _transform_coordinates(J, r_back)
        @test x_back ≈ x atol = 1.0e-10

        @test_throws DimensionMismatch _transform_coordinates(J, [1.0, 2.0])
        @test_throws "columns of J" _transform_coordinates(J, [1.0, 2.0])
        @test_throws DimensionMismatch _inverse_transform_coordinates(U, [1.0])
        @test_throws "columns of U" _inverse_transform_coordinates(U, [1.0])
    end
end

@testset "Additional Jacobi Transform Tests" begin
    @testset "Numeric values for equal masses" begin
        masses = [1.0, 1.0, 1.0]
        J, U = jacobi_transform(masses)

        μ1 = 1 / sqrt(2)
        μ2 = sqrt(2 / 3)

        @test size(J) == (2, 3)
        @test isapprox(J[1, 1], μ1; atol = 1.0e-12)
        @test isapprox(J[1, 2], -μ1; atol = 1.0e-12)
        @test isapprox(J[1, 3], 0.0; atol = 1.0e-12)

        @test isapprox(J[2, 1], μ2 / 2; atol = 1.0e-12)
        @test isapprox(J[2, 2], μ2 / 2; atol = 1.0e-12)
        @test isapprox(J[2, 3], -μ2; atol = 1.0e-12)
    end

    @testset "Numeric values for two masses" begin
        masses = [1.0, 2.0]
        J, U = jacobi_transform(masses)

        μ = sqrt(2.0 / 3.0)
        @test size(J) == (1, 2)
        @test isapprox(J[1, 1], μ; atol = 1.0e-12)
        @test isapprox(J[1, 2], -μ; atol = 1.0e-12)
    end

    @testset "Pseudoinverse (Moore–Penrose) properties" begin
        masses = [1.3, 2.5, 0.7, 4.1]
        J, U = jacobi_transform(masses)

        Ired = I(size(J, 1))
        @test isapprox(J * U, Ired; atol = 1.0e-10)

        # Moore-Penrose conditions
        @test isapprox(J * U * J, J; atol = 1.0e-10)
        @test isapprox(U * J * U, U; atol = 1.0e-10)

        @test isapprox((J * U)', J * U; atol = 1.0e-10)
        @test isapprox((U * J)', U * J; atol = 1.0e-10)
    end
end

@testset "Lambda and KineticOperator Tests" begin
    @testset "Λ for equal masses (analytic)" begin
        masses = [1.0, 1.0, 1.0]
        L = Λ(masses)
        @test size(L) == (2, 2)
        @test issymmetric(L)
        @test isapprox(Matrix(L), 0.5 * Matrix(I(2)); atol = 1.0e-12)
    end

    @testset "Λ for two masses (analytic)" begin
        masses = [1.0, 2.0]
        L = Λ(masses)
        @test size(L) == (1, 1)
        @test issymmetric(L)
        @test isapprox(L[1, 1], 0.5; atol = 1.0e-12)
    end

    @testset "Λ general properties" begin
        masses = [1.3, 2.5, 0.7, 4.1]
        L = Λ(masses)
        @test issymmetric(L)
        # positive semidefinite (numerical tolerance)
        vals = eigen(Symmetric(Matrix(L))).values
        @test minimum(vals) >= -1.0e-12
    end


end

@testset "Argument types" begin
    masses = [1.0, 2.0, 3.0]
    J, U = jacobi_transform(masses)

    @testset "jacobi_transform and Λ accept any real vector" begin
        for m in ([1, 2, 3], Float32[1, 2, 3], 1:3, view([0.0, 1.0, 2.0, 3.0], 2:4))
            Jm, Um = jacobi_transform(m)
            @test Jm isa Matrix{Float64} && Um isa Matrix{Float64}
            @test Jm ≈ J
            @test Um ≈ U
            @test Λ(m) ≈ Λ(masses)
            @test KineticOperator(m).K ≈ Λ(masses)
        end
        @test_throws ArgumentError jacobi_transform(OffsetArray(masses, 0:2))
    end

    @testset "coordinate transforms accept any real matrix" begin
        r = [1.0, 2.0, 3.0]
        @test _transform_coordinates(Float32.(J), r) ≈ J * r rtol = 1.0e-6
        @test _inverse_transform_coordinates(Float32.(U), J * r) ≈ U * (J * r) rtol = 1.0e-6
        @test_throws DimensionMismatch _transform_coordinates(J, OffsetArray(r, 0:2))
    end
end
