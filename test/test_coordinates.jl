using Test
using FewBodyECG
using LinearAlgebra
import FewBodyECG: _jacobi_transform, _generate_A_matrix, _shift_vectors, _transform_coordinates, _inverse_transform_coordinates

@testset "Coordinates Module Tests" begin
    @testset "_jacobi_transform" begin
        # Test with simple equal masses
        masses = [1.0, 1.0, 1.0]
        J, U = _jacobi_transform(masses)

        @test size(J) == (2, 3)
        @test size(U) == (3, 2)
        @test J * U ≈ I(2) atol = 1.0e-10

        # Test with different masses
        masses = [1.0, 2.0, 3.0]
        J, U = _jacobi_transform(masses)

        @test size(J) == (2, 3)
        @test size(U) == (3, 2)
        @test J * U ≈ I(2) atol = 1.0e-10

        # Test with exactly two masses
        masses = [1.0, 2.0]
        J, U = _jacobi_transform(masses)

        @test size(J) == (1, 2)
        @test size(U) == (2, 1)
        @test J * U ≈ [1.0] atol = 1.0e-10

        # Test error for fewer than two masses
        @test_throws AssertionError _jacobi_transform([1.0])
    end


    @testset "_transform_coordinates / _inverse_transform_coordinates" begin
        masses = [1.0, 2.0, 3.0]
        J, U = _jacobi_transform(masses)
        r = [1.0, 2.0, 3.0]

        x = _transform_coordinates(J, r)
        @test size(x) == (2,)

        r_back = _inverse_transform_coordinates(U, x)
        @test size(r_back) == (3,)

        # Instead of round-trip r → x → r_back, test projection recovery
        x_back = _transform_coordinates(J, r_back)
        @test x_back ≈ x atol = 1.0e-10

        # Error case
        @test_throws AssertionError _transform_coordinates(J, [1.0, 2.0])
        @test_throws AssertionError _inverse_transform_coordinates(U, [1.0])
    end
end

@testset "Additional Jacobi Transform Tests" begin
    @testset "Numeric values for equal masses" begin
        masses = [1.0, 1.0, 1.0]
        J, U = _jacobi_transform(masses)

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
        J, U = _jacobi_transform(masses)

        μ = sqrt(2.0 / 3.0)
        @test size(J) == (1, 2)
        @test isapprox(J[1, 1], μ; atol = 1.0e-12)
        @test isapprox(J[1, 2], -μ; atol = 1.0e-12)
    end

    @testset "Pseudoinverse (Moore–Penrose) properties" begin
        masses = [1.3, 2.5, 0.7, 4.1]
        J, U = _jacobi_transform(masses)

        # J * U should act like the identity on the reduced space
        Ired = I(size(J, 1))
        @test isapprox(J * U, Ired; atol = 1.0e-10)

        # Moore-Penrose conditions
        @test isapprox(J * U * J, J; atol = 1.0e-10)
        @test isapprox(U * J * U, U; atol = 1.0e-10)

        # Symmetry conditions
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

@testset "Dispatch and Type Behaviour Tests" begin
    @testset "Method signatures enforce Float64 matrices" begin
        Jf32 = zeros(Float32, 2, 3)
        r = [1.0, 2.0, 3.0]
        @test_throws MethodError _transform_coordinates(Jf32, r)

        Uf32 = zeros(Float32, 3, 2)
        x = [1.0, 2.0]
        @test_throws MethodError _inverse_transform_coordinates(Uf32, x)
    end


    @testset "Method errors on _jacobi_transform with non-Float64 masses" begin
        @test_throws MethodError _jacobi_transform([1, 2, 3])             # Integer vector
        @test_throws MethodError _jacobi_transform([1.0f0, 2.0f0, 3.0f0]) # Float32 vector
    end
end
