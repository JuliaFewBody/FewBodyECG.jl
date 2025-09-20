using Test
using LinearAlgebra

using FewBodyECG
import FewBodyECG: _compute_overlap_element

@testset "Hamiltonian / overlap helpers" begin

    @testset "compute_overlap_element for Rank0Gaussian" begin
        A = rand(2, 2)
        B = rand(2, 2)

        bra = Rank0Gaussian(A)
        ket = Rank0Gaussian(B)

        val = _compute_overlap_element(bra, ket)

        R = inv(A + B)
        n = length(R)
        expected = (π^n / det(A + B))^(3 / 2)

        @test isapprox(val, expected; atol = 1.0e-10)
    end


end
