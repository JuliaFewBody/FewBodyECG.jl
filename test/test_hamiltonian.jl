using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
import FewBodyECG: _compute_overlap_element, _build_operator_matrix, _compute_matrix_element

@testset "hamiltonian extra tests" begin

    A1 = reshape([1.0], 1, 1)
    A2 = reshape([1.0], 1, 1)
    A3 = reshape([1.0], 1, 1)
    g1 = Rank0Gaussian(A1, [10.0])
    g2 = Rank0Gaussian(A2, [20.0])
    g3 = Rank0Gaussian(A3, [30.0])
    basis3 = BasisSet{Rank0Gaussian}([g1, g2, g3])

    @eval FewBodyECG begin
        function _compute_matrix_element(b::Rank0Gaussian, k::Rank0Gaussian)
            return (b.s[1] + k.s[1]) / 10.0
        end
        function _compute_matrix_element(b::Rank0Gaussian, k::Rank0Gaussian, op::FewBodyHamiltonians.Operator)
            return (b.s[1] * k.s[1]) / 10.0
        end
    end

    S3 = build_overlap_matrix(basis3)
    manualS = zeros(Float64, 3, 3)
    for i in 1:3, j in 1:3
        manualS[i, j] = _compute_overlap_element(basis3.functions[i], basis3.functions[j])
    end
    @test S3 == manualS
    @test issymmetric(S3)
    @test eltype(S3) == Float64


end
