using Test
using LinearAlgebra
using FewBodyHamiltonians
using FewBodyECG
import FewBodyECG: _compute_overlap_element, _build_operator_matrix, _compute_matrix_element, normalized_overlap, is_linearly_independent


@testset "build_overlap_matrix" begin
    
    @testset "Size and symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.5 0.0; 0.0 1.5], [0.0, 0.0])
        g3 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.1, 0.1])
        
        basis = BasisSet([g1, g2, g3])
        S = build_overlap_matrix(basis)
        
        @test size(S) == (3, 3)
        @test issymmetric(S)
        @test all(isfinite, S)
    end
    
    
    @testset "Consistency" begin
        g1 = Rank0Gaussian([1.0;;], [0.0])
        g2 = Rank0Gaussian([2.0;;], [0.0])
        
        basis = BasisSet([g1, g2])
        S = build_overlap_matrix(basis)
        
        S11 = _compute_matrix_element(g1, g1)
        S12 = _compute_matrix_element(g1, g2)
        S22 = _compute_matrix_element(g2, g2)
        
        @test S[1,1] ≈ S11 rtol=1e-12
        @test S[1,2] ≈ S12 rtol=1e-12
        @test S[2,2] ≈ S22 rtol=1e-12
    end
end

@testset "build_hamiltonian_matrix" begin
    
    @testset "Size and symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.0, 0.0])
        basis = BasisSet([g1, g2])
        
        K = KineticOperator([0.5 0.0; 0.0 0.5])
        H = build_hamiltonian_matrix(basis, [K])
        
        @test size(H) == (2, 2)
        @test issymmetric(H)
        @test all(isfinite, H)
    end
    
    @testset "Additivity" begin
        # H(K+V) should equal H(K) + H(V)
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])
        
        K = KineticOperator([0.5 0.0; 0.0 0.5])
        V = CoulombOperator(-1.0, [1.0, 0.0])
        
        H_K = build_hamiltonian_matrix(basis, [K])
        H_V = build_hamiltonian_matrix(basis, [V])
        H_both = build_hamiltonian_matrix(basis, [K, V])
        
        @test H_both ≈ H_K + H_V rtol=1e-12
    end
    
    @testset "Empty operators" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])
        
        H = build_hamiltonian_matrix(basis, FewBodyHamiltonians.Operator[])
        
        @test H == zeros(1, 1)
    end
end


@testset "solve_generalized_eigenproblem" begin
    
    @testset "Basic solve" begin
        H = [2.0 0.5; 0.5 3.0]
        S = [1.0 0.1; 0.1 1.0]
        
        evals, evecs = solve_generalized_eigenproblem(H, S)
        
        @test length(evals) == 2
        @test size(evecs) == (2, 2)
        @test all(isfinite, evals)
        @test all(isfinite, evecs)
    end
    
    @testset "Eigenvalue equation H*v = λ*S*v" begin
        H = [3.0 0.5; 0.5 2.0]
        S = [1.0 0.2; 0.2 1.0]
        
        evals, evecs = solve_generalized_eigenproblem(H, S)
        
        # Check each eigenpair
        for i in 1:2
            λ = evals[i]
            v = evecs[:, i]
            residual = H * v - λ * S * v
            @test norm(residual) < 1e-9
        end
    end
    
    @testset "Requires positive definite S" begin
        H = [1.0 0.0; 0.0 1.0]
        S_bad = [-1.0 0.0; 0.0 1.0]
        
        @test_throws Exception solve_generalized_eigenproblem(H, S_bad)
    end
end


@testset "normalized_overlap" begin
        
    @testset "Symmetry" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([2.0 0.0; 0.0 2.0], [0.5, 0.5])
        
        overlap_12 = normalized_overlap(g1, g2)
        overlap_21 = normalized_overlap(g2, g1)
        
        @test overlap_12 ≈ overlap_21 rtol=1e-12
    end
    
    @testset "Range [0, 1]" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.5, 0.5])
        g3 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [5.0, 5.0])
        
        overlap_12 = normalized_overlap(g1, g2)
        overlap_13 = normalized_overlap(g1, g3)
        
        @test 0.0 <= overlap_12 <= 1.0 
        @test 0.0 <= overlap_13 <= 1.0 
    end
    
    @testset "Identical Gaussians" begin
        A = [1.5 0.0; 0.0 1.5]
        s = [0.3, 0.4]
        g1 = Rank0Gaussian(A, s)
        g2 = Rank0Gaussian(A, s)
        
        overlap = normalized_overlap(g1, g2)
        @test overlap ≈ 1.0 rtol=1e-10
    end
end

@testset "is_linearly_independent" begin
    
    @testset "Empty basis (always independent)" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        empty_basis = BasisSet(Rank0Gaussian[])
        
        @test is_linearly_independent(g, empty_basis; threshold=0.95)
    end
    
    
    @testset "Well-separated (should accept)" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [5.0, 5.0])
        
        basis = BasisSet([g1])
        
        @test is_linearly_independent(g2, basis; threshold=0.95)
    end
    
    @testset "Threshold behavior" begin
        g1 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        g2 = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.3, 0.3])
        
        basis = BasisSet([g1])
        overlap = normalized_overlap(g2, g1)
        
        @test is_linearly_independent(g2, basis; threshold=overlap + 0.01)
    end
    
    @testset "Invalid threshold" begin
        g = Rank0Gaussian([1.0 0.0; 0.0 1.0], [0.0, 0.0])
        basis = BasisSet([g])
        
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold=0.0)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold=1.0)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold=-0.5)
        @test_throws ArgumentError is_linearly_independent(g, basis; threshold=1.5)
    end
end
