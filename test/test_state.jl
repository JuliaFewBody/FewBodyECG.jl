using Test
using LinearAlgebra
using FewBodyECG
using FewBodyECG: BasisState, nfuns, commit!, rebuild_without,
    _candidate_columns, _draw_candidate!, _solution_basis_state,
    ConvergenceReport, Solution, SVM, StageResult

# hydrogen-like fixture
ops = Operators([1.0e15, 1.0], [+1.0, -1.0]); ops += "Kinetic"; ops += "Coulomb"
terms = ops.terms
w_list = [op.w for op in terms if op isa CoulombOperator]
d = length(w_list[1])

@testset "BasisState growth and caches" begin
    st = BasisState()
    @test nfuns(st) == 0
    for _ in 1:6
        cand = _draw_candidate!(st, 1.0, FewBodyECG.HaltonSample(), w_list, d)
        cols = _candidate_columns(cand, st.basis, terms)
        cols === nothing && continue
        commit!(st, cand, cols)
    end
    k = nfuns(st)
    @test k ≥ 4
    # caches match direct assembly
    bs = BasisSet(st.basis)
    @test st.S ≈ build_overlap_matrix(bs) atol = 1.0e-12
    @test st.H ≈ build_hamiltonian_matrix(bs, terms) atol = 1.0e-12
    # eigensolver state consistent with caches
    λ = eigen(Symmetric(st.H), Symmetric(st.S)).values
    @test minimum(st.eig.ε) ≈ minimum(λ) rtol = 1.0e-8

    st2 = BasisState(copy(st.basis), terms)     # rebuild from functions
    @test st2.S ≈ st.S atol = 1.0e-12
    @test minimum(st2.eig.ε) ≈ minimum(st.eig.ε) rtol = 1.0e-10

    r = rebuild_without(st, 2)
    @test nfuns(r) == k - 1
    idx = setdiff(1:k, 2)
    @test r.S ≈ st.S[idx, idx] atol = 1.0e-12
    λr = eigen(Symmetric(st.H[idx, idx]), Symmetric(st.S[idx, idx])).values
    @test minimum(r.eig.ε) ≈ minimum(λr) rtol = 1.0e-8
    @test r.draw == st.draw                      # QMC stream carried over
end

@testset "rebuild_without boundaries and warm start" begin
    st = BasisState()
    for _ in 1:8
        cand = _draw_candidate!(st, 1.0, FewBodyECG.HaltonSample(), w_list, d)
        cols = _candidate_columns(cand, st.basis, terms)
        cols === nothing && continue
        commit!(st, cand, cols)
    end
    k = nfuns(st)
    @test k ≥ 3
    for i in (1, k)                       # boundary removals
        r = rebuild_without(st, i)
        idx = setdiff(1:k, i)
        @test r.S ≈ st.S[idx, idx] atol = 1.0e-12
        @test r.H ≈ st.H[idx, idx] atol = 1.0e-12
    end
    st1 = BasisState()                    # k = 1 → empty state after removal
    cand1 = _draw_candidate!(st1, 1.0, FewBodyECG.HaltonSample(), w_list, d)
    cols1 = _candidate_columns(cand1, st1.basis, terms)
    @test cols1 !== nothing
    commit!(st1, cand1, cols1)
    @test nfuns(rebuild_without(st1, 1)) == 0

    rep = ConvergenceReport(true, :saturation, 0.0, 1.0e-4, 20, nothing, 1.0, String[])
    sol = Solution(
        copy(st.eig.ε), BasisSet(copy(st.basis)), FewBodyECG.coefficients(st.eig),
        Operator[terms...], 1,
        [StageResult(SVM(k), copy(st.E_hist), rep)], rep
    )
    ws = _solution_basis_state(sol, terms)
    @test nfuns(ws) == k
    @test ws.S ≈ st.S atol = 1.0e-10
    @test ws.H ≈ st.H atol = 1.0e-10
    @test minimum(ws.eig.ε) ≈ minimum(st.eig.ε) rtol = 1.0e-10
end
