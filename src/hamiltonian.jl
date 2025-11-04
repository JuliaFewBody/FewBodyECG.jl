using FewBodyHamiltonians
using LinearAlgebra

function _compute_overlap_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    _compute_matrix_element(bra, ket)
end

function build_overlap_matrix(basis::BasisSet{<:GaussianBase})
    n = length(basis.functions)
    S = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_overlap_element(basis.functions[i]::Rank0Gaussian, basis.functions[j]::Rank0Gaussian)
        S[i, j] = val
        S[j, i] = val
    end
    S
end

function _build_operator_matrix(basis::BasisSet{<:GaussianBase}, op::FewBodyHamiltonians.Operator)
    n = length(basis.functions)
    H = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:i
        val = _compute_matrix_element(basis.functions[i]::Rank0Gaussian, basis.functions[j]::Rank0Gaussian, op)
        H[i, j] = val
        H[j, i] = val
    end
    H
end

function build_hamiltonian_matrix(basis::BasisSet{<:GaussianBase}, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    n = length(basis.functions)
    H = zeros(Float64, n, n)
    for op in operators
        H .+= _build_operator_matrix(basis, op)
    end
    H
end

function solve_generalized_eigenproblem(H::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(S); check = true)
    L = F.L
    A = (L \ H) / L'
    evals, evecs = eigen(Symmetric(A))
    vecs = L' \ evecs
    real(evals), real(vecs)
end

function diagonalize(basis::BasisSet{<:GaussianBase}, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    H = build_hamiltonian_matrix(basis, operators)
    S = build_overlap_matrix(basis)
    solve_generalized_eigenproblem(H, S)
end

function solve_ECG(operators::Vector{<:FewBodyHamiltonians.Operator},
                   n::Int=50;
                   sampler=HaltonSample(),
                   method::Symbol=:quasirandom,
                   scale::Real=0.2,            
                   sscale::Real=0.1,           
                   verbose::Bool=true)

    b₁ = float(scale)
    basis_fns = Rank0Gaussian[]
    E_hist = Float64[]
    vecs_list = Any[]

    w_list = [op.w for op in operators if op isa CoulombOperator]
    n_pairs = length(w_list)
    d = length(w_list[1])

    for i in 1:n
        bij = generate_bij(method, i, n_pairs, b₁; qmc_sampler=sampler)
        A   = _generate_A_matrix(bij, w_list)
        s   = generate_shift(method, i, length(w_list[1]), sscale; qmc_sampler=sampler) 
        push!(basis_fns, Rank0Gaussian(A, s))

        basis = BasisSet{Rank0Gaussian}(basis_fns)
        H = build_hamiltonian_matrix(basis, operators)
        S = build_overlap_matrix(basis)

        λs, Us = solve_generalized_eigenproblem(H, S)
        E0 = minimum(λs)

        push!(E_hist, E0)
        push!(vecs_list, Us)
        verbose && @info "Step $i" E₀=E0
    end

    Emin = last(E_hist)
    @info "Minimum found" E₀=Emin
    return SolverResults(basis_fns, n, operators, method, sampler, b₁, Emin, E_hist, vecs_list)
end
