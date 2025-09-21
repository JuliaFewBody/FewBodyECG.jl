using FewBodyHamiltonians
using QuasiMonteCarlo

function _compute_overlap_element(bra::GaussianBase, ket::GaussianBase)
    A, B = bra.A, ket.A
    R = inv(A + B)
    return (π^length(R) / det(A + B))^(3 / 2)
end

function build_overlap_matrix(basis::BasisSet)
    n = length(basis.functions)
    S = zeros(n, n)
    for i in 1:n, j in 1:i
        val = _compute_overlap_element(basis.functions[i], basis.functions[j])
        S[i, j] = S[j, i] = val
    end
    return S
end

function _build_operator_matrix(basis::BasisSet, op::FewBodyHamiltonians.Operator)
    n = length(basis.functions)
    H = zeros(n, n)
    for i in 1:n, j in 1:i
        val = _compute_matrix_element(basis.functions[i], basis.functions[j], op)
        H[i, j] = H[j, i] = val
    end
    return H
end

function build_hamiltonian_matrix(basis::BasisSet, operators::AbstractVector{<:FewBodyHamiltonians.Operator})
    H = zeros(length(basis.functions), length(basis.functions))
    for op in operators
        H .+= _build_operator_matrix(basis, op)
    end
    return H
end

function solve_generalized_eigenproblem(H::Matrix{Float64}, S::Matrix{Float64})
    try
        F = cholesky(S; check = true)
        L = F.L
        Linv = inv(L)
        A = Linv * H * Linv'
        vals, vecs = eigen(Symmetric(A))
        vecs_orig = Linv' * vecs
        return real(vals), real(vecs_orig)
    catch err
        @warn "Cholesky on S failed, falling back to generalized eigen solver" exception = (err, catch_backtrace())
        vals, vecs = eigen(H, S)
        return real(vals), real(vecs)
    end
end

struct SolverResults
    basis_functions::Vector{GaussianBase}
    n_basis::Int
    operators::Vector{FewBodyHamiltonians.Operator}
    method::Symbol
    sampler::QuasiMonteCarlo.DeterministicSamplingAlgorithm
    length_scale::Float64
    ground_state::Float64
    energies::Vector{Float64}
    eigenvectors::Vector{Matrix{Float64}}
end

function convergence(sr::SolverResults)
    return 1:sr.n_basis, sr.energies
end

function solve_ECG(operators::Vector{FewBodyHamiltonians.Operator}, system::ParticleSystem, n::Int = 50; sampler = SobolSample(), method::Symbol = :quasirandom, verbose::Bool = true)
    b₁ = default_b0(system.scale)
    basis_fns = GaussianBase[]
    E₀ = Float64[]
    coulomb_length = count(x -> x isa CoulombOperator, operators)
    w_list = [op.w for op in operators if op isa CoulombOperator]
    E₀_list = Float64[]
    vecs_list = []
    for i in 1:n
        bij = generate_bij(:quasirandom, i, coulomb_length, b₁; qmc_sampler = sampler)
        A = _generate_A_matrix(bij, w_list)
        push!(basis_fns, Rank0Gaussian(A))

        basis = BasisSet(basis_fns)

        H = build_hamiltonian_matrix(basis, operators)
        S = build_overlap_matrix(basis)

        λs, Us = solve_generalized_eigenproblem(H, S)
        E₀ = minimum(λs)

        push!(E₀_list, E₀)
        push!(vecs_list, Us)
        verbose && @info "Step $i" E₀ = E₀
    end
    @info "Minimum found" E₀
    return SolverResults(basis_fns, n, operators, :quasirandom, sampler, b₁, E₀, E₀_list, vecs_list)
end
