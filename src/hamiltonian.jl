using FewBodyHamiltonians
using QuasiMonteCarlo

# Eq. 8: Rank-0 overlap M₀
function _compute_overlap_element(bra::Rank0Gaussian, ket::Rank0Gaussian)
    R = inv(bra.A + ket.A)
    n = size(R, 1)
    return (π^n / det(bra.A + ket.A))^(3 / 2)
end

# Eq. 16: Rank-1 overlap M₁ = ½ bᵀRa M₀
function _compute_overlap_element(bra::Rank1Gaussian, ket::Rank1Gaussian)
    R = inv(bra.A + ket.A)
    n = size(R, 1)
    M0 = (π^n / det(bra.A + ket.A))^(3 / 2)
    return 0.5 * dot(bra.a, R * ket.a) * M0
end

# Eq. 33: Rank-2 overlap M₂
function _compute_overlap_element(bra::Rank2Gaussian, ket::Rank2Gaussian)
    R = inv(bra.A + ket.A)
    n = size(R, 1)
    M0 = (π^n / det(bra.A + ket.A))^(3 / 2)
    a, b, c, d = bra.a, bra.b, ket.a, ket.b
    return 0.25 * (
        dot(a, R * b) * dot(c, R * d) +
            dot(a, R * c) * dot(b, R * d) +
            dot(a, R * d) * dot(b, R * c)
    ) * M0
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

function solve_ECG(operators::Vector{FewBodyHamiltonians.Operator}, system::ParticleSystem, n::Int = 50; sampler = SobolSample(), method::Symbol = :quasirandom, verbose::Bool = true)
    b₁ = default_b0(system.scale)
    basis_fns = GaussianBase[]
    E₀ = Float64[]
    coulomb_length = count(x -> x isa CoulombOperator, operators)
    w_list = [op.w for op in operators if op isa CoulombOperator]
    E₀_list = Float64[]
    vecs_list = []
    for i in 1:n
        bij = generate_bij(method, i, coulomb_length, b₁; qmc_sampler = sampler)
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
