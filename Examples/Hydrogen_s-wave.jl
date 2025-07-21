using FewBodyECG
using LinearAlgebra
using Plots

masses = [1e15, 1.0]  # proton, electron
psys = ParticleSystem(masses)

K = Diagonal([0.0, 0.5])
K_transformed = psys.J * K * psys.J'

w_raw = [psys.U' * [1, -1]]  # r₁ - r₂
coeffs = [-1.0]  # Coulomb attraction

n_basis = 20
method = :quasirandom 
b1 = 1.5

basis_fns = GaussianBase[]
E₀_list = Float64[]

# --- Build basis and compute ground state energy step by step
for i in 1:n_basis
    bij = generate_bij(method, i, length(w_raw), b1)
    A = generate_A_matrix(bij, w_raw)
    push!(basis_fns, Rank0Gaussian(A))

    basis = BasisSet(basis_fns)
    ops = Operator[
        KineticEnergy(K_transformed);
        (CoulombPotential(c, w) for (c, w) in zip(coeffs, w_raw))...
    ]

    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)
    
    λs, Us = eigen(S)
    keep = λs .> 1e-10
    S⁻¹₂ = Us[:, keep] * Diagonal(1 ./ sqrt.(λs[keep])) * Us[:, keep]'
    H̃ = Symmetric(S⁻¹₂ * H * S⁻¹₂)
    E₀ = minimum(eigen(H̃).values)
    
    push!(E₀_list, E₀)
    
    
end

E_exact = -0.5
E_min = minimum(E₀_list)
@show ΔE = abs(E_min - E_exact)

plot(1:n_basis, E₀_list, xlabel="Number of Gaussians", ylabel="E₀ [Hartree]",
     lw=2, label="E₀ estimate", title="s-wave Hydrogen Convergence")
hline!([E_exact], label="Exact: -0.5", linestyle=:dash)
