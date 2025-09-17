using FewBodyECG
using LinearAlgebra
using Plots

particles = [Particle(1.0e15, 1.0, nothing), Particle(1.0, -1.0, nothing)]
sys = System(particles, true)

ops = Operator[
    Kinetic(1, 1.0, 1.0),
    Coulomb(1, 2, -1.0),
]

overlap(bra::Rank0Gaussian, ket::Rank0Gaussian) = begin
    A, B = bra.A, ket.A
    R = inv(A + B)
    (π^size(R, 1) / det(A + B))^(3 / 2)
end

n_basis = 25
basis_fns = Vector{GaussianBase}(undef, 0)
E0_list = Float64[]

for i in 1:n_basis
    dim = 1
    α = 0.2 + 0.1 * i
    A = α * I(dim)
    push!(basis_fns, Rank0Gaussian(A))

    basis = ECGBasis(basis_fns)

    N = length(basis.functions)
    H = zeros(Float64, N, N)
    S = zeros(Float64, N, N)

    for p in 1:N, q in 1:N
        bra = basis.functions[p]
        ket = basis.functions[q]
        S[p, q] = overlap(bra, ket)

        for op in ops
            if op isa Kinetic
                H[p, q] += compute_matrix_element(bra, ket, op, K)
            elseif op isa Coulomb
                H[p, q] += compute_matrix_element(bra, ket, op, w)
            end
        end
    end


    F = eigen(H, S)
    vals = real(F.values)
    E0 = minimum(vals)
    push!(E0_list, E0)
    @info "step $i" E0
end

E_exact = -0.125
E_min = minimum(E0_list)
ΔE = abs(E_min - E_exact)
@show ΔE

plot(
    1:n_basis, E0_list, xlabel = "Number of Gaussians", ylabel = "E₀ [Hartree]",
    lw = 2, label = "E₀ estimate", title = "S-wave Hydrogen (minimal demo)"
)
hline!([E_exact], label = "Exact: -0.125", linestyle = :dash)
