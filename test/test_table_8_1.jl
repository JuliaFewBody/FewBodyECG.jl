using Test
using FewBodyECG
using OptimKit: LBFGS

# Table 8.1 (p. 153) uses 200–600 basis functions. A 16-function basis
# and 30 gradient iterations keep all S and P cases practical in the test suite.
# Energies are in Hartree and masses in electron masses.
@testset "Suzuki–Varga Table 8.1" begin
    mμ, md, mt = 206.7686, 3670.481, 5496.918
    ps = (masses = ones(3), charges = [1.0, -1.0, -1.0], exchange = [1, 3, 2])
    atom(Z) = (masses = [1.0e15, 1.0, 1.0], charges = [Z, -1.0, -1.0], exchange = [1, 3, 2])
    ttμ = (masses = [mt, mt, mμ], charges = [1.0, 1.0, -1.0], exchange = [2, 1, 3])
    tdμ(m_t, m_d, m_μ) = (masses = [m_t, m_d, m_μ], charges = [1.0, 1.0, -1.0], exchange = nothing)

    # (system, particles, state, rank, exchange sign, book energy)
    rows = [
        ("Ps⁻", ps, "¹Sᵉ", 0, 1, -0.262005070226),
        ("∞H⁻", atom(1.0), "¹Sᵉ", 0, 1, -0.527710163), # as printed in the book
        ("ttμ", ttμ, "¹Sᵉ", 0, 1, -112.973),
        ("ttμ", ttμ, "¹Pᵒ", 1, -1, -110.2621),
        ("tdμ", tdμ(mt, md, mμ), "¹Sᵉ", 0, 1, -111.36444),
        ("tdμ", tdμ(5496.92158, 3670.483014, 206.768262), "¹Pᵒ", 1, 1, -108.17923), # the book's mass set for this row
        ("∞He", atom(2.0), "¹Sᵉ", 0, 1, -2.9037243769),
        ("∞He", atom(2.0), "³Sᵉ", 0, -1, -2.1752291),
        ("∞He", atom(2.0), "¹Pᵒ", 1, 1, -2.1238423),
        ("∞He", atom(2.0), "³Pᵒ", 1, -1, -2.1331635),
        ("∞Li⁺", atom(3.0), "¹Sᵉ", 0, 1, -7.279913),
    ]

    optimizer = LBFGS(; maxiter = 30, gradtol = 1.0e-8, verbosity = 0, ls_verbosity = 0)
    for (system, particles, state, rank, sign, book) in rows
        @testset "$system $state" begin
            (; masses, charges, exchange) = particles
            symmetry = exchange === nothing ? nothing : PermutationSymmetry(masses, [exchange]; signs = [sign])
            method = SVM(16; rank, candidates = 40) → Refine(2; candidates = 30) → GVM(; optimizer)
            sol = solve(Kinetic(masses) + Coulomb(charges), method; symmetry)
            # The small basis has larger errors for tdμ without exchange projection.
            rtol = exchange === nothing ? 0.015 : 0.0025
            @test energy(sol) ≈ book rtol = rtol
        end
    end
end
