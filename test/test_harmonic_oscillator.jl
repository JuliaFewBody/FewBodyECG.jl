using Test
using LinearAlgebra
using FewBodyECG
import Antique

# Isotropic 3D harmonic oscillator (k = m = ℏ = 1) modelled as a unit-mass
# particle bound to a fixed centre by an OscillatorOperator.  Its l = 0 states
# (nᵣ = 0,1,2) coincide with Antique's odd 1D states n = 1,3,5 at E = 3/2, 7/2,
# 11/2.  A tempered geometric basis of unshifted Gaussians resolves all three
# to spectroscopic accuracy; the sqrt(2) factor converts Antique's full-line
# odd-state normalization to the half-line reduced-radial normalization
# ∫₀^∞ u² dr = 1.
@testset "Harmonic oscillator vs Antique" begin
    ops = Operators([1.0e15, 1.0])
    ops += "Kinetic"
    ops += ("Oscillator", 1, 2, 0.5)

    w = only(op.w for op in ops.terms if op isa OscillatorOperator)
    αs = exp10.(range(-1.0, 1.0, length = 16))
    basis = BasisSet([Rank0Gaussian(α .* (w * transpose(w)), zeros(1, 3)) for α in αs])
    H = build_hamiltonian_matrix(basis, ops)
    S = build_overlap_matrix(basis)
    ecg_energies, coeffs = solve_generalized_eigenproblem(H, S)

    HO = Antique.HarmonicOscillator(k = 1.0, m = 1.0, ℏ = 1.0)
    @test ecg_energies[1:3] ≈ [Antique.E(HO; n = n) for n in (1, 3, 5)] atol = 1.0e-5

    radius = range(1.0e-4, 6, length = 800)
    ψ = Wavefunction(basis, coeffs[:, 1])
    ecg_wavefunction = [ψ([r]) for r in radius]
    u_ecg = sqrt(4π) .* radius .* ecg_wavefunction
    u_ref = sqrt(2) .* [Antique.ψ(HO, r; n = 1) for r in radius]
    u_ecg .*= sign(dot(u_ecg, u_ref))
    @test maximum(abs.(u_ecg - u_ref)) < 1.0e-3
end
