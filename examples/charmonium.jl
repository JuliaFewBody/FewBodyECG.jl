# Charmonium ground state from a Cornell-type potential with a Gaussian-smeared
# spin-spin term, in GeV units (ħ = c = 1).
using FewBodyECG
using OptimKit: LBFGS
using Plots

m_c = 1.836                    # charm-quark mass (GeV)
σσ = -3                        # ⟨σ₁·σ₂⟩ in the spin singlet
Λ, λ, κ, κκ, A, B = 0.8321, 0.1653, 0.5069, 1.8609, 1.6553, 0.2204
μ = m_c / 2                    # reduced mass
r₀ = A * (2μ)^(-B)
spin_spin = 2π * κκ / (3 * m_c^2) / (sqrt(π) * r₀)^3 * σσ

ops = Kinetic([m_c, m_c]; units = UnitSystem("GeV", "GeV⁻¹")) +
    Coulomb(1, 2; coefficient = -κ) +
    Custom(r -> λ * r, 1, 2) +
    Gaussian(1, 2; coefficient = spin_spin, exponent = 1 / r₀^2)

sol = solve(ops, GVM(basis = 5, optimizer = LBFGS(; maxiter = 100)))
@info "Charmonium ground state (GeV)" mass = sol.E₀ + 2m_c - Λ

plot(convergence(sol))
plot(wavefunction(sol); coord = 1, rmax = 5.0)
