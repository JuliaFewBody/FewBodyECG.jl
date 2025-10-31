using FewBodyECG
using LinearAlgebra
using Plots
using QuasiMonteCarlo

masses = [1.0e15, 1.0, 1.0]

Λmat = Λ(masses)
kin = KineticOperator(Λmat)
J, U = _jacobi_transform(masses)

w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]

w_raw = [U' * w for w in w_list]
coeffs = [-1.0, -1.0, +1.0]

ops = Operator[
    kin;
    (CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw))...
]

result = solve_ECG(ops, masses, 100)

E = -0.527751016523
ΔE = abs(result.ground_state - E)
@info "Energy difference" ΔE

n, E = convergence(result)
plot(n, E)