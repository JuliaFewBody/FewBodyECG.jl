using FewBodyECG
using LinearAlgebra
using Plots
using QuasiMonteCarlo

masses = [1.0e15, 1.0, 1.0]
psys = ParticleSystem(masses)

K = Diagonal([0.0, 1 / 2, 1 / 2])
K_transformed = psys.J * K * psys.J'

w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]

w_raw = [psys.U' * w for w in w_list]
coeffs = [-1.0, -1.0, +1.0]


ops = Operator[
    KineticOperator(K_transformed);
    (CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw))...
]

A = solve_ECG(ops, psys, 100)

E = -0.527751016523
ΔE = abs(A.ground_state - E)
@info "Energy difference" ΔE

convergence(A)
