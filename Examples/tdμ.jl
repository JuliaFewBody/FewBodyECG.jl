using FewBodyECG, LinearAlgebra
using QuasiMonteCarlo
import FewBodyECG: default_scale, convergence
using Plots

masses = [5496.918, 3670.481, 206.7686]

Λmat = Λ(masses)
kin = KineticOperator(Λmat)

J, U = _jacobi_transform(masses)
base_w = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
coeffs = [+1.0, -1.0, -1.0]
w_list = [c .* w for (c, w) in zip(coeffs, base_w)]
w_raw = [U' * w for w in w_list]

coulomb_ops = [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]

ops = Operator[kin; coulomb_ops...]
scale = default_scale(masses)
result = solve_ECG(ops, 550, sampler = HaltonSample(); scale = scale)
println("E ≈ ", result.ground_state)

a, b = correlation_function(result)
conv = FewBodyECG.convergence(result)
