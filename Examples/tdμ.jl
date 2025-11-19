using FewBodyECG, LinearAlgebra

masses = [5496.918, 3670.481, 206.7686]

Λmat = Λ(masses)
kin = KineticOperator(Λmat)

J, U = _jacobi_transform(masses)
base_w = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
coeffs = [+1.0, +1.0, -1.0]
w_list = [c .* w for (c, w) in zip(coeffs, base_w)]
w_raw = [U' * w for w in w_list]

coulomb_ops = [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]

ops = Operator[kin; coulomb_ops...]

result = solve_ECG(ops, 250, sampler = HaltonSample(); scale = 0.025)
println("E ≈ ", result.ground_state)
