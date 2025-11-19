using FewBodyECG, LinearAlgebra

masses = [1.0, 1.0, 1.0]

Λmat = Λ(masses)
kin = KineticOperator(Λmat)

J, U = _jacobi_transform(masses)
w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
w_raw = [U' * w for w in w_list]

coeffs = [+1.0, -1.0, -1.0]
coulomb_ops = [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]

ops = Operator[kin; coulomb_ops...]

result = solve_ECG(ops, 250, sampler = SobolSample(); scale = 0.2)
println("E ≈ ", result.ground_state)
