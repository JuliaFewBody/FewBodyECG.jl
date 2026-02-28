# Examples 
    
Suppose you want to calculate the ground state energy of the hydrogen anion in the rest-frame of the proton. 

```@example example1
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

result = solve_ECG(ops, 250, scale = 1.0)

E = -0.527751016523
ΔE = abs(result.ground_state - E)
@info "Energy difference" ΔE

n, E = convergence(result)
plot(n, E)
```

## Variational optimisation with `solve_ECG_variational`

The stochastic solver above builds the basis greedily from random samples.
`solve_ECG_variational` instead treats all Gaussian parameters (the `A`
matrices encoded through their Cholesky factors, and the shift vectors `s`)
as continuous variables and minimises the ground-state energy directly with
L-BFGS via [OptimKit.jl](https://github.com/Jutho/OptimKit.jl).

### Fresh start

```@example example_var_fresh
using FewBodyECG
using LinearAlgebra
using Plots

masses = [1.0e15, 1.0, 1.0]   # H⁻: fixed nucleus + 2 electrons

Λmat = Λ(masses)
kin  = KineticOperator(Λmat)
J, U = _jacobi_transform(masses)

w_list = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
w_raw  = [U' * w for w in w_list]
coeffs = [-1.0, -1.0, +1.0]

ops = Operator[kin; [CoulombOperator(c, w) for (c, w) in zip(coeffs, w_raw)]...]

sr = solve_ECG_variational(ops, 20; scale = 1.0, max_iterations = 200, verbose = false)

E_exact = -0.527751016523
println("Variational E₀ = ", round(sr.ground_state, digits=8),
        "  ΔE = ", round(sr.ground_state - E_exact, sigdigits=3))

xs, ys = convergence_history(sr)
plot(xs, ys; xlabel="fg evaluations", ylabel="Energy (Ha)",
     label="Variational", lw=2)
hline!([E_exact]; label="Exact", linestyle=:dot, color=:black)
```

### Warm start from stochastic result

When the stochastic solver has already found a reasonable basis, refining it
with `loss_type = :trace` (minimise `Tr(S⁻¹H)`) often converges faster than a
fresh L-BFGS run.

```@example example_var_warm
using FewBodyECG
using LinearAlgebra

masses = [1.0e15, 1.0, 1.0]
Λmat = Λ(masses); kin = KineticOperator(Λmat)
J, U = _jacobi_transform(masses)
w_raw = [U' * w for w in [[1,-1,0],[1,0,-1],[0,1,-1]]]
ops = Operator[kin; [CoulombOperator(c,w) for (c,w) in zip([-1.,-1.,1.], w_raw)]...]

sr_stoch = solve_ECG(ops, 20; scale = 1.0, verbose = false)
basis0   = BasisSet(Rank0Gaussian[sr_stoch.basis_functions...])

sr_warm  = solve_ECG_variational(ops, 20;
    initial_basis = basis0,
    loss_type     = :trace,
    max_iterations = 200,
    verbose        = false,
)

E_exact = -0.527751016523
println("Stochastic E₀ = ", round(sr_stoch.ground_state, digits=8),
        "  ΔE = ", round(sr_stoch.ground_state - E_exact, sigdigits=3))
println("Warm-start E₀ = ", round(sr_warm.ground_state,  digits=8),
        "  ΔE = ", round(sr_warm.ground_state  - E_exact, sigdigits=3))
```