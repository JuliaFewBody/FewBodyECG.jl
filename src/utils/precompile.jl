using PrecompileTools: @compile_workload

# Compile the solvers, the pilot runs behind `scale = :adaptive`, and result
# display at precompile time.  The matrix elements and the gradient (fixed
# chunk size) do not specialize on the number of particles, so this two-body
# workload also covers larger systems.
@compile_workload begin
    H = Kinetic([1.0e15, 1.0]) + Coulomb([+1.0, -1.0])
    lbfgs = LBFGS(; maxiter = 3, gradtol = 1.0e-6, verbosity = 0, ls_verbosity = 0)
    sol = solve(H, SVM(basis = 4, candidates = 2) → Refine(1; candidates = 2) → GVM(; optimizer = lbfgs))
    solve(H, DynamicGVM(; basis = 3, candidates = 2, optimizer = lbfgs))
    show(devnull, MIME"text/plain"(), sol)
    wavefunction(sol)([0.5])
end
