module FewBodyECG

using LinearAlgebra: Diagonal, Hermitian, I, Symmetric, UpperTriangular,
    cholesky, cond, cross, det, diag, dot, eigen, isposdef, issymmetric,
    norm, pinv, tr
import FewBodyHamiltonians

"""
    Operator

Alias for `FewBodyHamiltonians.Operator`, exported so raw operator vectors can
be typed as `Operator[...]` alongside the `Operators` builder.
"""
const Operator = FewBodyHamiltonians.Operator

# system building
export Operators, Operator,
    KineticOperator, CoulombOperator, GaussianOperator,
    OscillatorOperator, ManyBodyGaussianOperator, NumericalPotential, numerical,
    GaussianTensorOperator, GaussianSpinOrbitOperator,
    SpinProjection, SpinState, SpinGaussian,
    GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet
# solving
export solve, SolverMethod, StochasticMethod, GradientMethod,
    SVM, Refine, GVM, DynamicGVM, Pipeline, →
# results
export Solution, ConvergenceReport, StageResult, converged, energies
export wavefunction, Wavefunction, convergence, radial_profile
# matrix-level layer and generic names: documented and supported, reached via
# `FewBodyECG.name` or `using FewBodyECG: name`
public build_hamiltonian_matrix, build_overlap_matrix,
    solve_generalized_eigenproblem, Λ, jacobi_transform, default_scale,
    coulomb_weights, up, down

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("numerical_matrix_elements.jl")
include("operators.jl")
include("linalg.jl")
include("eigen.jl")
include("sampling.jl")
include("methods.jl")
include("solution.jl")
include("state.jl")
include("solve.jl")
include("gradient.jl")
include("utils/wavefunction.jl")
include("utils/convergence.jl")
include("utils/observables.jl")
include("utils/plotting.jl")


end
