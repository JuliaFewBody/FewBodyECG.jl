module FewBodyECG

using LinearAlgebra: Diagonal, Hermitian, I, RowMaximum, Symmetric, UpperTriangular,
    cholesky, cond, cross, det, diag, dot, eigen, eigmin, eigvals, isposdef, issymmetric,
    norm, opnorm, pinv, tr
import FewBodyHamiltonians
import ForwardDiff

"""
    Operator

Alias for `FewBodyHamiltonians.Operator`, the supertype of every term and
Jacobi-frame operator, exported so raw operator vectors can be typed as
`Operator[...]`.
"""
const Operator = FewBodyHamiltonians.Operator

# system building
export Operator, UnitSystem,
    Hamiltonian, Kinetic, Coulomb, Gaussian, ManyBodyGaussian, Oscillator, Custom,
    Tensor, SpinOrbit,
    KineticOperator, CoulombOperator, GaussianOperator,
    OscillatorOperator, ManyBodyGaussianOperator, NumericalPotential,
    GaussianTensorOperator, GaussianSpinOrbitOperator,
    PermutationOperator, ContactDensityOperator,
    PermutationSymmetry, ProjectedGaussian,
    DipoleOperator, QuadrupoleOperator,
    MomentumFourthOperator, MassVelocityOperator, OrbitOrbitOperator,
    SpinExchangeOperator, IsospinExchangeOperator, IsospinProjection, IsospinState,
    SpinProjection, SpinState, SpinGaussian,
    GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet
# solving
export solve, SolverMethod, StochasticMethod, GradientMethod,
    SVM, Refine, GVM, DynamicGVM, Pipeline, →
# results
export Solution, ConvergenceReport, StageResult, converged, energy, energy_history
export wavefunction, Wavefunction, convergence, radial_profile, expectation
# matrix-level layer and generic names: documented and supported, reached via
# `FewBodyECG.name` or `using FewBodyECG: name`
public build_hamiltonian_matrix, build_overlap_matrix, build_operator_matrix,
    solve_generalized_eigenproblem, Λ, jacobi_transform, default_scale,
    coulomb_weights, up, down, proton, neutron

include("types.jl")
include("physics/units.jl")
include("hamiltonian.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("physics/polarized_elements.jl")
include("physics/permutation_contact.jl")
include("physics/spin_exchange.jl")
include("physics/multipoles.jl")
include("physics/relativistic.jl")
include("physics/symmetry.jl")
include("jacobi_frame.jl")
include("linalg.jl")
include("eigen.jl")
include("sampling.jl")
include("methods.jl")
include("utils/solution.jl")
include("state.jl")
include("solve.jl")
include("gradient.jl")
include("utils/wavefunction.jl")
include("utils/convergence.jl")
include("utils/observables.jl")
include("utils/plotting.jl")
include("utils/precompile.jl")


end
