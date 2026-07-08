module FewBodyECG

using LinearAlgebra
import Antique
using FewBodyHamiltonians

"""
    Operator

Alias for `FewBodyHamiltonians.Operator`, exported so raw operator vectors can
be typed as `Operator[...]` alongside the `Operators` builder.
"""
const Operator = FewBodyHamiltonians.Operator

# system building
export Operators, coulomb_weights, Operator,
    KineticOperator, CoulombOperator, GaussianOperator,
    GaussianBase, Rank0Gaussian, Rank1Gaussian, Rank2Gaussian, BasisSet
# solving
export solve, SVM, Refine, Variational, GrowVariational, Pipeline, →, AutoDiff
# results
export Solution, ConvergenceReport, StageResult, converged, energies
export wavefunction, Wavefunction
# power-user layer
export build_hamiltonian_matrix, build_overlap_matrix,
    solve_generalized_eigenproblem, Λ, jacobi_transform, default_scale

include("types.jl")
include("coordinates.jl")
include("matrix_elements.jl")
include("operators.jl")
include("linalg.jl")
include("eigen.jl")
include("sampling.jl")
include("methods.jl")
include("solution.jl")
include("state.jl")
include("solve.jl")
include("gradient.jl")
include("observables.jl")
include("recipes.jl")


end
