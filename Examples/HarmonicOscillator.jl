using FewBodyECG
using FewBodyHamiltonians
using Plots

exponents = 10 .^ range(-1, 1, length = 15)
basis_functions = Rank0Gaussian[
    Rank0Gaussian([α;;], zeros(1, 3)) for α in exponents
]
basis = BasisSet(basis_functions)
operators = Operator[
    KineticOperator([1 / 2;;]),
    OscillatorPotential(1 / 2, [1]),
]

energies, eigenvectors = solve_generalized_eigenproblem(
    build_hamiltonian_matrix(basis, operators),
    build_overlap_matrix(basis),
)
exact_energies = [3 / 2, 7 / 2, 11 / 2]

for state in 1:3
    println("n_r = $(state - 1): E = $(energies[state]), exact = $(exact_energies[state])")
end

function radial_probability(r, coefficients)
    coordinates = reshape([0.0, 0.0, r], 1, 3)
    return r^2 * abs2(ψ₀(coordinates, coefficients, basis_functions))
end

radius = range(-6.0, 6.0, length = 600)
radial_states = plot(
    xlabel = "r",
    ylabel = "normalized r²|ψ(r)|²",
    title = "3D harmonic-oscillator radial states",
)
for state in 1:3
    density = radial_probability.(radius, Ref(eigenvectors[:, state]))
    density ./= sum((density[i] + density[i + 1]) * (radius[i + 1] - radius[i]) / 2 for i in 1:(length(radius) - 1))
    plot!(radial_states, radius, density, label = "n_r = $(state - 1)")
end
display(radial_states)
