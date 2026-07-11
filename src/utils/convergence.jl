"""
    convergence(sol::Solution) -> (steps, history)

Return the cumulative solver-step indices `1:length(energies(sol))` together
with the per-step target-state energy `history = energies(sol)`, ready for
plotting a convergence curve.  See also [`energies`](@ref) and `plot(sol)`.
"""
function convergence(sol::Solution)
    history = energies(sol)
    return 1:length(history), history
end
