"""
    convergence(sol::Solution) -> (steps, history)

Return the cumulative solver-step indices `1:length(energy_history(sol))` together
with the per-step target-state energy `history = energy_history(sol)`, ready for
plotting a convergence curve.  See also [`energy_history`](@ref) and `plot(sol)`.
"""
function convergence(sol::Solution)
    history = energy_history(sol)
    return 1:length(history), history
end
