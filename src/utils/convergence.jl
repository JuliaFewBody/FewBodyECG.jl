"""
    convergence(sol::Solution) -> (steps, history)
    convergence(history; steps=1:length(history)) -> (steps, history)

Return the cumulative solver-step indices `1:length(energy_history(sol))` together
with the per-step target-state energy `history = energy_history(sol)`, ready for
plotting a convergence curve.  See also [`energy_history`](@ref) and `plot(sol)`.

For manually assembled bases, pass the recorded energies as `history` and
optionally their basis sizes as `steps`. Both forms support `plot(convergence(...))`.
"""
convergence(sol::Solution) = convergence(energy_history(sol))

function convergence(history::AbstractVector{<:Real}; steps::AbstractVector{<:Real} = 1:length(history))
    length(steps) == length(history) || throw(DimensionMismatch("steps and energy history must have the same length"))
    return steps, history
end
