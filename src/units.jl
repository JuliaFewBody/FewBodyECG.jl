"""
    UnitSystem(energy, length)

Names of the energy and length units in which a system's masses and coupling
constants are expressed. FewBodyECG never converts between unit systems: the
kinetic operator is ``-\\sum_i \\nabla_i^2 / 2m_i`` (``\\hbar = 1``), so the
masses and potential parameters passed to [`Operators`](@ref) fix the units,
and a `UnitSystem` only labels results (printed energies, plot axes).

Presets, selected with `Operators(...; units = name)`:

| name | energy | length | masses |
|---|---|---|---|
| `:atomic` (default) | `Ha` | `bohr` | multiples of the electron mass |
| `:nuclear` | `MeV` | `fm` | ``mc^2 / (\\hbar c)^2`` in MeV⁻¹ fm⁻², with ``\\hbar c = 197.327`` MeV fm |

For any other convention, pass a `UnitSystem` directly, e.g.
`UnitSystem("GeV", "GeV⁻¹")` for ``\\hbar = c = 1`` with masses in GeV.
"""
struct UnitSystem
    energy::String
    length::String
end

const _UNIT_PRESETS = (
    atomic = UnitSystem("Ha", "bohr"),
    nuclear = UnitSystem("MeV", "fm"),
)

_unitsystem(u::UnitSystem) = u
function _unitsystem(name::Symbol)
    haskey(_UNIT_PRESETS, name) || throw(
        ArgumentError(
            "unknown unit system :$name; use one of " *
                join((":$k" for k in keys(_UNIT_PRESETS)), ", ") *
                " or pass a UnitSystem(energy, length)"
        )
    )
    return _UNIT_PRESETS[name]
end

Base.show(io::IO, u::UnitSystem) = print(io, "UnitSystem(\"", u.energy, "\", \"", u.length, "\")")
