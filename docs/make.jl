using Documenter, FewBodyECG, DocumenterCitations, Literate

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

# Scripts in examples/ rendered as documentation pages; their code runs on every
# docs build. Each script's `# ` comment lines become the page's prose.
const LITERATE_EXAMPLES = [
    "Nuclear potentials" => "nuclear.jl",
    "Spin-orbit splitting" => "spin_orbit.jl",
    "Hooke's atom" => "harmonium.jl",
    "Muonic molecular ions" => "muonic_ions.jl",
]
const GENERATED = joinpath(@__DIR__, "src", "generated")
for (_, script) in LITERATE_EXAMPLES
    Literate.markdown(
        joinpath(@__DIR__, "..", "examples", script), GENERATED;
        documenter = true, credit = false
    )
end

makedocs(
    build = "build",
    modules = [FewBodyECG],
    checkdocs = :public,
    sitename = "FewBodyECG.jl",
    pages = [
        "Introduction" => "index.md",
        "Theoretical background" => "theory.md",
        "Building systems" => "systems.md",
        "Choosing a solver" => "solvers.md",
        "Examples" => [
            "Benchmarks" => "examples.md",
            (
                title => joinpath("generated", replace(script, ".jl" => ".md"))
                    for (title, script) in LITERATE_EXAMPLES
            )...,
        ],
        "API" => "API.md",
    ],
    plugins = [bib],
    format = Documenter.HTML()
)

deploydocs(
    repo = "github.com/JuliaFewBody/FewBodyECG.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main"
)
