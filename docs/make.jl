using Documenter, FewBodyECG, DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

makedocs(
    build = "build",
    modules = [FewBodyECG],
    checkdocs = :exports,
    sitename = "FewBodyECG.jl",
    pages = [
        "Introduction" => "index.md",
        "Theoretical background" => "theory.md",
        "Building systems" => "systems.md",
        "Choosing a solver" => "solvers.md",
        "Examples" => "examples.md",
        "API" => "API.md",
    ],
    plugins=[bib],
    format = Documenter.HTML()
)

deploydocs(
    repo = "github.com/JuliaFewBody/FewBodyECG.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main"
)
