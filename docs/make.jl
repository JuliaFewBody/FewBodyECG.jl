using Documenter, FewBodyECG

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
        "Convergence" => "convergence.md",
        "API" => "API.md",
    ],
    format = Documenter.HTML()
)

deploydocs(
    repo = "github.com/JuliaFewBody/FewBodyECG.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main"
)
