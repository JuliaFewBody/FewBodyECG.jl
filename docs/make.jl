using Documenter, FewBodyECG

makedocs(
    build = "build",
    sitename = "FewBodyECG.jl",
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Examples" => "examples.md",
        "Resources" => "resources.md",
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
