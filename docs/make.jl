using Documenter, FewBodyECG

makedocs(
    sitename = "FewBodyECG.jl",
    format = Documenter.HTML(
        repolink = "https://github.com/JuliaFewBody/FewBodyECG.jl"
    ),
    modules = [FewBodyECG],
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Examples" => "examples.md",
        "Resources" => "resources.md",
        "API" => "API.md"
    ],
    repo = "https://github.com/JuliaFewBody/FewBodyECG.jl",
)

deploydocs(
    repo = "github.com/JuliaFewBody/FewBodyECG.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main"  # or "master" if you're using that
)
