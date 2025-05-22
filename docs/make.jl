using Documenter, Pkg
using FewBodyECG

push!(LOAD_PATH,"../src/")
makedocs(
    source  = "src", 
    sitename = "FewBodyECG.jl",
    modules = [FewBodyECG], 
    pages = [
        "index.md",
        "theory.md",
        "examples.md",
        "resources.md",
        "API.md"
        ]

)
deploydocs(
    repo = "github.com/JuliaFewBody/FewBodyECG.jl",
    target = "build",
    branch="gh-pages",
)