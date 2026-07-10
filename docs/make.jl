using Documenter, Literate, FewBodyECG

const EXDIR = joinpath(@__DIR__, "..", "examples")
const OUTDIR = joinpath(@__DIR__, "src", "examples")
mkpath(OUTDIR)
for f in readdir(EXDIR; join = true)
    endswith(f, ".jl") && Literate.markdown(f, OUTDIR; documenter = true)
end

makedocs(
    build = "build",
    modules = [FewBodyECG],
    checkdocs = :exports,
    sitename = "FewBodyECG.jl",
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Building systems" => "systems.md",
        "Choosing a solver" => "solvers.md",
        "Convergence" => "convergence.md",
        "Examples" => [
            "Hydrogen" => "examples/hydrogen.md",
            "Positronium" => "examples/positronium.md",
            "Helium and H-" => "examples/helium.md",
            "tdmu" => "examples/tdmu.md",
            "H2+ (non-BO)" => "examples/h2plus.md",
            "Gaussian wells" => "examples/gaussian_well.md",
            "Hooke's atom" => "examples/harmonium.md",
            "Three-body Gaussian" => "examples/three_body_gaussian.md",
            "Spin-orbit doublet" => "examples/spin_orbit.md",
            "Workflow" => "examples/workflow.md",
        ],
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
