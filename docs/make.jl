using EarthSciMLBase
using Documenter

DocMeta.setdocmeta!(EarthSciMLBase, :DocTestSetup, :(using EarthSciMLBase); recursive=true)

makedocs(;
    modules=[EarthSciMLBase],
    authors="EarthSciML Authors and Contributors",
    repo="https://github.com/EarthSciML/EarthSciMLBase.jl/blob/{commit}{path}#{line}",
    sitename="EarthSciMLBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EarthSciML.github.io/EarthSciMLBase.jl",
        assets=String[],
        repolink="https://github.com/EarthSciML/EarthSciMLBase.jl"
    ),
    pages=[
        "Home" => "index.md",
        "Composition" => "composition.md",
        "Operator Composition" => "operator_compose.md",
        "Parameter Replacement" => "param_to_var.md",
        "Initial and Boundary Conditions" => "icbc.md",
        "Advection" => "advection.md",
        "Coordinate Transforms" => "coord_transforms.md",
        "Simulator" => "simulator.md",
        "Examples" => [
            "All Together" => "example_all_together.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/EarthSciML/EarthSciMLBase.jl",
    devbranch="main",
)
