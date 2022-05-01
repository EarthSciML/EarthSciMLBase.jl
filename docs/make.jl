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
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/EarthSciML/EarthSciMLBase.jl",
    devbranch="main",
)
