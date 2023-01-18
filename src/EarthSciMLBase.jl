module EarthSciMLBase
using ModelingToolkit, Symbolics, Catalyst
using DocStringExtensions

include("composition.jl")
include("add_dims.jl")
include("icbc.jl")
include("composed_system.jl")
include("operator_compose.jl")

end
