module EarthSciMLBase
using ModelingToolkit, Symbolics, Catalyst
using Graphs, MetaGraphsNext
using DocStringExtensions
using DynamicQuantities
using OrdinaryDiffEq, DomainSets
using SciMLBase: DECallback, CallbackSet
using DiffEqCallbacks
using LinearAlgebra, BlockBandedMatrices
using Accessors
using ProgressLogging
using Graphs

include("add_dims.jl")
include("domaininfo.jl")
include("operator.jl")
include("coupled_system.jl")
include("operator_compose.jl")
include("advection.jl")
include("coord_trans.jl")
include("param_to_var.jl")
include("graph.jl")
include("simulator_utils.jl")
include("simulator.jl")
include("simulator_strategies.jl")
include("simulator_strategy_strang.jl")

end
