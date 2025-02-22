module EarthSciMLBase
using ModelingToolkit, Symbolics, Catalyst
using ModelingToolkit: AbstractSystem
using Graphs, MetaGraphsNext
using DocStringExtensions
using DynamicQuantities, Dates
using DomainSets
using SciMLBase: DECallback, CallbackSet, ODEProblem, SplitODEProblem, reinit!, solve!,
    init, remake
using SciMLOperators: cache_operator, NullOperator, FunctionOperator, TensorProductOperator
using Statistics
using DiffEqCallbacks
using LinearAlgebra
using SymbolicIndexingInterface: setp
using Accessors
using Graphs
using MacroTools, RuntimeGeneratedFunctions

include("add_dims.jl")
include("domaininfo.jl")
include("operator.jl")
include("coupled_system.jl")
include("coupled_system_utils.jl")
include("operator_compose.jl")
include("advection.jl")
include("coord_trans.jl")
include("param_to_var.jl")
include("graph.jl")
include("mtk_grid_func.jl")
include("solver_strategies.jl")
include("solver_strategy_strang.jl")
include("blockdiagonal.jl")

end
