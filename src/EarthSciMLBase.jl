module EarthSciMLBase
using ModelingToolkit, Symbolics
using ModelingToolkit: AbstractSystem
using Graphs, MetaGraphsNext
using DocStringExtensions
using DynamicQuantities, Dates
using DomainSets
using SciMLBase: DECallback, CallbackSet, SplitODEProblem, reinit!, solve!,
                 init, remake
import SciMLBase: ODEProblem
using Statistics
using DiffEqCallbacks
using LinearAlgebra
using SymbolicIndexingInterface: setp
using Accessors
using Graphs
using MacroTools, RuntimeGeneratedFunctions
import SciMLOperators as SMO
import ThreadsX
import AcceleratedKernels as AK

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
include("map_algorithm.jl")
include("blockdiagonal.jl")
include("jacobian.jl")
include("mtk_grid_func.jl")
include("solver_strategies.jl")
include("solver_strategy_strang.jl")

end
