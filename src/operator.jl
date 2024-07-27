export Operator

"""
Operators are objects that modify the current state of a `Simulator` system.
Each operator should be define a `run` function with the signature:

    `EarthSciMLBase.run!(op::Operator, s::Simulator, time, step_length)`

which should modify the `s.u` system state, and can also modify the `s.du` derivative cache if desired. 
It should also implement:

    `EarthSciMLBase.timestep(op::Operator)`

which returns the timestep length for the operator.

The Operator may also optionally implement the initialize! and finalize! methods,
which will be run before the simulation starts and after it ends, respectively, if they are defined.

    `EarthSciMLBase.initialize!(op::Operator, s::Simulator)`
    `EarthSciMLBase.finalize!(op::Operator, s::Simulator)`
"""
abstract type Operator end

timestep(op::Operator) = ArgumentError("Operator $(typeof(op)) does not define a timestep function.")
run!(op::Operator, _, _, _) = ArgumentError("Operator $(typeof(op)) does not define a run! function.")
initialize!(_::Operator, _) = nothing
finalize!(_::Operator, _) = nothing