"""
Operators are objects that modify the current state of a `Simulator` system.
Each operator should be define a `run` function with the signature:

    `run!(op::Operator, s::Simulator, time)`

which modifies the `s.du` field in place. It should also implement:

    `timestep(op::Operator)`

which returns the timestep length for the operator.
"""
abstract type Operator end
