export Operator

"""
Operators are objects that modify the current state of a `Simulator` system.
Each operator should be define a function with the signature:

    `EarthSciMLBase.get_odefunction(::Operator, csys::CoupledSystem, mtk_sys, coord_args,
        domain::DomainInfo, u0, p, alg::MapAlgorithm)::AbstractSciMLOperator`

which should return a function that can be used as an ODE function, i.e. it should have
methods `f(u, p, t)` and optionally `f(du, u, p, t)` where `u` is a state vector, `p` is parameters,
`t` is time, and `du` is a cache for the result of the function.
For more information, see [here](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/).

Operators should also define a function with the signature:

    `EarthSciMLBase.get_needed_vars(op::Operator, csys, mtk_sys, domain::DomainInfo)`

which should return a list of variables that are needed by the operator.
"""
abstract type Operator end

function get_odefunction(
        op::Operator, csys, mtk_sys, coord_args, domain::DomainInfo, u0, p, alg)
    ArgumentError("Operator $(typeof(op)) does not define a EarthSciMLBase.get_odefunction method with the correct signature.")
end

function get_needed_vars(op::Operator, csys, mtk_sys, domain::DomainInfo)
    ArgumentError("Operator $(typeof(op)) does not define a EarthSciMLBase.get_needed_vars method with the correct signature.")
end
