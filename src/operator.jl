export Operator

"""
Operators are objects that modify the current state of a `Simulator` system.
Each operator should be define a function with the signature:

    `EarthSciMLBase.get_scimlop(::Operator, mtk_sys, domain::DomainInfo, obs_functions, coordinate_transform_functions, u0, p)::AbstractSciMLOperator`

which should return a [SciMLOperators.AbstractSciMLOperator](https://docs.sciml.ai/SciMLOperators/stable/interface/).
Refer to the [SciMLOperators.jl](https://docs.sciml.ai/SciMLOperators/stable/)
documentation for more information on how to define operators.
"""
abstract type Operator end

function get_scimlop(op::Operator, mtk_sys, domain::DomainInfo, obs_functions, coordinate_transform_functions, u0, p)
    ArgumentError("Operator $(typeof(op)) does not define a EarthSciMLBase.get_scimlop method.")
end
