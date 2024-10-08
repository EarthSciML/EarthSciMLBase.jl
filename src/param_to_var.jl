export param_to_var

"""
Add the units and description in the variable `from` to the variable `to`.
"""
function add_metadata(to, from)
    unit = ModelingToolkit.get_unit(from)
    to = Symbolics.setmetadata(to, ModelingToolkit.VariableUnit, unit)
    desc = ModelingToolkit.getdescription(from)
    to = Symbolics.setmetadata(to, ModelingToolkit.VariableDescription, desc)
end

"""
Replace the parameter `p` in the system `sys` with a new variable that has 
the same name, units, and description as `p`.

$(SIGNATURES)

This can be useful to replace a parameter that does not change in time in a model component 
with one specified by another system that does change in time (or space). For example, the 
code below specifies a first-order loss equation, and then changes the temperature (which 
determines the loss rate) with a temperature value that varies in time.
```
"""
function param_to_var(sys::ModelingToolkit.AbstractSystem, ps::Symbol...)
    params = parameters(sys)
    replace = Dict()
    for p ∈ ps
        iparam = findfirst(isequal(p), Symbol.(params))
        param = params[iparam]

        iv = ModelingToolkit.get_iv(sys)
        newvar = only(@variables $p(iv))
        newvar = add_metadata(newvar, param)
        replace[param] = newvar
    end

    newsys = SymbolicUtils.substitute(sys, replace)
    ODESystem(equations(newsys), ModelingToolkit.get_iv(newsys); 
        name=nameof(newsys), metadata=ModelingToolkit.get_metadata(sys))
end

