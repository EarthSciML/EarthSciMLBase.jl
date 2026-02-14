export param_to_var

"""
Add the units and description in the variable `from` to the variable `to`.
"""
function add_metadata(to, from; exclude_default = false)
    unit = ModelingToolkit.get_unit(from)
    to = Symbolics.setmetadata(to, ModelingToolkit.VariableUnit, unit)
    desc = ModelingToolkit.getdescription(from)
    to = Symbolics.setmetadata(to, ModelingToolkit.VariableDescription, desc)
    if ModelingToolkit.hasdefault(from) && !exclude_default
        default = ModelingToolkit.getdefault(from)
        to = ModelingToolkit.setdefault(to, default)
    end
    to
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
```
"""
function param_to_var(sys::ModelingToolkit.AbstractSystem, ps::Symbol...)
    params = parameters(sys)
    defaults = copy(getfield(sys, :initial_conditions))
    replace = Dict()
    for p in ps
        if p in ModelingToolkit.tosymbol.(unknowns(sys), escape = false) # Skip if it is already a variable.
            continue
        end
        iparam = findfirst(isequal(p), Symbol.(params))
        @assert !isnothing(iparam) "Parameter `$p` not found in the system parameters $(Symbol.(params))"
        param = params[iparam]

        iv = ModelingToolkit.get_iv(sys)
        newvar = only(@variables $p(iv))
        newvar = add_metadata(newvar, param; exclude_default = true)
        replace[param] = newvar
        delete!(defaults, param)
    end

    if isempty(replace)
        return sys # Nothing to replace
    end
    newsys = SymbolicUtils.substitute(sys, replace)
    copy_with_change(newsys;
        metadata = ModelingToolkit.get_metadata(sys),
        discrete_events = ModelingToolkit.get_discrete_events(sys),
        continuous_events = ModelingToolkit.get_continuous_events(sys),
        defaults = defaults
    )
end
