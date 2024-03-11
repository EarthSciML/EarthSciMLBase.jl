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

This can be useful to replace a parameter that does not change in time in a model component 
with one specified by another system that does change in time (or space). For example, the 
code below specifies a first-order loss equation, and then changes the temperature (which 
determines the loss rate) with a temperature value that varies in time.

```jldoctest
using ModelingToolkit, EarthSciMLBase, DynamicQuantities

# Specify the original system with constant temperature.
struct Loss <: EarthSciMLODESystem
    sys::ODESystem
    Loss(sys::ModelingToolkit.ODESystem) = new(sys)
    function Loss(t)
        @variables A(t)=1 [unit=u"kg"]
        @parameters k=1 [unit=u"s^-1"]
        @parameters T=300 [unit=u"K"]
        @constants T₀=300 [unit=u"K"]
        eq = Differential(t)(A) ~ -k*exp(T/T₀) * A
        new(ODESystem([eq]; name=:loss))
    end
end

# Specify the temperature that varies in time.
struct Temperature <: EarthSciMLODESystem
    sys::ODESystem
    function Temperature(t)
        @variables T(t)=300 [unit=u"K"]
        @constants Tc=1.0 [unit=u"K/s"]
        @constants tc=1.0 [unit=u"s"]
        eq = Differential(t)(T) ~ sin(t/tc)*Tc
        new(ODESystem([eq]; name=:temperature))
    end
end

# Specify how to compose the two systems using `param_to_var`.
function Base.:(+)(loss::Loss, temp::Temperature)
    loss = Loss(param_to_var(loss.sys, :T))
    losseqn = loss.sys
    teqn = temp.sys
    ComposedEarthSciMLSystem(
        ConnectorSystem([losseqn.T ~ teqn.T], loss, temp), 
        loss, temp,
    )
end

# Create the system components and the composed system.
@variables t [unit=u"s", description="time"]
l = Loss(t)
t = Temperature(t)
variable_loss = l+t

equations(get_mtk(variable_loss))

# output
3-element Vector{Equation}:
 loss₊T(t) ~ temperature₊T(t)
 Differential(t)(loss₊A(t)) ~ -loss₊k*loss₊A(t)*exp(loss₊T(t) / loss₊T₀)
 Differential(t)(temperature₊T(t)) ~ temperature₊Tc*sin(t / temperature₊tc)
```
"""
function param_to_var(sys::ModelingToolkit.AbstractSystem, ps::Symbol...)
    params = parameters(sys)
    replace = Dict()
    for p ∈ ps
        iparam = findfirst(isequal(p), Symbol.(params))
        param = params[iparam]

        iv = ModelingToolkit.get_iv(sys)
        newvar = (@variables $p(iv))[1]
        newvar = add_metadata(newvar, param)
        replace[param] = newvar
    end

    SymbolicUtils.substitute(sys, replace)
end

