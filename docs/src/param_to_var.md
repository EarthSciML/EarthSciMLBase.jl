## Converting parameters to variables

This can be useful to replace a parameter that does not change in time in a model component 
with one specified by another system that does change in time (or space). For example, the 
code below specifies a first-order loss equation, and then changes the temperature (which 
determines the loss rate) with a temperature value that varies in time.

As an example, we will create a loss equation that depends on the temperature, starting with a constant temperature. We will then create a temperature equation that varies in time, and use the [`param_to_var`](@ref) function to replace the constant temperature in the loss equation with the time-varying temperature.

So first, let's specify the original system with constant temperature.

```@example param_to_var
using ModelingToolkit, EarthSciMLBase, Unitful

@variables t [unit=u"s", description="time"]

struct LossCoupler sys end
function Loss(t)
    @variables A(t)=1 [unit=u"kg"]
    @parameters k=1 [unit=u"s^-1"]
    @parameters T=300 [unit=u"K"]
    @constants T₀=300 [unit=u"K"]
    eq = Differential(t)(A) ~ -k*exp(T/T₀) * A
    ODESystem([eq]; name=:Loss, metadata=Dict(:coupletype=>LossCoupler))
end

Loss(t)
```

Next, we specify the temperature that varies in time.

```@example param_to_var
struct TemperatureCoupler sys end
function Temperature(t)
    @variables T(t)=300 [unit=u"K"]
    @constants Tc=1.0 [unit=u"K/s"]
    @constants tc=1.0 [unit=u"s"]
    eq = Differential(t)(T) ~ sin(t/tc)*Tc
    ODESystem([eq]; name=:Temperature, metadata=Dict(:coupletype=>TemperatureCoupler))
end

Temperature(t)
```

Now, we specify how to compose the two systems using `param_to_var`.

```@example param_to_var
function EarthSciMLBase.couple2(loss::LossCoupler, temp::TemperatureCoupler)
    loss, temp = loss.sys, temp.sys
    loss = param_to_var(loss, :T)
    ConnectorSystem([loss.T ~ temp.T], loss, temp)
end
```

Finally, we create the system components and the composed system.
```@example param_to_var
l = Loss(t)
t = Temperature(t)
variable_loss = couple(l, t)

get_mtk(variable_loss)
```

If we wanted to, we could then run a simulation with the composed system.