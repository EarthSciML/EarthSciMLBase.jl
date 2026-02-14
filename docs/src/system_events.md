# Events that depend on the fully coupled system

Sometimes, something needs to happen to a component process in system which depends information about the fully coupled system.
For example, the data in data loaders may need to be periodically updated as the simulation runs, but we would want to avoid updating
any data loaders that are included in the data component of the model but are not needed for updating any of the state variables in
the full model that is running.

Take, for example, the model below, which has two variables, but only one of them ($x$) is associated with a differential equation.
As you see, when we create the system, only $x$ shows up in the equations that the model needs to run to solve the system:

```@example system_events
using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t_nounits, D_nounits
t, D = t_nounits, D_nounits
using DifferentialEquations
using Plots

@parameters a=0 b=0
@variables begin
    x(t_nounits) = 0
    y(t_nounits)
end

@named sys1 = System([D(x) ~ a], t_nounits, [x], [a])
@named sys2 = System([y ~ b], t_nounits, [y], [b])

model1 = couple(sys1, sys2)
sys = convert(System, model1)
```

The equation for $y$ shows up in the "observed" equations:

```@example system_events
observed(sys)
```

If we pretend that the parameters $a$ and $b$ are actually data that needs to be updated over time, and that the data updating process is computationally expensive, then we find ourselves in a situation where we need to update the data for $a$ as the simulation runs, but we don't want to update the data for $b$ because it is not needed for solving the differential equation system.
However, the systems for $x$ and $y$ could be used in different ways, so we don't necessarily know ahead of time that we don't need to update $b$.

To handle this type of situation, `EarthSciML` includes the idea of "system events" which are associated with individual system components, but are configured after the fully coupled system is created.

To create a system event, we need to create a function that takes a ModelingToolkit system as the argument and returns a ModelingToolkit event,
and then associate that function with the `SysDiscreteEvent` key in the metadata dictionary of the system component in question.

The code below does several things. First, it creates a function to determine whether a given variable is needed to solve the system or not.
Then, it creates "system event functions" for our two component systems, directing each system to increment its parameter value during the simulation, but only if the corresponding variable is needed for the simulation.
Finally, it includes two "runcount" variables to keep track of how many times each system event is called.

```@example system_events
# Utility function to check if a variable is needed in the system,
# i.e., if one of the state variables depends on it.
function is_var_needed(var, sys)
    var = EarthSciMLBase.var2symbol(var)
    if var in EarthSciMLBase.var2symbol.(unknowns(sys))
        return true
    end
    exprs = [eq.rhs for eq in equations(sys)]
    needed_obs = ModelingToolkit.observed_equations_used_by(sys, exprs)
    needed_vars = getproperty.(observed(sys)[needed_obs], :lhs)
    return var in EarthSciMLBase.var2symbol.(needed_vars)
end

runcount1 = 0
function sysevent1(sys)
    function f1!(mod, obs, ctx, integ)
        if is_var_needed(sys.sys1₊x, sys) # Only run if x is needed in the system.
            global runcount1 += 1
            return (sys1₊a = 1,)
        end
        return (sys1₊a = mod.sys1₊a,)
    end
    return [3.0] => (f = f1!, modified = (sys1₊a = sys.sys1₊a,))
end
runcount2 = 0
function sysevent2(sys)
    function f2!(mod, obs, ctx, integ)
        if is_var_needed(sys.sys2₊y, sys) # Only run if y is needed in the system.
            global runcount2 += 1
            return (sys2₊b = 1,)
        end
        return (sys2₊b = mod.sys2₊b,)
    end
    return [5.0] => (f = f2!, modified = (sys2₊b = sys.sys2₊b,))
end
```

Now, we need to recreate our systems, but this time we will add the system event functions to the metadata of the systems.
Then, we can run the simulation:

```@example system_events
sys1 = System([D(x) ~ a], t_nounits, [x], [a]; name = :sys1,
    metadata = Dict(SysDiscreteEvent => sysevent1))
sys2 = System([y ~ b], t_nounits, [y], [b]; name = :sys2,
    metadata = Dict(SysDiscreteEvent => sysevent2))

model1 = couple(sys1, sys2)
sys = convert(System, model1)
sol = solve(ODEProblem(sys, [], (0, 10)))
```

If we plot the results, we can see that the system event did indeed update the parameter value for $a$, which changed the rate of change of $x$.

```@example system_events
plot(sol)
```

Finally, we can look at our "run counters" to confirm that the first data updater did run, but the second one never did, thus saving us some computational time.

```@example system_events
runcount1, runcount2
```
