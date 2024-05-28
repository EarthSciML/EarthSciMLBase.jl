# Model Composition

A main goal of the `EarthSciMLBase` package is to allow model components to be created independently and composed together. 
We achieve this by creating type, [EarthSciMLODESystem](@ref), which is a wrapper around the `ODESystem` type from the [ModelingToolkit](https://mtk.sciml.ai/stable/) package.

For example, below we define three model components, `Photolysis`, `Chemistry`, and `Emissions`, which represent different processes in the atmosphere.
Each of these components is defined as a subtype of `EarthSciMLODESystem` and contains an `ODESystem` object named `sys`.
Each of them also included a constructor function which takes a time variable `t` as an argument and returns an instance of the model component, including an initialized version of the ModelingToolkit model component.

```@example composition
using ModelingToolkit, Catalyst, EarthSciMLBase

struct Photolysis <: EarthSciMLODESystem
    sys::ODESystem
    function Photolysis(t)
        @parameters j_unit = 1
        @variables j_NO2(t) = 0.0149
        eqs = [
            j_NO2 ~ j_unit
        ]
        new(ODESystem(eqs, t, [j_NO2], [j_unit]; name=:photolysis))
    end
end

struct Chemistry <: EarthSciMLODESystem
    sys::ODESystem
    Chemistry(sys::ODESystem) = new(sys)
    function Chemistry(t)
        @parameters jNO2 = 0.0149
        @species NO2(t) = 10.0
        rxs = [
            Reaction(jNO2, [NO2], [], [1], [1])
        ]
        rxn_sys = ReactionSystem(rxs, t, [NO2], [jNO2]; 
            combinatoric_ratelaws=false, name=:chemistry)
        new(convert(ODESystem, rxn_sys))
    end
end

struct Emissions <: EarthSciMLODESystem
    sys::ODESystem
    function Emissions(t)
        @parameters emis = 1
        @variables NO2(t) = 0.00014
        eqs = [NO2 ~ emis]
        new(ODESystem(eqs, t, [NO2], [emis]; name=:emissions))
    end
end
nothing # hide
```

Now, we need to define ways to couple the model components together.
We can do this by defining a method of the `EarthSciMLBase.couple` function for each pair of model components that we want to couple.
The code below define a method for coupling the `Chemistry` and `Photolysis` components. 
First, it uses the [`param_to_var`](@ref param_to_var) function to convert the photolysis rate parameter `jNO2` from the `Chemistry` component to a variable, then it creates a new `Chemistry` component with the updated photolysis rate, and finally, it creates a [`ConnectorSystem`](@ref ConnectorSystem) object that sets the `j_NO2` variable from the `Photolysis` component equal to the `jNO2` variable from the `Chemistry` component.
The next effect is that the photolysis rate in the `Chemistry` component is now controlled by the `Photolysis` component.

```@example composition
function EarthSciMLBase.couple(c::Chemistry, p::Photolysis)
    sys = param_to_var(c.sys, :jNO2)
    c = Chemistry(sys)
    ConnectorSystem([c.sys.jNO2 ~ p.sys.j_NO2], c, p)
end
nothing # hide
```
Next, we define a method for coupling the `Chemistry` and `Emissions` components.
To do this we use the [`operator_compose`](@ref operator_compose) function to add the `NO2` species from the `Emissions` component to the time derivative of `NO2` in the `Chemistry` component.

```@example composition
EarthSciMLBase.couple(c::Chemistry, emis::Emissions) = operator_compose(c, emis, Dict(
    c.sys.NO2 => emis.sys.NO2,
))
nothing # hide
```

Finally, we can compose the model components together to create our complete model. To do so, we just initialize each model component and add the components together using the `+` operator.

```@example composition
@parameters t

model  = Photolysis(t) + Chemistry(t) + Emissions(t)
```

Finally, we can use the [`graph`](@ref graph) function to visualize the model components and their connections.

```@example composition
using MetaGraphsNext
using CairoMakie, GraphMakie

g = graph(model)

f, ax, p = graphplot(g; ilabels=labels(g))
hidedecorations!(ax); hidespines!(ax); ax.aspect = DataAspect()

f
```