# Model Composition

A main goal of the `EarthSciMLBase` package is to allow model components to be created independently and composed together.
We achieve this by creating a registry of "coupling functions" that define how to connect different model components together.

To demonstrate how this works, below we define three model components, `Photolysis`, `Chemistry`, and `Emissions`, which represent different processes in the atmosphere.

```@example composition
using ModelingToolkit, Catalyst, EarthSciMLBase
@parameters t

function Photolysis(t)
    @variables j_NO2(t)
    eqs = [
        j_NO2 ~ max(sin(t/86400),0)
    ]
    ODESystem(eqs, t, [j_NO2], []; name=:Docs₊Photolysis)
end

Photolysis(t)
```

```@example composition
function Chemistry(t)
    @parameters jNO2
    @species NO2(t)
    rxs = [
        Reaction(jNO2, [NO2], [], [1], [1])
    ]
    ReactionSystem(rxs, t, [NO2], [jNO2]; 
        combinatoric_ratelaws=false, name=:Docs₊Chemistry)
end

Chemistry(t)
```

```@example composition
function Emissions(t)
    @parameters emis = 1
    @variables NO2(t)
    eqs = [NO2 ~ emis]
    ODESystem(eqs, t, [NO2], [emis]; name=:Docs₊Emissions)
end

Emissions(t)
```

Now, we need to define ways to couple the model components together.
We can do this by defining a coupling function for each pair of model components that we want to couple, and registering the coupling functions with the [`register_coupling`](@ref) function.
Each coupling function should have the signature `(f::Function, sys1::AbstractSystem, sys2::AbstractSystem)`, where `sys1` and `sys2` are instances of the model components to be coupled, and `f` is the coupling function that takes the two model components as arguments, makes any edits to the components as needed, and returns a [`ConnectorSystem`](@ref) which defines the relationship between the two components.

The code below defines a method for coupling the `Chemistry` and `Photolysis` components. 
First, it uses the [`param_to_var`](@ref param_to_var) function to convert the photolysis rate parameter `jNO2` from the `Chemistry` component to a variable, then it creates a new `Chemistry` component with the updated photolysis rate, and finally, it creates a [`ConnectorSystem`](@ref ConnectorSystem) object that sets the `j_NO2` variable from the `Photolysis` component equal to the `jNO2` variable from the `Chemistry` component.
The next effect is that the photolysis rate in the `Chemistry` component is now controlled by the `Photolysis` component.

```@example composition
register_coupling(Chemistry(t), Photolysis(t)) do c, p 
    c = param_to_var(convert(ODESystem, c), :jNO2)
    ConnectorSystem([c.jNO2 ~ p.j_NO2], c, p)
end
nothing # hide
```
Next, we define a method for coupling the `Chemistry` and `Emissions` components.
To do this we use the [`operator_compose`](@ref operator_compose) function to add the `NO2` species from the `Emissions` component to the time derivative of `NO2` in the `Chemistry` component.

```@example composition
register_coupling(Chemistry(t), Emissions(t)) do c, emis
    operator_compose(convert(ODESystem, c), emis, Dict(
        c.NO2 => emis.NO2,
    ))
end
nothing # hide
```

Finally, we can compose the model components together to create our complete model. To do so, we just initialize each model component and add the components together using the [`couple`](@ref) function. We can use the [`get_mtk`](@ref) function to convert the composed model to a [ModelingToolkit](https://mtk.sciml.ai/dev/) model so we can see what the combined equations look like.

```@example composition
model = couple(Photolysis(t), Chemistry(t), Emissions(t))
get_mtk(model)
```

Finally, we can use the [`graph`](@ref) function to visualize the model components and their connections.

```@example composition
using MetaGraphsNext
using CairoMakie, GraphMakie

g = graph(model)

f, ax, p = graphplot(g; ilabels=labels(g))
hidedecorations!(ax); hidespines!(ax); ax.aspect = DataAspect()

f
```