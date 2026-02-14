# Model Composition

A main goal of the `EarthSciMLBase` package is to allow model components to be created independently and composed together.
We achieve this by creating `coupletype`s that allow us to use multiple dispatch on the [`EarthSciMLBase.couple2`](@ref) function to specify how particular systems should be coupled together.

To demonstrate how this works, below we define three model components, `Photolysis`, `Chemistry`, and `Emissions`, which represent different processes in the atmosphere.

```@example composition
using ModelingToolkit, Catalyst, EarthSciMLBase
using ModelingToolkit: t_nounits
t = t_nounits

struct PhotolysisCoupler
    sys
end
function Photolysis(; name = :Photolysis)
    @variables j_NO2(t)
    eqs = [
        j_NO2 ~ max(sin(t / 86400), 0)
    ]
    System(eqs, t, [j_NO2], [], name = name,
        metadata = Dict(CoupleType => PhotolysisCoupler))
end

Photolysis()
```

You can see that the system defined above is mostly a standard ModelingToolkit ODE system,
except for two things:

The first unique part is that we define a `PhotolysisCoupler` type:

```julia
struct PhotolysisCoupler
    sys
end
```

It is important that this type is a struct, and that the struct has a single field named `sys`.
When defining your own coupled systems, you can just copy the line of code above but change the
name of the type (i.e., change it to something besides `PhotolysisCoupler`).

The second unique part is that we define some metadata for our System to tell it what coupling
type to use:

```julia
metadata = Dict(CoupleType => PhotolysisCoupler)
```

Again, when making your own components, just copy the code above but change `PhotolysisCoupler` to something else.

Let's follow the same process for some additional components:

```@example composition
struct ChemistryCoupler
    sys
end
function Chemistry(; name = :Chemistry)
    @parameters jNO2
    @species NO2(t)
    rxs = [
        Reaction(jNO2, [NO2], [], [1], [])
    ]
    rsys = complete(ReactionSystem(rxs, t, [NO2], [jNO2];
        combinatoric_ratelaws = false, name = name))
    ode_model(rsys; metadata = Dict(CoupleType => ChemistryCoupler))
end

Chemistry()
```

For our chemistry component above, because it's is originally a ReactionSystem instead of an
ModelingToolkit System, we convert it to a ModelingToolkit System before adding the metadata.

```@example composition
struct EmissionsCoupler
    sys
end
function Emissions(; name = :Emissions)
    @parameters emis = 1
    @variables NO2(t)
    eqs = [NO2 ~ emis]
    System(eqs, t; name = name, metadata = Dict(CoupleType => EmissionsCoupler))
end

Emissions()
```

Now, we need to define ways to couple the model components together.
We can do this by defining a coupling function (as a method of [`EarthSciMLBase.couple2`](@ref)) for each pair of model components that we want to couple.
Each coupling function should have the signature `EarthSciMLBase.couple2(a::ACoupler, b::BCoupler)::ConnectorSystem`, and should assume that the two ModelingToolkit `System`s are inside their respective couplers in the `sys` field.
It should make any edits to the components as needed and return a [`ConnectorSystem`](@ref) which defines the relationship between the two components.

The code below defines a method for coupling the `Chemistry` and `Photolysis` components.
First, it uses the [`param_to_var`](@ref param_to_var) function to convert the photolysis rate parameter `jNO2` from the `Chemistry` component to a variable, then it creates a new `Chemistry` component with the updated photolysis rate, and finally, it creates a [`ConnectorSystem`](@ref ConnectorSystem) object that sets the `j_NO2` variable from the `Photolysis` component equal to the `jNO2` variable from the `Chemistry` component.
The next effect is that the photolysis rate in the `Chemistry` component is now controlled by the `Photolysis` component.

```@example composition
function EarthSciMLBase.couple2(c::ChemistryCoupler, p::PhotolysisCoupler)
    c, p = c.sys, p.sys
    c = param_to_var(convert(System, c), :jNO2)
    ConnectorSystem([c.jNO2 ~ p.j_NO2], c, p)
end
nothing # hide
```

Next, we define a method for coupling the `Chemistry` and `Emissions` components.
To do this we use the [`operator_compose`](@ref operator_compose) function to add the `NO2` species from the `Emissions` component to the time derivative of `NO2` in the `Chemistry` component.

```@example composition
function EarthSciMLBase.couple2(c::ChemistryCoupler, emis::EmissionsCoupler)
    c, emis = c.sys, emis.sys
    operator_compose(convert(System, c), emis, Dict(
        c.NO2 => emis.NO2,
    ))
end
nothing # hide
```

Finally, we can compose the model components together to create our complete model. To do so, we just initialize each model component and add the components together using the [`couple`](@ref) function. We can use the `convert` function to convert the composed model to a [ModelingToolkit](https://mtk.sciml.ai/dev/) `System` so we can see what the combined equations look like.

```@example composition
model = couple(Photolysis(), Chemistry(), Emissions())
convert(System, model)
```

Finally, we can use the [`graph`](@ref) function to visualize the model components and their connections.

```@example composition
using MetaGraphsNext
using CairoMakie, GraphMakie

g = graph(model)

f, ax, p = graphplot(g; ilabels = labels(g))
hidedecorations!(ax);
hidespines!(ax);
ax.aspect = DataAspect();

f
```
