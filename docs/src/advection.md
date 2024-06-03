# Advection

The `Advection` function adds advection to a system of equations. This is useful for modeling the transport of a substance by a fluid.
Advection is implemented with the [`Advection`](@ref) type.

!warning
    Fully symbolic partial differential equations like those shown here don't currently work on domains that have a large number of grid cells. See [here](https://docs.sciml.ai/MethodOfLines/stable/performance/) for additional information.

To demonstrate how this can work, we will start with a simple system of equations:

```@example advection
using EarthSciMLBase, ModelingToolkit, Unitful

@parameters t

function ExampleSys(t)
    @variables y(t)
    @parameters p=2.0
    D = Differential(t)
    ODESystem([D(y) ~ p], t; name=:Docs₊ExampleSys)
end

ExampleSys(t)
```

We also need to create our initial and boundary conditions.
```@example advection
using DomainSets
@parameters x
domain = DomainInfo(constIC(0.0, t ∈ Interval(0, 1.0)), constBC(1.0, x ∈ Interval(0, 1.0)))
nothing # hide
```

Now we convert add advection to each of the state variables.
We're also adding a constant wind ([`ConstantWind`](@ref)) in the x-direction, with a speed of 1.0.

```@example advection
sys_advection = couple(ExampleSys(t), domain, ConstantWind(t, 1.0), Advection())
sys_mtk = get_mtk(sys_advection)
```

Finally, we can discretize the system and solve it:

```@example advection
using MethodOfLines, DifferentialEquations, Plots
discretization = MOLFiniteDifference([x=>10], t, approx_order=2)
@time prob = discretize(sys_mtk, discretization)
@time sol = solve(prob, Tsit5(), saveat=0.1)


# Plot the solution.
discrete_x = sol[x]
discrete_t = sol[t]
soly = sol[sys_mtk.dvs[3]]
anim = @animate for k in 1:length(discrete_t)
    plot(discrete_x, soly[k, 1:end], title="t=\$(discrete_t[k])", ylim=(0,2.5), lab=:none)
end
gif(anim, fps = 8)
```