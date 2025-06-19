```@meta
CurrentModule = EarthSciMLBase
```

# Example using different components of EarthSciMLBase together

This example shows how to define and couple different components of EarthSciMLBase together to create a more complex model. First, we create several model components.

Our first example system is a simple reaction system:

```@example ex1
using EarthSciMLBase
using ModelingToolkit, Catalyst, DomainSets, MethodOfLines, DifferentialEquations
using ModelingToolkit: t_nounits, D_nounits
t = t_nounits
D = D_nounits
using Plots

# Create our independent variable `t` and our partially-independent variables `x` and `y`.
@parameters x y

struct ExampleSys1Coupler
    sys
end
function ExampleSys1()
    @species c₁(t)=5.0 c₂(t)=5.0
    rs = ReactionSystem(
        [Reaction(2.0, [c₁], [c₂])],
        t; name = :Sys1, combinatoric_ratelaws = false)
    convert(System, complete(rs), metadata = Dict(:coupletype => ExampleSys1Coupler))
end

ExampleSys1()
```

Our second example system is a simple ODE system, with the same two variables.

```@example ex1
struct ExampleSys2Coupler
    sys
end
function ExampleSys2()
    @variables c₁(t)=5.0 c₂(t)=5.0
    @parameters p₁=1.0 p₂=0.5 x=1 y=1
    System(
        [D(c₁) ~ -p₁, D(c₂) ~ p₂],
        t, [c₁, c₂], [p₁, p₂, x, y]; name = :Sys2,
        metadata = Dict(:coupletype => ExampleSys2Coupler))
end

ExampleSys2()
```

Now, we specify what should happen when we couple the two systems together.
In this case, we want the the derivative of the composed system to
be equal to the sum of the derivatives of the two systems.
We can do that using the [`operator_compose`](@ref) function
from this package.

```@example ex1
function EarthSciMLBase.couple2(sys1::ExampleSys1Coupler, sys2::ExampleSys2Coupler)
    sys1, sys2 = sys1.sys, sys2.sys
    sys1 = convert(System, sys1)
    operator_compose(sys1, sys2)
end
nothing # hide
```

Once we specify all of the above, it is simple to create our two individual systems and then couple them together.

```@example ex1
sys1 = ExampleSys1()
sys2 = ExampleSys2()
sys = couple(sys1, sys2)

convert(System, sys)
```

At this point we have an ODE system that is composed of two other ODE systems.
We can inspect its observed variables using the `observed` function.

```@example ex1
simplified_sys = convert(System, sys)
```

```@example ex1
observed(simplified_sys)
```

We can also run simulations using this system:

```@example ex1
odeprob = ODEProblem(simplified_sys, [], (0.0, 10.0))
odesol = solve(odeprob)
plot(odesol)
```

Once we've confirmed that our model works in a 0D "box model" setting,
we can expand it to 1, 2, or 3 dimensions using by adding in initial
and boundary conditions.
We will also add in advection using constant-velocity wind fields
add the same time.

```@example ex1
x_min = y_min = t_min = 0.0
x_max = y_max = t_max = 1.0
domain = DomainInfo(
    constIC(4.0, t ∈ Interval(t_min, t_max)),
    periodicBC(x ∈ Interval(x_min, x_max)),
    zerogradBC(y ∈ Interval(y_min, y_max))
)

sys_pde = couple(sys, domain, ConstantWind(t, 1.0, 1.0), Advection())

sys_pde_mtk = convert(PDESystem, sys_pde)
```

Now we can inspect this new system that we've created:

```@example ex1
sys_pde_mtk.dvs
```

```@example ex1
sys_pde_mtk.bcs
```

Finally, we can run a simulation using this system:

```@example ex1
discretization = MOLFiniteDifference([x => 10, y => 10], t, approx_order = 2)
@time pdeprob = discretize(sys_pde_mtk, discretization)
@time pdesol = solve(pdeprob, Tsit5(), saveat = 0.1)

# Plot the solution.
discrete_x, discrete_y, discrete_t = pdesol[x], pdesol[y], pdesol[t]
@variables Sys1₊c₁(..) Sys1₊c₂(..)
solc1, solc2 = pdesol[Sys1₊c₁(t, x, y)], pdesol[Sys1₊c₂(t, x, y)]
anim = @animate for k in 1:length(discrete_t)
    p1 = heatmap(solc1[k, 1:(end - 1), 1:(end - 1)],
        title = "c₁ t=\$(discrete_t[k])", clim = (0, 4.0), lab = :none)
    p2 = heatmap(solc2[k, 1:(end - 1), 1:(end - 1)],
        title = "c₂ t=\$(discrete_t[k])", clim = (0, 7.0), lab = :none)
    plot(p1, p2, layout = (1, 2), size = (800, 400))
end
gif(anim, fps = 8)
```

Because our system is a system of ordinary differential equations rather than partial differential equations, all of the grid cells in the animation above have the same value.
Refer to the [advection example](@ref Advection) for an example of a system of partial differential equations.
