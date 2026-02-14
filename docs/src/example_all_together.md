```@meta
CurrentModule = EarthSciMLBase
```

# Example using different components of EarthSciMLBase together

This example shows how to define and couple different components of EarthSciMLBase together to create a more complex model. First, we create several model components.

Our first example system is a simple reaction system:

```@example ex1
using EarthSciMLBase
using ModelingToolkit, Catalyst, DomainSets, DifferentialEquations
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
    ode_model(complete(rs); metadata = Dict(CoupleType => ExampleSys1Coupler))
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
        metadata = Dict(CoupleType => ExampleSys2Coupler))
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

!!! note

    This model can also be expanded to 1, 2, or 3 dimensions by adding initial and boundary conditions, advection, etc. See the [advection example](@ref Advection) for more details. Discretization and numerical solution of PDE systems requires [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/), which is not currently compatible with the latest ModelingToolkit ecosystem.
