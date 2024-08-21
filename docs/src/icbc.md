# Initial and Boundary conditions

Oftentimes we will want to do a 1, 2, or 3-dimensional simulation, rather than the 0-dimensional simulation we get by default with a system of ordinary differential equations.
In these cases, we will need to specify initial and boundary conditions for the system.

To demonstrate how to do this, we will use the following simple system of ordinary differential equations:

```@example icbc
using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t_nounits, D_nounits
t = t_nounits
D = D_nounits

@parameters x y

function ExampleSys()
    @variables u(t) v(t)
    eqs = [
        D(u) ~ √abs(v),
        D(v) ~ √abs(u),
    ]
    ODESystem(eqs, t; name=:Docs₊Example)
end

ExampleSys()
```
Next, we specify our initial and boundary conditions using the [`DomainInfo`](@ref) type.
We initialize [`DomainInfo`](@ref) with sets of initial and boundary conditions.
In the example below, we set constant initial conditions using [`constIC`](@ref) and constant boundary conditions using [`constBC`](@ref).

```@example icbc
using DomainSets

x_min = y_min = t_min = 0.0
x_max = y_max = t_max = 1.0

# Create constant initial conditions = 16.0 and boundary conditions = 4.0.
icbc = DomainInfo(
    constIC(4.0, t ∈ Interval(t_min, t_max)),
    constBC(16.0,
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
    ),
)
nothing # hide
```

It is also possible to use periodic boundary conditions with [`periodicBC`](@ref) and zero-gradient boundary conditions with [`zerogradBC`](@ref).

Finally, we combine our initial and boundary conditions with our system of equations using the [`couple`](@ref) function.

```@example icbc
model = couple(ExampleSys(), icbc)

eq_sys = convert(PDESystem, model)
```

We can also look at the expanded boundary conditions of the resulting equation system:

```@example icbc
eq_sys.bcs
```