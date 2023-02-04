```@meta
CurrentModule = EarthSciMLBase
```

# Initial and Boundary condition example

```@example
using EarthSciMLBase
using ModelingToolkit, DomainSets

# Set up ODE system
@parameters x y t
@variables u(t) v(t)
Dt = Differential(t)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

eqs = [
    Dt(u) ~ √abs(v),
    Dt(v) ~ √abs(u),
]

@named sys = ODESystem(eqs)

# Create constant initial conditions = 16.0 and boundary conditions = 4.0.
icbc = DomainInfo(
    constIC(4.0, t ∈ Interval(t_min, t_max)),
    constBC(16.0, 
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
    ),
)

# Convert to PDESystem and add constant initial and boundary conditions.
pdesys = sys + icbc

pdesys.bcs
```