# Simulator for large-scale 3D simulations

In this documentation so far, we have talked about creating systems of ordinary differential equations in ModelingToolkit and then converting them to systems of partial differential equations to perform 1-, 2-, or 3-dimensional simulations.
However, currently this does not work for large scale simulations.

While this ModelingToolkit functionality is being built, we have a different solution based on the [`Simulator`](@ref) type in this package.
Using this system, we still define systems of ODEs to define behavior in a single grid cell, and we also have [`Operator`](@ref) processes that define behavior between grid cells.
The [`Simulator`](@ref) then integrates the ODEs and the Operators together using [Strang Splitting](https://en.wikipedia.org/wiki/Strang_splitting).

## ODE System

As an example, let's first define a system of ODEs:

```@example sim
using Main.EarthSciMLBase
using ModelingToolkit, DomainSets, DifferentialEquations
using Plots

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
lat = GlobalScope(lat)
lon = GlobalScope(lon)
lev = GlobalScope(lev)
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t) = 1.0
)
Dt = Differential(t)

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat,
    windspeed ~ lat + lon + lev,
]
sys = ODESystem(eqs, t; name=:Docs₊sys)
```

The equations above don't really have any physical meaning, but they include two state variables, some parameters, and a constant. 
There is also a variable `windspeed` which is "observed" based on the parameters, rather than being a state variable, which will be important later.
Also, note that we have the coordinates `lat`, `lon`, and `lev` (for latitude, longitude, and vertical level number), which we define as global variables using for example `lat = GlobalScope(lat)`.
This is important for things to work correctly below.

## Operator

Next, we define an operator. To do so, first we create a new type that is a subtype of `Oerator`:

```@example sim
mutable struct ExampleOp <: Operator
    α::Num # Multiplier from ODESystem
end
```
In the case above, we're setting up our operator so that it can hold a parameter from our ODE system.

Next, we need to define a method of [`EarthSciMLBase.run!`](@ref) for our operator. This method will be called by the simulator to calculate the effect of the operator on the state variables.

```@example sim
function EarthSciMLBase.run!(op::ExampleOp, s::Simulator, t)
    f = s.obs_fs[s.obs_fs_idx[op.α]]
    for ix ∈ 1:size(s.u, 1)
        for (i, c1) ∈ enumerate(s.grid[1])
            for (j, c2) ∈ enumerate(s.grid[2])
                for (k, c3) ∈ enumerate(s.grid[3])
                    # Demonstrate coordinate transforms
                    t1 = s.tf_fs[1](t, c1, c2, c3)
                    t2 = s.tf_fs[2](t, c1, c2, c3)
                    t3 = s.tf_fs[3](t, c1, c2, c3)
                    # Demonstrate calculating observed value.
                    fv = f(t, c1, c2, c3)
                    # Set derivative value.
                    s.du[ix, i, j, k] = (t1 + t2 + t3) * fv / 10
                end
            end
        end
    end
end
```
The function above also doesn't have any physical meaning, but it demonstrates some functionality of the `Simulator` "`s`".
First, it retrieves a function to get the current value of an observed variable in our
ODE system using the `s.obs_fs` field, and it demonstrates how to call the resulting 
function to get that value.
It also demonstrates how to get coordinate transforms using the `s.tf_fs` field.
Coordinate transforms are discussed in more detail in the documentation for the [`DomainInfo`](@ref) type.

Finally, we define a method of [`EarthSciMLBase.timestep`](@ref) for our operator. This method will be called by the simulator to determine the timestep for the operator.
```@example sim
EarthSciMLBase.timestep(op::ExampleOp) = 1.0
```

## Domain

Once we have an ODE system and an operator, the final component we need is a domain to run the simulation on.
Defining a domain is covered in more depth in the documentation for the [`DomainInfo`](@ref) type, but for now we'll just define a simple domain:

```@example sim
t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))
nothing #hide
```

Note that our domain includes a coordinate transform to convert from degrees latitude and longitude to meters.

!!! warning
    Initial and boundary conditions are not fully implemented for the `Simulator`, so regardless
    of the conditions you specify, the initial conditions will be the default values
    of the variables in the ODE system, and the boundary conditions will be zero.

## Coupling and Running the Simulator

Next, initialize our operator, giving the the `windspeed` observed variable, and we can couple our ODESystem, Operator, and Domain together into a single model:

```@example sim
op = ExampleOp(sys.windspeed)

csys = couple(sys, op, domain)
nothing #hide
```

...and then create a Simulator. 
Our simulator specification needs to include grid spacing the the `lon`, `lat`, and `lev`
coordinates, which we set as 0.1π, 0.1π, and 1, respectively, and it needs us to 
chose an ODE solver from the [options available in DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).
We choose the `Tsit5` solver:

```@example sim
sim = Simulator(csys, [0.1π, 0.1π, 1], Tsit5())
nothing #hide
```

Finally, we can run the simulation:

```@example sim
run!(sim)
```

...and plot the result at the end of the simulation:

```@example sim
plot(
    heatmap(sim.u[1, :, :, 1]),
    heatmap(sim.u[1, :, :, 1]),
)
```