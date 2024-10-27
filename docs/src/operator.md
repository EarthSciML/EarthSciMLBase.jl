# Operators for Large-scale 3D simulations

In this documentation so far, we have talked about creating systems of ordinary differential equations in ModelingToolkit and then converting them to systems of partial differential equations to perform 1-, 2-, or 3-dimensional simulations.
However, currently this does not work for large scale simulations.

While this ModelingToolkit functionality is being built, we have a different solution based on the [`Operator`](@ref) type in this package.
Using this system, we still define systems of ODEs to define behavior in a single grid cell, and we also have [`Operator`](@ref) processes that define behavior between grid cells.

## ODE System

As an example, let's first define a system of ODEs:

```@example sim
using EarthSciMLBase
using ModelingToolkit, DifferentialEquations
using SciMLOperators, Plots
using DomainSets
using ModelingToolkit: t_nounits, D_nounits
t = t_nounits
D = D_nounits

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 α = 10.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t) = 1.0
)

eqs = [D(u) ~ -α * √abs(v) + lon,
    D(v) ~ -α * √abs(u) + lat + 1e-14 * lev,
    windspeed ~ lat + lon + lev,
]
sys = ODESystem(eqs, t; name=:sys)
```

The equations above don't really have any physical meaning, but they include two state variables, some parameters, and a constant. 
There is also a variable `windspeed` which is "observed" based on the parameters, rather than being a state variable, which will be important later.

## Operator

Next, we define an operator. To do so, first we create a new type that is a subtype of [`Operator`](@ref):

```@example sim
mutable struct ExampleOp <: Operator
    α::Num # Multiplier from ODESystem
end
```
In the case above, we're setting up our operator so that it can hold a parameter from our ODE system.

Next, we need to define a method of `EarthSciMLBase.get_scimlop` for our operator. This method will be called to get a [`SciMLOperators.AbstractSciMLOperator`](https://docs.sciml.ai/SciMLOperators/stable/interface/) that will be used conjunction with the ModelingToolkit system above to integrate the simulation forward in time.

```@example sim
function EarthSciMLBase.get_scimlop(op::ExampleOp, mtk_sys, domain::DomainInfo, obs_functions, coordinate_transform_functions, u0, p)
    obs_f = obs_functions(op.α)
    grd = EarthSciMLBase.grid(domain)
    function run(du, u, p, t)
        u = reshape(u, size(u0)...)
        du = reshape(du, size(u0)...)
        for ix ∈ 1:size(u, 1)
            for (i, c1) ∈ enumerate(grd[1])
                for (j, c2) ∈ enumerate(grd[2])
                    for (k, c3) ∈ enumerate(grd[3])
                        # Demonstrate coordinate transforms
                        t1 = coordinate_transform_functions[1](t, c1, c2, c3)
                        t2 = coordinate_transform_functions[2](t, c1, c2, c3)
                        t3 = coordinate_transform_functions[3](t, c1, c2, c3)
                        # Demonstrate calculating observed value.
                        fv = obs_f(t, c1, c2, c3)
                        # Set derivative value.
                        du[ix, i, j, k] = (t1 + t2 + t3) * fv
                    end
                end
            end
        end
        nothing
    end
    FunctionOperator(run, u0[:], p=p)
end
```
The function above also doesn't have any physical meaning, but it demonstrates some functionality of the `Operator` "`s`".
First, it retrieves a function to get the current value of an observed variable in our
ODE system using the `obs_functions` argement, and it demonstrates how to call the resulting 
function to get that value.
It also demonstrates how to get coordinate transforms using the `coordinate_transform_functions` argument.
Coordinate transforms are discussed in more detail in the documentation for the [`DomainInfo`](@ref) type.

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
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
    grid_spacing = [0.1π, 0.1π, 1])
nothing #hide
```

Note that our domain includes a coordinate transform to convert from degrees latitude and longitude to meters.
Our domain specification also includes grid spacing the the `lon`, `lat`, and `lev`
coordinates, which we set as 0.1π, 0.1π, and 1, respectively.

!!! warning
    Initial and boundary conditions are not fully implemented for this case, so regardless
    of the conditions you specify, the initial conditions will be the default values
    of the variables in the ODE system, and the boundary conditions will be zero.

## Coupling and Running the Simulation

Next, initialize our operator, giving the the `windspeed` observed variable, and we can couple our ODESystem, Operator, and Domain together into a single model:

```@example sim
op = ExampleOp(sys.windspeed)

csys = couple(sys, op, domain)
```

Finally, we can choose a [`EarthSciMLBase.SolverStrategy`](@ref) and run the simulation.
We choose the [`SolverStrangThreads`](@ref) strategy, which needs us to 
specify an ODE solver from the [options available in DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) for both the MTK system.
We choose the `Tsit5` solver.
Then we create an [ODEProblem](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/) which can be used to run the simulation.
Finally, we solve the problem using the [solve](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/#CommonSolve.solve-Tuple%7BSciMLBase.AbstractDEProblem,%20Vararg%7BAny%7D%7D) function. 
At this point we need to choose a solver for the Operator part of the system, and we choose the `Euler` solver.
We also choose a splitting time step of 1.0 seconds, which we pass both to our `SolverStrangThreads` strategy and to the `solve` function.

```@example sim
dt = 1.0 # Splitting time step
st = SolverStrangThreads(Tsit5(), 1.0)

prob = ODEProblem(csys, st)
sol = solve(prob, Euler(); dt=1.0)
nothing #hide
```

After the simulation finishes, we can plot the result:

```@example sim
anim = @animate for i ∈ 1:length(sol.u)
    u = reshape(sol.u[i], :, size(domain)...)
    plot(
        heatmap(u[1, :, :, 1]),
        heatmap(u[1, :, :, 1]),
    )
end
gif(anim, fps = 15)
```