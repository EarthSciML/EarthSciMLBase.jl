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
using Plots
using DomainSets
using DynamicQuantities
using ModelingToolkit: t_nounits, D_nounits
t = t_nounits
D = D_nounits

@parameters y lon=0.0 lat=0.0 lev=1.0 t α=10.0
@constants p = 1.0
@variables(u(t)=1.0, v(t)=1.0, x(t), [unit=u"1/m"], y(t), [unit=u"1/m"],
    z(t), windspeed(t))
Dt = Differential(t)

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lat + lon + lev,
    x ~ 1.0 / EarthSciMLBase.lon2meters(lat),
    y ~ 1.0 / EarthSciMLBase.lat2meters,
    z ~ 1.0 / lev
]
sys = System(eqs, t; name = :sys)
```

The equations above don't really have any physical meaning, but they include two state variables, some parameters, and a constant.
There is also a variable `windspeed` which is "observed" based on the parameters, rather than being a state variable, which will be important later.

## Operator

Next, we define an operator. To do so, first we create a new type that is a subtype of [`Operator`](@ref):

```@example sim
struct ExampleOp <: Operator
end
```

Next, we need to define a method of `EarthSciMLBase.get_odefunction` for our operator. This method will be called to get a function that can be used as an ODE function, i.e. it should have
methods `f(u, p, t)` and optionally `f(du, u, p, t)` where `u` is a state vector, `p` is parameters,
`t` is time, and `du` is a cache for the result of the function.
For more information, see [here](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/).

```@example sim
function EarthSciMLBase.get_odefunction(
        op::ExampleOp, csys::CoupledSystem, mtk_sys, coord_args,
        domain::DomainInfo, u0, p, alg::MapAlgorithm)
    α, trans1, trans2, trans3 = EarthSciMLBase.get_needed_vars(op, csys, mtk_sys, domain)

    obs_f = EarthSciMLBase.build_coord_observed_function(mtk_sys, coord_args,
        [α, trans1, trans2, trans3])

    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = EarthSciMLBase.grid(domain)
    obscache = zeros(EarthSciMLBase.dtype(domain), 4)
    sz = length.(EarthSciMLBase.grid(domain))

    function run(du, u, p, t) # In-place
        u = reshape(u, :, sz...)
        du = reshape(du, :, sz...)
        II = CartesianIndices(tuple(sz...))
        for ix in 1:size(u, 1)
            for I in II
                # Demonstrate coordinate transforms and observed values
                obs_f(obscache, view(u, :, I), p, t, c1[I[1]], c2[I[2]], c3[I[3]])
                t1, t2, t3, fv = obscache
                # Set derivative value.
                du[ix, I] = (t1 + t2 + t3) * fv
            end
        end
        nothing
    end
    function run(u, p, t) # Out-of-place
        u = reshape(u, :, sz...)
        II = CartesianIndices(size(u)[2:end])
        du = vcat([begin
                       t1, t2,
                       t3, fv = obs_f(view(u, :, I), p, t, c1[I[1]], c2[I[2]], c3[I[3]])
                       (t1 + t2 + t3) * fv
                   end
                   for ix in 1:size(u, 1), I in II]...)
        reshape(du, :)
    end
    return run
end
nothing
```

The function above also doesn't have any physical meaning, but it demonstrates some functionality of the `Operator` "`s`".
First, it retrieves a function to get the current value of an observed variable in our
ODE system using the `obs_functions` argument, and it demonstrates how to call the resulting
function to get that value.
It also demonstrates how to get coordinate transforms using the `coordinate_transform_functions` argument.
Coordinate transforms are discussed in more detail in the documentation for the [`DomainInfo`](@ref) type.

We also need to define a method of `EarthSciMLBase.get_needed_vars`, which will return which variables are needed by the operator.

```@example sim
function EarthSciMLBase.get_needed_vars(::ExampleOp, csys, mtk_sys, domain::DomainInfo)
    return [mtk_sys.sys₊windspeed, mtk_sys.sys₊x, mtk_sys.sys₊y, mtk_sys.sys₊z]
end
nothing
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

Next, initialize our operator, giving the the `windspeed` observed variable, and we can couple our System, Operator, and Domain together into a single model:

```@example sim
op = ExampleOp()

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
sol = solve(prob, Euler(); dt = 1.0)
nothing #hide
```

After the simulation finishes, we can plot the result:

```@example sim
anim = @animate for i in 1:length(sol.u)
    u = reshape(sol.u[i], :, size(domain)...)
    plot(
        heatmap(u[1, :, :, 1]),
        heatmap(u[1, :, :, 1])
    )
end
gif(anim, fps = 15)
```
