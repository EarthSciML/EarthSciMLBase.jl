# Coordinate Transforms for Partial Derivatives

Often, the coordinates of a grid may be defined in a different coordinate system than the one in which the partial derivatives are desired.
For example, grids are often defined in latitude and longitude, but partial derivatives may be required in units of meters to correspond with wind speeds in meters per second.

To handle this, the [`DomainInfo`](@ref) type can be used to define coordinate system transformations. To use it, a coordinate transform function first needs to be defined, for example [`partialderivatives_δxyδlonlat`](@ref) which transforms partial derivatives from longitude and latitude to meters:

```@example trans
using EarthSciMLBase
using ModelingToolkit
using DomainSets
using Unitful

@parameters lon [unit = u"rad"]
@parameters lat [unit = u"rad"]
@parameters lev

partialderivatives_δxyδlonlat([lev, lon, lat])
```

As you can see in the output of the code above, the function should take as arguments a list of the coordinates describing the grid (in the case above we have a 3-dimensional grid with vertical level, latitude, and longitude), and return a Dictionary relating the index of each coordinate with a factor to multiply the partial derivative by to convert it to the desired units.
The function only needs to return factors for the coordinates that are being transformed.

To include a coordinate transform in our domain, we include the function in the `DomainInfo` constructor:

```@example trans
deg2rad(x) = x * π / 180.0
domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(0.0, t ∈ Interval(0.0f0, 3600.0f0)),
    periodicBC(lat ∈ Interval(deg2rad(-90.0f0), deg2rad(90.0f0))),
    periodicBC(lon ∈ Interval(deg2rad(-180.0f0), deg2rad(180.0f0))),
    zerogradBC(lev ∈ Interval(1.0f0, 10.0f0)),
);
```

Multiple functions can be included in the `DomainInfo` constructor, just by including them as a vector, e.g.:

```julia
domain = DomainInfo(
    [transform1, transform2, ...],
    constIC(0.0, t ∈ Interval(0.0f0, 3600.0f0)),
    ...
)
```

After we include the coordinate transforms in our domain, in general everything should be handled automatically. 
The coordinate transforms may also be automatically added when different model components are coupled together, so you may not need to worry about them at all in many cases.
However, if you would like to use the transformed partial derivatives, for example to create a new PDE equation system, you can get them using the [`EarthSciMLBase.partialderivatives`](@ref) function:

```@example trans
δs = partialderivatives(domain)
```

This returns a list of functions, one corresponding to each coordinate in our domain.
Then we can calculate the symbolic partial derivative of a variable by just calling each function:

```@example trans
@variables u

[δs[i](u) for i ∈ eachindex(δs)]
```

You can see an example of how this is implemented in the source code for the [`Advection`](@ref) model component.

Additional transformation functions may be defined in other packages.
We recommend that the names of these functions start with `partialderivatives_` to make it clear that they are intended to be used in this context.