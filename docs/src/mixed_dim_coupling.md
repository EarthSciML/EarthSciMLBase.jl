# Mixed-Dimension Coupling

When building Earth science models, it is common to couple systems that operate
in different numbers of spatial dimensions. For example, a 2D surface fire-spread
model may need meteorological data from a 3D atmospheric data source that
includes vertical pressure levels. EarthSciMLBase supports this through
*per-system DomainInfo*, which allows individual ODE systems to carry their own
[`DomainInfo`](@ref) and be promoted to PDESystems with only the spatial
dimensions they need.

## The Problem

By default, all ODE systems in a [`CoupledSystem`](@ref) share a single
[`DomainInfo`](@ref). When the coupled system is converted to a `PDESystem`,
every ODE system is promoted using that shared DomainInfo, meaning every variable
gets the same spatial dimensions.

This creates a conflict when systems need different numbers of dimensions:
- A 2D surface model needs `{t, x, y}`
- A 3D data source needs `{t, x, y, lev}`

If the shared DomainInfo is 2D, the 3D data source loses its vertical dimension.
If it is 3D, the 2D model gets an unwanted `lev` dimension that doesn't appear in
the original PDE system.

## Solution: `SysDomainInfo` Metadata

Systems that need a different DomainInfo than the default can declare their own
by adding a [`SysDomainInfo`](@ref) entry to their metadata:

```@example mixed_dim
using ModelingToolkit, EarthSciMLBase, DomainSets
using ModelingToolkit: t_nounits, D_nounits
t = t_nounits
D = D_nounits

@parameters x y z

# 2D DomainInfo (default for the coupled system)
domain_2d = DomainInfo(
    constIC(0.0, t ∈ Interval(0.0, 1.0)),
    constBC(0.0, x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0))
)

# 3D DomainInfo (for a system that needs a vertical dimension)
domain_3d = DomainInfo(
    constIC(0.0, t ∈ Interval(0.0, 1.0)),
    constBC(0.0, x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
            z ∈ Interval(0.0, 1.0))
)

nothing # hide
```

An ODE system declares its own DomainInfo by including `SysDomainInfo => domaininfo`
in the metadata dictionary alongside the usual `CoupleType`:

```julia
# Example: a data source that needs 3 spatial dimensions
System([D(v) ~ p_v], t; name = :data3d,
    metadata = Dict(
        SysDomainInfo => domain_3d,
        CoupleType => MyDataCoupler,
    ))
```

## Cross-Dimension Coupling Example

Here is a complete example where a 2D PDE system receives forcing from a 3D
ODE data source at ground level (`z=0`).

First, define coupler types and the systems:

```@example mixed_dim
struct SurfaceCoupler
    sys
end
struct DataSource3DCoupler
    sys
end

@variables u(..)
Dx = Differential(x)

# A 2D PDE system with CoupleType metadata
pde_2d = PDESystem(
    [D(u(t, x, y)) ~ Dx(Dx(u(t, x, y)))],
    [u(0, x, y) ~ 1.0,
     u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
     u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0],
    [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
    [t, x, y], [u(t, x, y)], [];
    name = :surface,
    metadata = Dict(CoupleType => SurfaceCoupler)
)

# A 3D ODE system with its own DomainInfo and CoupleType
@variables v(t) = 0.0
@parameters p_v = 3.0
ode_3d = System([D(v) ~ p_v], t; name = :data3d,
    metadata = Dict(
        SysDomainInfo => domain_3d,
        CoupleType => DataSource3DCoupler,
    ))
nothing # hide
```

Define how the 2D and 3D systems couple. The `EarthSciMLBase.couple2` method
receives the promoted PDESystems, so we extract the 3D dependent variable and use
[`slice_variable`](@ref) to fix the vertical coordinate at ground level (`z=0`):

```@example mixed_dim
function EarthSciMLBase.couple2(s::SurfaceCoupler, d::DataSource3DCoupler)
    a_sys, b_sys = s.sys, d.sys
    # Find the 3D dependent variable from the promoted data source system.
    # After promotion, the variable name includes the system prefix (e.g., "data3d₊v").
    b_v = first(filter(dv -> occursin("v", string(dv)), b_sys.dvs))
    # Slice at z=0 to create a 2D variable and a defining equation.
    sliced_v, slice_eq = slice_variable(b_v, z, 0.0)
    # Add the sliced variable as a forcing term to u's equation.
    coupling_eqs = [
        D(u(t, x, y)) ~ sliced_v,
        slice_eq,
    ]
    ConnectorSystem(coupling_eqs, a_sys, b_sys)
end
nothing # hide
```

Couple and convert. The 2D PDE system, 3D ODE data source, and default 2D
domain are all passed to [`couple`](@ref):

```@example mixed_dim
cs = couple(pde_2d, ode_3d, domain_2d)
merged = convert(PDESystem, cs)
```

The merged system has the union of all independent variables (`{t, x, y, z}`),
but each dependent variable retains only its own dimensions:

```@example mixed_dim
for dv in merged.dvs
    println(dv)
end
```

The equations include the original diffusion equation with the coupling term
added, plus a slice equation extracting the 3D variable at `z=0`:

```@example mixed_dim
for eq in equations(merged)
    println(eq)
end
```

## How It Works Internally

The mixed-dimension coupling mechanism works through these steps:

1. **Grouping**: ODE systems are partitioned by their effective [`DomainInfo`](@ref).
   Systems with [`SysDomainInfo`](@ref) metadata use their own; others use the
   [`CoupledSystem`](@ref)'s default.

2. **Same-group ODE coupling**: Within each group, `EarthSciMLBase.couple2` methods
   run as normal, because all systems share the same dimensions.

3. **Individual promotion**: Each group is composed into a flat `System` and
   promoted to a `PDESystem` via `system + domaininfo`. Metadata (including
   [`CoupleType`](@ref)) is preserved through this promotion.

4. **Cross-group PDE coupling**: After promotion, `EarthSciMLBase.couple2` methods
   are checked between all `PDESystem` pairs (including across groups). These
   methods can handle dimension mismatches using tools like
   [`slice_variable`](@ref).

5. **Merging**: [`merge_pdesystems`](@ref) computes the union of all
   independent variables and domains, keeping each dependent variable at its
   original dimensionality.
